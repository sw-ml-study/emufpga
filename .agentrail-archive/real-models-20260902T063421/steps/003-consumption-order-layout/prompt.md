Lay the TRM weights out in CONSUMPTION ORDER, and prove a forward
sweep never seeks.

WHY THIS STEP EXISTS. Step 002's extractor emitted tensors with
Python's `sorted()`, which is alphabetical. Reading TRM's own trm.py
shows the execution order per layer is
`qkv_proj -> o_proj -> gate_up_proj -> down_proj`, while alphabetical
gives `down_proj, gate_up_proj, o_proj, qkv_proj` -- exactly reversed.
A single forward sweep of that file would have to seek backward, which
is the one thing this architecture forbids. docs/research.txt's
instruction is "arrange weights physically in exactly the order the
tensor engine consumes them", and step 002 did not.

THE EXECUTION ORDER, read from trm.py rather than guessed:

    forward:
      for _ in range(H_cycles - 1):        # 2
        for _ in range(L_cycles):          # 4
          z_L = L_level(z_L, z_H + input)
        z_H = L_level(z_H, z_L)
      for _ in range(L_cycles):            # 4
        z_L = L_level(z_L, z_H + input)
      z_H = L_level(z_H, z_L)

    L_level(hidden, injection):
      hidden = hidden + injection
      for layer in layers:                 # 2
        hidden = rms_norm(hidden + self_attn(hidden))
        hidden = rms_norm(hidden + mlp(hidden))

So 15 `L_level` calls per forward, each sweeping the same 8 matrices:

    layers.0.self_attn.qkv_proj
    layers.0.self_attn.o_proj
    layers.0.mlp.gate_up_proj
    layers.0.mlp.down_proj
    layers.1.self_attn.qkv_proj
    layers.1.self_attn.o_proj
    layers.1.mlp.gate_up_proj
    layers.1.mlp.down_proj

That run of eight is the ROTATING REGION -- the thing swept and
rewound 15 times per forward, and up to 240 times per puzzle at
halt_max_steps 16. It must be contiguous in the file, because a rewind
returns to the start of the stream and anything sitting in the middle
of that run would be re-read for nothing.

The remaining seven tensors are one-shot: H_init, L_init,
embed_tokens, puzzle_emb, lm_head, q_head.weight, q_head.bias. They
are read once per forward, not per cycle.

WORK:

1. An explicit order specification, not a heuristic. A text file
   listing tensor names one per line, in consumption order; the
   importer emits streams in that order. A name in the file but not in
   the checkpoint is an error, and so is the reverse -- silence there
   is how a layout drifts from the model.

2. Ship the TRM order as a tracked file. It is small text, it is the
   layout contract for this model, and it belongs in the repository
   next to the code that reads it.

3. Make the one-shot tensors and the rotating region distinguishable.
   Record where the rotating region begins and ends, so a consumer can
   rewind to the right place rather than to byte zero. Say how you
   represent it: a sidecar field is fine, a format change is not.

4. Prove it. A test that walks the stream directory in order and
   asserts the sequence a forward pass needs is exactly the sequence
   the file provides, with no index ever going backward. Do this
   against the REAL order file, hermetically -- the order file is
   tracked, so no download is needed to test the ordering even though
   testing the weights would need one.

Do NOT implement attention, RMSNorm, SwiGLU or the recursion
arithmetic yet. This step is about layout and the no-seek property.
The operators are next, and they will be much easier to trust once the
weights arrive in the right order.

Gates as always. `just check` green, sw-checklist 0 failed and no new
warnings, no `#[allow]`, no file over 1 MiB.

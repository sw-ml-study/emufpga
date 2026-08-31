Run TRM's forward pass over the streamed weights.

First rung of the model ladder: 7M now, then 27M (HRM), then 120M,
then 1 GB. Prove the machinery here, where every number is checkable
by hand, before anything larger.

The weights are already imported and laid out in consumption order
(saga 2 steps 2-3). This step makes them compute.

WHAT TRM NEEDS, read from the checkpoint's own trm.py:

    L_level(hidden, injection):
      hidden = hidden + injection
      for layer in layers:                    # 2
        hidden = rms_norm(hidden + self_attn(hidden), eps=1e-5)
        hidden = rms_norm(hidden + mlp(hidden), eps=1e-5)

    self_attn: qkv_proj -> RoPE -> softmax attention -> o_proj
               8 heads, head_dim 64, hidden 512
    mlp:       SwiGLU -- gate_up_proj splits into gate and up,
               silu(gate) * up, then down_proj

    rms_norm here has NO learned weight. It is pure normalization,
    which is why the checkpoint contains no norm tensors. Do not invent
    one.

DIVIDE THE WORK THE WAY THE ARCHITECTURE DOES. The four projections
per layer are the streamed part -- they are the rotating region and
99.87% of the weights. Everything else (RMSNorm, RoPE, softmax, SiLU,
the residuals) runs on resident activations, which docs/plan.md
section 3 explicitly allows in ordinary memory. Do not stream
activations and do not try to make the elementwise ops sequential;
that is not what the restriction is about.

So the new work splits cleanly:

1. A streamed linear: consume one `.spm` stream and compute `y = Wx`
   for a batch of activation vectors. This is `spm-gemv-ref`
   generalized -- it currently runs only the first stream in a file
   and only ternary. It needs to run stream N of M, and f32.
2. The activation-side operators, conventional and resident.
3. The recursion driving `rewind()` between L_level calls.

VALIDATE AGAINST A RESIDENT REFERENCE, NOT PYTORCH. Implement each
matmul twice: once reading weights from a resident array, once
streaming from the `.spm`. They must agree BIT-EXACTLY, for the same
reason as the fabric model -- consecutive weights land on different
accumulators so no summation is reordered. Torch is not installed and
is not the point here: this step proves the streaming path matches a
conventional path on identical weights. Comparing against the
published model is the NEXT step and needs torch.

Measure, and record in docs/results.md:
- weights streamed per forward pass, and how many times each is read
- `Ps` with the recursion counted, not just the batch. TRM re-reads
  its rotating region 15 times per forward; the current metric cannot
  see that and will under-report reuse by 15x.
- bytes read versus bytes resident

Hermetic tests as always -- synthetic weights of TRM's shapes, not the
27 MB download. The real checkpoint is a manual verification recorded
in docs/.

Gates as always. No file over 1 MiB.

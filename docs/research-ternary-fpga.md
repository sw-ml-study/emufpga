# Would a ternary ISA on an FPGA help the serial approach?

A research note, not a plan. Everything here is arithmetic over
measurements this project already took, plus projections that are
labelled as projections. Nothing has been run on hardware and no
ternary model has been run at all.

The short answer: **yes, and for a better reason than the obvious
one.** Ternary is usually argued for on storage grounds -- 16x smaller
than f32. That argument alone would not help here, because this
project has measured twice that it is compute-bound rather than
storage-bound. Ternary helps because it attacks **both** terms of the
ratio at once.

## 1. Why "smaller weights" would not be enough

Two independent measurements say bytes are not the scarce resource:

- The CPU reference engine is ~196x too slow to saturate even a
  page-cached file read (saga 1 step 6).
- The fabric model puts the store-bound/compute-bound crossing between
  4 and 16 bytes per cycle for an 8-lane datapath (saga 1 step 9).

A change that only shrinks the parameter stream moves the wrong term.
The engine would still be waiting on arithmetic.

## 2. What ternary actually changes

Arithmetic intensity, in MACs per weight-byte, is the number that
matters:

| encoding | MACs per weight-byte |
| --- | --- |
| f32 | `batch * 0.25` |
| bf16 | `batch * 0.50` |
| **ternary, 2 bits** | **`batch * 4.00`** |

Sixteen times better than f32 -- but that is only half of it. The
other half is that **a ternary weight does not need a multiplier**:

```
+1 -> ADD the activation
 0 -> BYPASS
-1 -> SUBTRACT the activation
```

`mlpl/spm_stream.mlpl` demonstrates this executably: the same answer,
`5 6 6`, with no multiply anywhere in the sweep.

That is the synergy. Fewer bytes per weight means a given store feeds
more lanes; cheaper arithmetic per weight means more lanes fit in the
fabric. Both terms move the same way.

The fabric model's feeding requirement makes it concrete:

| lanes | f32 | bf16 | ternary |
| ---: | ---: | ---: | ---: |
| 8 | 32 B/cyc | 16 B/cyc | **2 B/cyc** |
| 64 | 256 B/cyc | 128 B/cyc | **16 B/cyc** |
| 256 | 1024 B/cyc | 512 B/cyc | **64 B/cyc** |

Saga 1 step 9 measured 0.889 occupancy at 16 bytes per cycle with 8
lanes. At ternary, 16 bytes per cycle feeds **64** lanes. The same
modest store supports an eight times wider datapath.

## 3. "A LUT implements a trit" understates it

Storing a trit is two bits -- two flip-flops or two bits of block RAM,
not a lookup table. Framing it as storage misses where the win is.

The win is that the **operation** collapses. An f32 multiply-
accumulate wants a hardware multiplier; a ternary one wants an adder
with a sign select and an enable, which maps onto the carry chain
rather than onto a DSP block. On a part with few multipliers and many
LUTs, that is the difference between a handful of lanes and many.

The encoding this project already uses is built for exactly that:

```
bit 0  NONZERO_BIT   accumulator enable
bit 1  NEGATIVE_BIT  add/subtract select
```

Those are control lines, not operands. The parameter stream is not
data that gets multiplied -- **it is an instruction stream two bits
wide, and the program is the model.**

**An open question worth recording:** `0b10` is "negative but not
nonzero", which is permanently invalid. A two-bit ISA with a spare
code point is unusual. It could encode a run-length skip for sparse
regions, a group boundary, or an end-of-stream marker -- each of which
would change the fetch pattern rather than the arithmetic. Nobody has
looked at whether that is worth anything.

**Unverified:** the specific LUT, DSP and BRAM counts of the Tang Nano
9K's GW1NR-9 are still marked UNVERIFIED in `docs/plan.md` section 6.
The argument above is about the *shape* of the trade and does not
depend on the exact numbers, but no lane count should be claimed until
a real place-and-route contradicts or confirms it.

## 4. Where the PC/FPGA boundary must fall

This is the part that decides whether the aux-processor idea works at
all, and the measurements answer it cleanly.

**Weights must never cross the host link.** If parameters travel
PC -> USB -> FPGA, the PC has already touched every byte, and nothing
has been gained over computing on the PC. Worse, the link becomes the
store: over a ~35 MB/s USB 2.0 path, one sweep of SmolLM2-135M costs
6.07 s at bf16 and 0.76 s at ternary -- 1.3 tokens/s at best. The
FPGA needs its **own** path to the parameter store: SPI-NOR, SD, or a
dedicated link.

**Activations may cross freely.** They are small enough that the link
is irrelevant:

| what crosses | per token, 5 clients |
| --- | ---: |
| hidden state, once per token | 11.5 KB |
| hidden state, per layer, both ways | 0.69 MB |

Even the pessimistic per-layer round trip is 0.02 s per token over
USB 2.0.

### The partition falls exactly where step 11 already found it

The serving measurement split a decode step in two, and the split maps
onto the PC/FPGA boundary without being designed to:

| | scales with clients? | carries weights? | belongs |
| --- | --- | --- | --- |
| the streamed matmuls | no -- one sweep serves all | yes | **FPGA** |
| attention and its KV cache | yes, linearly | no | **PC RAM** |

That is not a coincidence. The reason attention does not amortize is
the same reason it has no weights, and the reason the matmuls amortize
is that their weights are shared. **The architecture's own asymmetry
is the partition.**

It also resolves a problem the FPGA could not otherwise solve. The KV
cache for five clients at 512 context is 118 MB (saga 2 step 11) --
far beyond any small FPGA's block RAM. It does not need to be there.
It belongs on the side that has cheap RAM, and only the hidden state
crosses.

## 5. What an MLPL-like description would be for

Not as an ISA. As a **golden model**.

This project's postmortems keep returning to one rule: a decision made
in software must match a decision made in silicon, and pure decision
functions are the ones that can be golden-tested against RTL. A
ternary datapath is almost entirely decision -- decode two bits,
select add or subtract, enable or bypass.

`mlpl/spm_stream.mlpl` already contains that specification in about a
dozen lines, and it executes. The same file could be the reference
that the Rust engine and eventual RTL are both checked against, which
is exactly the seam `docs/research2.txt` proposes between the two
repositories.

## 6. What would kill this

- **No ternary model has been run here at all.** `spm-codec-any`
  refuses the profile explicitly rather than supporting it. Every
  number above is arithmetic, not measurement.
- **Post-training ternary quantization would destroy a small model.**
  The 1.58-bit results that motivate this are either trained ternary
  (BitNet-style) or quantizations of very large models whose
  redundancy absorbs the loss. SmolLM2-135M quantized to ternary after
  the fact would almost certainly be unusable, and nothing here has
  tested that. **The right experiment is a model trained ternary, not
  a model crushed into it.**
- **The activation side can still blow up.** BDH (saga 2 step 7) has a
  working set that exceeds its entire weight set at long sequences.
  Shrinking weights 16x does nothing about that, and for a device
  whose fast memory is measured in kilobytes it is the binding
  constraint.
- **The interconnect could still eat it.** The partition above assumes
  the FPGA reads parameters independently. If that path does not exist
  on the boards on hand, the idea is untestable as stated regardless
  of how good the arithmetic looks.

## 7. The experiment that would settle it

In order, each cheap and each able to falsify the next:

1. **Run ternary end to end in the existing engine.** Take a trained
   ternary model, import it, and check it against its reference the
   way every other rung was checked. This needs a scale-aware
   accumulator the streamed path does not have. Until it exists,
   everything above is projection.
2. **Re-measure the serving amortization at ternary.** The prediction
   is 5.31 MB per generated token at five clients, against 42.5 MB at
   bf16. If it does not land there, the model of the traffic is wrong.
3. **Only then** ask what fits in a real part, with a real
   place-and-route, against the device profiles that
   `docs/plan.md` section 6 still marks UNVERIFIED.

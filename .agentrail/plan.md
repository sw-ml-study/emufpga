# SPM Walking Skeleton

Vision: prove the Serial Parameter Machine vertical slice end to end on
synthetic ternary matrices -- a `.spm` physical execution layout, a
seek-free weight stream, a multiplier-free ternary GEMV reference, the
batch-amortization crossover measurement, and a Gowin/Tang Nano resource
fit report -- so that later sagas, hand-written RTL, and the rack-Linux
and RP2350 fronts all share one golden model. Full detail in docs/plan.md.

1. **repo-scaffold** -- components/ workspace skeleton, scripts/ build and
   gate entry points, justfile delegation, shared target/ via
   .cargo/config.toml, CLAUDE.md with the agentrail block, code_metrics
   doc, recorded sw-checklist baseline.
2. **spm-format** -- components/format/: spm-header, spm-codec, spm-layout,
   spm-file. Ternary 2-bit packing, per-group scales, stream directory,
   op descriptors, byte layout pinned by a golden fixture.
3. **spm-stream** -- components/stream/: WeightStream trait with no seek in
   its surface, memory and file impls, and metrics for bandwidth, eta,
   scan productivity Ps and residency ratio Rp.
4. **spm-tensor-ref** -- components/tensor/: batched accumulator banks,
   multiplier-free ternary GEMV over a WeightStream, f32 reference matmul,
   error metrics, golden vector generation.
5. **spm-pack-cli** -- components/cli/: `emufpga pack` converts a dense
   matrix to ternary .spm in consumption order; CLI help/version passes
   sw-checklist validation.
6. **batch-amortization-bench** -- `emufpga bench --batch 1,2,4,8,16,32`.
   The make-or-break measurement: find the crossover where compute, not
   storage, becomes limiting. Record in docs/results.md.
7. **gowin-device-profiles** -- components/device/: the five Tang Nano
   boards, every figure cited from Project Apicula or a Gowin datasheet,
   unsourceable figures recorded as unknown rather than guessed.
8. **resource-budget-and-fit** -- gowin-budget and gowin-timing; `emufpga
   fit` reports LUT4/BSRAM/DSP/IO utilization plus predicted cycles and
   throughput per board, with every assumption written into
   docs/fit-model.md so a later real place-and-route can contradict it.
9. **saga-1-wrapup** -- README results table, docs/architecture.md,
   docs/spm-format.md, final sw-checklist counts, saga 2 defined.

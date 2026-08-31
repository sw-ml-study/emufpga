# components/device

Gowin device profiles for the five Tang Nano boards on hand.

| Crate | Responsibility |
| --- | --- |
| `gowin-profile` | the five board profiles, every figure cited |

`gowin-budget` and `gowin-timing` arrive in step 8.

## Provenance is part of the data

Every number carries the document it came from, in the code rather
than only in a doc. `Figure` is deliberately not `Option<u32>`: an
`Option` invites `unwrap_or(0)`, and a zero LUT4 count that reads as a
number is exactly the failure this type exists to prevent. An
`Unknown` carries a note saying what was tried.

## The two gaps that matter

**Bulk memory bandwidth is unknown for every board**, and it is the
figure the whole architecture turns on. Step 6 measured the CPU
reference as ~196x too slow to saturate a page-cached read
(../../docs/results.md); which side of that a board lands on depends
on what its PSRAM or SDRAM sustains, which board documentation does
not state.

**Achievable fabric fmax is unknown for every board.** Datasheets give
per-primitive timing; the honest source is a real place-and-route.

A test asserts both stay unknown. If either is ever sourced, that test
fails and tells you to revisit the fit model rather than letting a new
number silently change results.

Built by saga 1 step 7 (gowin-device-profiles).

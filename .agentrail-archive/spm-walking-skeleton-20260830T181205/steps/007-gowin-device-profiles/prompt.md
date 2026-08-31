Build `components/device/` -- Gowin device profiles for the five Tang
Nano boards on hand. Read docs/plan.md section 6 and
docs/code_metrics.md first.

Gates: 25 LOC/function, **4** functions/module, **4** modules/crate
(lib.rs counts), 350 LOC/file. Only `src/` is measured. Let the module
ceiling decide the crate count.

THE POINT OF THIS STEP IS SOURCING, NOT CODING. docs/plan.md section 6
currently carries a table of figures marked UNVERIFIED, which I wrote
from memory. Replace every one of them with a value you can cite, or
record it as unknown. Do not let a plausible number survive because it
looks right.

For each board -- Tang Nano 1K (GW1NZ-1), 4K (GW1NSR-4C), 9K
(GW1NR-9), 20K (GW2AR-18), 25K (GW5A-25) -- a profile records at
minimum:

  - LUT4 count
  - flip-flop count
  - B-SRAM block count and bits per block
  - DSP block count and shape (e.g. 18x18)
  - user IO count
  - an achievable fmax band
  - type, width and bandwidth of on-board bulk memory

The last field matters most. The architecture's premise is trading
random access for cheap sequential bandwidth, so on-board memory
bandwidth is what decides whether any of this is interesting on a
given board.

Sourcing rules:

1. Prefer Project Apicula's device database -- it is machine-readable
   and already installed on the toolchain path if the open flow is
   present. Check for it before reaching for anything else.
2. Otherwise cite a Gowin datasheet or the Sipeed board page, by
   document name and revision.
3. Anything you cannot source is `Unknown`, not a guess. Model this in
   the type system so an unknown cannot be silently read as a number
   -- a test must fail if any field of the PRIMARY profile (9K) is
   unknown, while other boards may legitimately carry gaps.
4. Record every citation next to the value, in the code, not only in a
   doc.

Also carry forward what step 006 measured: the CPU reference is
compute-bound by roughly 196x against a page-cached read
(docs/results.md). Note in the profile docs that a board's bulk memory
bandwidth is the number that ratio has to be closed against, since
step 008's fit model will use both.

Testing:

- Every profile's primary fields present, or explicitly Unknown.
- The 9K profile complete, asserted, with no placeholders.
- Ordering/consistency sanity: a larger part does not report fewer
  LUT4s than a smaller one.

Gate before committing: `just check` green, `sw-checklist` at 0 failed
and no new warnings (the sw-install Binary Freshness warning is
expected and not yours to retire), no `#[allow]`.

Do NOT build the fit model or the `fit` subcommand -- that is step 008.
This step is data plus the types that hold it honestly.

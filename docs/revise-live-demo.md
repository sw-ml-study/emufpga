look at my "live-demo" for an experimental repo (linked in the footer): https://sw-ml-study.github.io/emufpga/ - I am not happy with this site. It looks too much like AI slop. How can it be improved to actually provide value and stand out from other AI generated UIs? For starters, I'd prefer a Paper Banana approach to diagrams and styling. Also, there is too much info shown and I do not see the relevance.
You are right to be dissatisfied. The current page contains good research, but it presents that research like a generated “AI accelerator product page”: hero statement, numbered sections, KPI cards, animated datapath, scenario laboratory, comparison cards, status badges, dense metrics, and multiple layers of caveats. The result is visually polished but cognitively noisy. It makes the visitor work too hard to discover the one interesting idea.

The underlying project is much more distinctive than the UI suggests. The repo is not “another FPGA AI accelerator.” It is an experiment asking whether model parameters can be treated as a serial stream rather than resident random-access memory, specifically to make oversized models feasible on old/cheap hardware. The README states this much more clearly than the live demo does.

I would radically simplify the site
I would change the live demo from a dashboard into an interactive technical notebook / lab bench.

The page should answer, in order:

What is the strange idea?
Why might it work?
Show me one concrete pass through the machine.
What has actually been measured?
What would falsify it?

Everything else belongs behind disclosure controls or in the repo documentation.

Right now, the site introduces three concepts immediately—what it is, what is measured, what is predicted—and then throws the visitor into counters, schedules, scenarios, derived numbers, resident-GPU comparisons, routing unions, throughput distributions, prefetch experiments, and hardware-shaped cycles.

That is almost the inverse of what you want for an experimental project.

The central visual should be one Paper-Banana-style drawing
I would make the first screen almost absurdly simple.

Something like this:

      THE USUAL WAY

   ┌─────────────────────────────┐
   │          GPU VRAM           │
   │                             │
   │  [ expert ][ expert ] ...   │
   │  [ expert ][ expert ] ...   │
   │                             │
   │     entire model lives      │
   │       near compute          │
   └─────────────────────────────┘

            ↓ too big?


      THE EXPERIMENT

   cheap storage
   ────────────────────────────────→
   [E0][E1][E2][E3][E4][E5]...

                 │
                 ▼
          ┌────────────┐
          │ tiny buffer│
          └─────┬──────┘
                │
                ▼
          ┌────────────┐
          │ FPGA-ish   │
          │ tensor     │
          │ engine     │
          └─────┬──────┘
                │
                ▼
             GPU/state
But drawn as an intentionally imperfect engineering sketch:

off-white paper
black or graphite linework
muted yellow highlight
handwritten annotations
arrows that look drawn rather than SVG-perfect
maybe a strip of “tape” representing serialized weights
very restrained motion
The message underneath:

Can we trade memory capacity for sequential bandwidth?

And then:

Instead of keeping hundreds of GB of immutable expert weights in RAM/VRAM, stream only the weights needed by the current computation past a small compute engine.

That is the project.

The page presently says essentially this, but buried in prose and surrounded by instrumentation.

The site should feel like a lab notebook, not SaaS
This is where the Paper Banana influence could help enormously.

I would establish a visual vocabulary like:

Paper

warm off-white background
faint dot grid or engineering paper
no gradients
no glassmorphism
no glowing borders
no giant rounded “cards”
Ink

near-black primary
graphite gray secondary
one yellow highlighter color
optionally one muted red for “failed / unsupported”
optionally one blue pencil color for measured data
Typography

readable serif or humanist sans for explanation
monospace for measurements
hand-written annotation font used very sparingly
avoid giant 64–80 px marketing headlines
Components

diagrams instead of cards
ruled-note callouts instead of badges
hand-drawn arrows instead of connector animations
real tables when a table is useful
marginal notes for caveats
The aesthetic should imply:

engineer's experiment book

rather than:

startup landing page for an AI infrastructure company

That distinction alone would make it stand out dramatically.

Most of the current homepage should disappear
There is too much information because the site is trying to simultaneously be:

an introduction
an emulator
a benchmark report
a hardware architecture document
a result explorer
an experiment notebook
a roadmap
a defense of the hypothesis
Those can all exist, but not simultaneously in the main reading path.

I would reduce the top-level site to perhaps five sections.

Section	Purpose
The idea	Explain serial parameter execution in ~30 seconds
Watch one pass	Interactive visualization of one layer/expert stream
Why bother?	Show RAM/VRAM avoided vs bytes transferred
What we learned	3–5 actual experimental findings
What would prove it?	Current unknowns + next experiment
That's it.

Everything else can be under Lab notebook →.

In particular, remove the “three card” opening
The current:

1 · WHAT IS THIS?
2 · WHAT DO WE MEASURE?
3 · WHAT DO WE PREDICT?

is one of the strongest sources of the “AI-generated landing page” feeling.

That pattern has become extremely common because LLMs generate it constantly.

Replace all three with one human statement:

What if model weights behaved more like tape than RAM?
Then explain the experiment in perhaps 50 words.

That is much more memorable.

The animation should teach one thing
The animated datapath is potentially the site's most valuable feature, but presently it asks the visitor to understand too many components at once:

expert identity
useful versus wasted bytes
FIFO behavior
MAC activity
schedules
stream utilization
Q6 decoding
routing
counters
The site itself tells the visitor to watch green experts, gray experts, MAC lanes, and FIFO pulses.

That is too much.

Instead make the interaction a guided experiment with three steps.

Step 1 — Dumb serial scan
Show eight experts on a “tape”:

storage

[E0]—[E1]—[E2]—[E3]—[E4]—[E5]—[E6]—[E7]
                ★               ★

             needed          needed
Then animate all eight crossing the compute boundary.

At the bottom:

Read: 8 experts
Used: 2 experts
Useful traffic: 25%

No more metrics.

Step 2 — Route first
The visitor clicks:

[ Route before reading ]

Now only E2 and E6 move.

Read: 2 experts
Used: 2 experts
Useful traffic: 100%

Now they understand the route-aware MoE proposition.

Step 3 — Add another request
Click:

[ + another inference ]

Maybe request A needs E2/E6 and request B needs E2/E5.

Then show:

A: E2 E6
B: E2 E5

union: E2 E5 E6
and explain:

E2 crosses the storage boundary once but serves both requests.

That demonstrates scan productivity intuitively before you ever introduce Ps.

This is much more valuable than an animation that assumes the visitor already understands the architecture.

Then introduce your unusual metrics visually
The repo has three genuinely interesting project-specific metrics:

η
Ps
Rp
Those are distinctive. The README explains them much better than the current page surfaces them.

But I would not put the symbols first.

Introduce each as a question.

How much of the model must stay resident?

usual inference
████████████████████████  ~100%

serial machine
▏                         ~0%

                            Rp
Then:

How many useful computations do we get
from each parameter fetched?

1 request       1×
8 requests      8×
128 requests  128×

             scan productivity Ps
Then:

Can compute consume the stream
as fast as storage supplies it?

storage ───────────────→
          compute ████

                η
Those visuals would be both educational and specific to your work.

The negative result should be prominent
This is another place where your project can differentiate itself from AI-slop demos.

Most generated project pages trumpet success.

Your README says:

Saga 1 measured the make-or-break experiment. The headline is a negative result with a usable number attached.

That is excellent scientific storytelling.

Put that on the site.

Perhaps a notebook page:

Experiment 1: Can the software engine keep up with storage?
Big handwritten result:

No.

Then:

page-cached storage  ───────────────────────────────→

scalar CPU engine    →

                    ≈196× gap
And underneath:

This does not disprove the architecture.
It tells us exactly what the FPGA must solve: arithmetic throughput.

That is interesting.

That sounds like research.

That does not sound like marketing.

Separate measured, simulated, projected far more aggressively
Your current site does try to label these, which is good:

MEASURED
SIMULATED
TO MEASURE
DERIVED / PROJECTED
But because they are all interleaved on one long page, the cognitive distinction becomes weak.

I'd instead use literal “notebook tabs”:

[ measured ] [ simulated ] [ conjecture ]
with Measured selected initially.

The first page should contain only things you actually know.

For example:

We have demonstrated
✓ bit-exact computation against the reference
✓ parameter residency can be tiny
✓ batching increases scan productivity
✓ expert routing changes traffic economics
✓ scalar software compute is much slower than storage

We have NOT demonstrated
✗ a synthesized FPGA implementation
✗ a physical fabric clock rate
✗ end-to-end oversized-MoE serving advantage
✗ tasks/kWh advantage
✗ superiority over CPU/RAM offload

That honesty is one of the strongest things about the repo.

Make it a design feature.

Hide the Scenario Laboratory
The current Scenario Laboratory has roughly eleven input parameters:

Model GiB, Fast memory GiB, Total experts, Active experts, Batch, Context, KV bytes/token, Storage GB/s, Compute TOPS, Active params B, Hypothetical clock MHz.

That is where the page becomes an engineering spreadsheet.

Useful? Yes.

Homepage material? No.

Move it to:

Lab tools → Capacity calculator

And perhaps provide three presets:

Tiny demonstration model
Granite-style MoE
Hypothetical 300 GB MoE
Let advanced users open “all parameters.”

This is progressive disclosure.

Likewise, don't show all architecture alternatives at once
The current page ends up mentioning HDD, NVMe, PMEM, CPU/DRAM, GPU/VRAM, FPGA, MCU/PIO and multiple proposed flows.

That belongs in a separate:

Experiments we could build
Then each architecture gets one Paper Banana sketch.

For example:

Cheap-storage serial machine
HDD/NVMe → FPGA → GPU
 weights   GEMV    state
Host-assisted
NVMe → DRAM → CPU → GPU
MCU + FPGA appliance
flash → MCU → FPGA → USB → host
Those could look like pages torn from an engineering notebook.

This is exactly where your older-server / small-FPGA hardware philosophy becomes visually distinctive rather than yet another abstract AI architecture.

The opening should probably say “old hardware”
The live site's current headline is already better than most of the page:

Can old hardware run oversized MoE models well enough?

I would keep that basic idea.

But I'd make it even more concrete:

Can a ten-year-old server run a model that doesn't fit in memory?
And under it:

This experiment treats immutable model weights as a serial data stream instead of something that must live in RAM or VRAM.

Then:

[Watch one layer move →]

That anchors the work in your actual motivation rather than generic ML acceleration.

I would also change the site's name treatment
emufpga is a repo name, not a concept.

Don't make it the visual brand.

The concept is stronger:

Serial Parameter Machine
Small caption:

emufpga — behavioral research emulator

That gives people something to remember and something to discuss.

Your README already uses Serial Parameter Machine (SPM) as the conceptual framing.

A concrete redesigned page
I can imagine the whole home page being only about 2–3 screens long before optional details.

------------------------------------------------------------

SERIAL PARAMETER MACHINE
emufpga research notebook

Can a ten-year-old server run a model
that doesn't fit in memory?

[ hand-drawn "RAM vs stream" diagram ]

Instead of storing all immutable weights near compute,
we stream them past compute as they are needed.

                     [ watch one pass ↓ ]

------------------------------------------------------------

ONE LAYER

[ interactive tape / expert visualization ]

        [ dumb scan ]   [ route first ]

requested: E2, E6

read          8 experts
use           2 experts
useful        25%

[ + add another inference ]

One stream can serve several independent requests.

------------------------------------------------------------

WHAT DID WE ACTUALLY LEARN?

[ notebook sketch ]

✓ Correct computation
✓ Tiny parameter residency
✓ Batching reuses streamed parameters

✗ Current scalar engine is ~196× too slow
  to consume a page-cached stream.

This gives the FPGA experiment a concrete job:
make arithmetic keep up with the bytes.

------------------------------------------------------------

THE CLAIM WE HAVEN'T PROVEN

Can this beat CPU/RAM offload for an oversized MoE
on the same old machine?

        NOT YET MEASURED

Success means:
≥ 4 independent agents
≥ 2 token/s each
≥ 25% more correct tasks/hour

[ experiment plan → ]

------------------------------------------------------------

lab notebook
results · methodology · calculators · architectures · source

© Software Wrighter
------------------------------------------------------------
That would be vastly stronger.

What makes this resist “AI slop”
It isn't merely the hand-drawn style.

AI slop is usually recognizable because it has too much presentation and too little editorial judgment.

The antidote is not “different CSS.” It is aggressive authorship.

Your page should make judgments such as:

This is the one idea worth understanding.

This result failed.

This number matters.

These numbers do not matter yet.

This is measured.

This is speculation.

Here is the next experiment.

That editorial hierarchy is what the current site lacks.

The current page contains plenty of responsible caveats—it even explicitly says the deployment advantage is not yet measured and that physical FPGA timing remains unknown. But those statements currently compete visually with dozens of other things.

They should instead define the story.

I would apply a severe design constraint
For the front page:

Every visualization must answer exactly one question.

And:

Every number must change the reader's understanding of the hypothesis.

If a chart, counter, toggle, or paragraph does neither, it goes into the lab notebook.

That probably removes 60–75% of the current visible UI without losing any research value.

And I think the resulting site would be considerably more memorable: less “AI accelerator dashboard,” more Bell Labs notebook crossed with an interactive hardware experiment. That is a much better fit for what emufpga actually is.

that's funny that you picked on the three card opening, as that was my idea to add (not the AI) :-)
Ha — and that’s a useful correction. :-) The three-card idea itself isn't AI slop. I was too quick to associate the visual pattern with the problem.

In fact, your three questions are quite sensible:

What is this?
What do we measure?
What do we predict?
That is actually good epistemic organization for an experimental project. In particular, separating measurement from prediction is something I very much want to preserve.

I think the real problem is how those cards participate in the page, not their existence.

I'd revise my recommendation
Keep the three-part conceptual structure, but make it feel like an experiment legend rather than three equal SaaS feature cards.

For example:

WHAT IS THIS?
─────────────
An experiment in treating model parameters
as a serial stream instead of resident memory.

        ↓

WHAT CAN WE MEASURE?
────────────────────
correctness · bytes read · residency
scan productivity · stalls

        ↓

WHAT ARE WE TRYING TO FIND OUT?
───────────────────────────────
Can cheap sequential storage + small compute
make oversized models practical on old hardware?
That's a good progression: mechanism → evidence → hypothesis.

I might even strengthen your original wording by changing “What do we predict?” to “What are we trying to find out?” Prediction sounds like the site is presenting an architectural performance model as an answer. The latter emphasizes that this is research.

And visually, Paper-Banana styling could make these look like three annotations on the same sheet of engineering paper rather than three independently bordered, rounded cards.

So I retract the criticism of the three-part idea. I still think the page has a serious hierarchy/density problem, but your three questions may actually be useful scaffolding for solving it.

The bigger target I'd attack is everything that happens after those three questions: the site keeps answering more and more questions before the visitor has had an opportunity to internalize the first one.

yes, I'd like to incorporate your suggestions, and only keep what is useful, but do not present it on one page.
Yes. I think that changes the design substantially for the better.

The goal should not be “make the current long page prettier.” It should be to turn the live demo into a small explorable research site, where each page has one job and one primary idea.

Proposed information architecture
I would use about six small pages, with the Paper-Banana / engineering-notebook visual language shared across them.

Page	Question it answers	Main visual
Home	What is this, and why should I care?	RAM vs serial-stream sketch
How it works	What actually happens to the weights?	Interactive weight/expert stream
Experiments	What have you actually tested?	Notebook-style experiment index
Results	What did you learn?	A few focused diagrams, including failures
Hypothesis	What remains unproven?	Evidence/unknowns/next-test map
Lab	Let me explore the numbers myself	Existing calculators, knobs, detailed emulator
And then Source / Docs takes the serious reader to the repo rather than duplicating all the repository documentation.

This creates a deliberate progression:

idea → mechanism → experiment → evidence → unanswered question → exploration

rather than today's progression of “here is everything we know.”

1. Home — explain exactly one idea
I'd keep your three-part opening, but make it essentially the entire homepage.

Serial Parameter Machine
Can a ten-year-old server run a model that doesn't fit in memory?

Large Paper-Banana-style sketch:

CONVENTIONAL

             ┌──────────── GPU ────────────┐
             │                             │
             │   model weights             │
             │   ██████████████████████    │
             │                             │
             └─────────────────────────────┘
                         ↑
                   must fit here


SERIAL PARAMETER MACHINE

 cheap storage
 ════════════════════════════════════▶
 [weights][weights][weights][weights]
                     │
                     ▼
                ┌─────────┐
                │ compute │
                └────┬────┘
                     ▼
                 small state
Then your three concepts, perhaps as annotations rather than cards:

What is this?
A research experiment that treats immutable model parameters as a sequential stream rather than something that must all remain in RAM or VRAM.

What do we measure?
Correctness, parameter residency, storage traffic, useful work per parameter read, and where the pipeline stalls.

What are we trying to find out?
Whether cheap sequential storage plus small purpose-built compute can make oversized models practical on older hardware.

Then just two paths:

Watch how it works →

See what we've measured →

And stop.

No scenario calculator. No HDD/NVMe comparison. No detailed FPGA datapath. No dozen metrics.

The homepage should take perhaps 60 seconds to consume.

2. How It Works — make this the actual “live demo”
This is where I'd put most of the effort.

Rather than showing the whole FPGA-ish datapath immediately, make it a step-through visualization.

Step 1 — A model is mostly immutable parameters
Show the model as a long paper tape.

MODEL

[E0][E1][E2][E3][E4][E5][E6][E7] ...
═══════════════════════════════════▶
Brief explanation.

Next →

Step 2 — Don't load the tape into memory
Animate the tape moving past compute.

storage                              compute

[E0][E1][E2][E3][E4] ─────────▶  [ Q6 decode ]
                                      ↓
                                   [ MAC ]
                                      ↓
                                accumulator
Now introduce the tiny FIFO.

Step 3 — MoE makes this particularly interesting
Now color/mark experts:

 E0   E1   E2   E3   E4   E5   E6   E7
           ★                   ★

         needed              needed
Then demonstrate naïve scan versus route-aware access.

Step 4 — Multiple requests can share a scan
This is where Ps becomes understandable.

request A       E2          E6
request B       E2     E5

stream          E2     E5   E6
                 │
                 ├── A
                 └── B
Only after seeing that:

We call this scan productivity (Ps): useful parameter applications per parameter value fetched.

That's much better pedagogy than displaying Ps = 5 and expecting the reader to understand why it matters.

Step 5 — Reveal the actual datapath
Only now show your existing detailed FPGA emulator.

At this point FIFO, Q6_K decoder, MAC lanes, accumulator BRAM and router mask have meaning.

I'd put a switch here:

Conceptual | Detailed

The detailed view is for people who came specifically for the FPGA work.

3. Experiments — turn the project history into a strength
I think this could become one of the most distinctive pages.

Don't make it a dashboard.

Make it an experimental notebook index.

Something like:

EXPERIMENT 01                         Aug 2026

Can batching reuse one parameter stream?

Hypothesis
    storage traffic should remain approximately
    constant as batch increases.

Result
    ✓ YES

    batch       1    8    32    128
    traffic     ────────────────────
    useful work ███████████████████▶

Measured:
    Ps = batch size

Reproduce:
    $ just bench

[ methodology ] [ raw results ] [ source ]
Then experiment 2.

Can the CPU engine keep up with storage?
Result:

✗ NO
And prominently:

Approximately 196× too slow against the tested page-cached read.

That's a great result for an experimental site because it leads naturally to:

Therefore: the next question isn't “can storage feed it?” It's “can purpose-built compute consume it?”

The README explicitly characterizes Saga 1 as a negative result and warns that the current engine is compute-bound throughout the measured range.

That scientific progression should be much more visible.

4. Results — evidence, not every number
This page answers:

Where does the project stand today?

I'd have three sections.

Established
Things supported by your experiments:

bit-exact reference behavior
very low parameter residency is possible
batching increases scan productivity without proportional parameter traffic
routing changes the economics of MoE streaming
current scalar CPU implementation cannot consume the available stream quickly enough
Observed but limited
Things that need qualification:

prefetch results
particular storage-tier behavior
Q6_K experiments
routing distributions
cycle-model behavior
Not established
This is extremely important:

We have not demonstrated that an FPGA Serial Parameter Machine beats conventional inference.

And underneath:

NOT YET KNOWN

 FPGA throughput          ?
 real fabric clock        ?
 end-to-end tokens/sec    ?
 joules/token             ?
 agents/kWh               ?
 cost advantage           ?
That's a powerful page.

It actively resists hype.

5. Hypothesis — where is this going?
This replaces much of the speculative material currently mixed with results.

I'd call it Questions, Next Experiments, or Hypothesis, rather than “Roadmap.”

The central Paper-Banana diagram could be an evidence tree:

                     BIG QUESTION

             Can serial parameters make
             oversized MoE economical?
                       │
          ┌────────────┼─────────────┐
          │            │             │
          ▼            ▼             ▼
       MEMORY        COMPUTE        ENERGY
          │            │             │
          ✓            ?             ?
       proven       FPGA test      measure
          │
          ▼
       ROUTING
          │
          ✓
       promising
And importantly distinguish:

Measured → Derived → Projected → Speculative

Those shouldn't just be tiny badges attached to numbers. They are different epistemic categories.

6. Lab — this is where most of today's page goes
The existing work isn't necessarily bad. Much of it is simply in the wrong place.

Put the engineering toys here.

Lab tools
Serial MoE visual emulator
Scenario calculator
memory/residency calculator
storage bandwidth experiments
batch scheduler
routing-union explorer
FIFO visualization
Q6_K decoder visualization
HDD/NVMe/PMEM comparisons
CPU/GPU/FPGA comparisons
This page can be dense.

The visitor explicitly chose:

I want the knobs.

So now density is useful rather than overwhelming.

I'd even make URLs preserve experiments:

/lab?model=300&experts=128&active=8&batch=5...

That makes scenarios shareable and reproducible.

Navigation should be unusually restrained
Something like:

SPM

Idea    How it works    Experiments    Results    Lab       GitHub
I wouldn't have dropdown menus.

And probably no sticky side navigation on the explanatory pages.

Paper Banana should be structural, not merely decorative
This is important.

I don't think the answer is to take today's UI and give its cards wobbly borders and handwritten fonts.

That would become Paper-Banana-themed AI slop.

Instead, use the metaphor to influence information design.

A research notebook naturally has:

one experiment per page
sketches
annotations
crossed-out hypotheses
measurements
dates
marginal caveats
references to previous experiments
arrows connecting observations
unanswered questions
raw data available behind the polished summary
That maps extremely well onto this project.

For example, a negative experiment might literally show:

Hypothesis:

storage will become the bottleneck
at sufficiently large batch.

              ↓ test

~~ crossover around batch N ~~     ← nope

Observed:
compute bound throughout measured range.

Next:
build enough parallel arithmetic to move
the crossover into the measurable region.
That feels authored because it exposes the reasoning process, including where the reasoning was wrong.

One more page may eventually be worthwhile: Build It
Not yet necessarily.

But as you move from behavioral emulator → FPGA dev board → MCU+FPGA prototype → potentially custom PCB, I think a Build page becomes compelling.

It could eventually contain photographs rather than generated illustrations:

software model
      ↓
Tang Nano FPGA
      ↓
MCU + FPGA dev boards
      ↓
custom PCB
      ↓
?????
That would make the project increasingly resistant to generic AI presentation because the evidence becomes physical artifacts, oscilloscope captures, logic analyzer traces, boards, measurements, and reproducible code.

The Paper Banana drawings explain the idea; photographs and measurements prove you actually built it.

What I would keep, move, simplify, or delete
Current material	Action
Three opening questions	Keep, simplify visually
Old-hardware question	Keep prominently
SPM explanation	Keep, drastically shorten
Animated datapath	Keep, move to How It Works and progressively reveal
Measured/simulated/projected distinction	Keep and strengthen
Ps, Rp, η	Keep, teach visually before showing notation
Scenario Laboratory	Move to Lab
Detailed knobs	Move to Lab
HDD/NVMe/PMEM alternatives	Move to Lab/Hypothesis
Detailed routing experiments	Experiments/Lab
Negative results	Promote
Giant collections of numeric results	Remove from narrative pages
Caveat paragraphs	Replace with explicit evidence state where possible
Reproducibility links	Keep everywhere relevant
Repo documentation duplication	Remove; link instead
The important architectural change is therefore not really one page → six pages. It's:

one giant presentation → a progressive research narrative with an escape hatch into the laboratory.

I'd make Home → How It Works → Experiments → Results the curated path for someone encountering the idea for the first time, while Lab becomes the playground for someone like you—or another engineer—who actually wants all those knobs and measurements.

And I would make the site progressively less polished as one goes deeper: clean notebook illustration on Home, annotated technical diagrams in How It Works, plots/raw numbers in Experiments, and unapologetically dense engineering controls in Lab. That progression itself communicates: introduction → explanation → evidence → instrument.

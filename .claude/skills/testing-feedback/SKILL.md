---
name: testing-feedback
description: Take Eviatar's proposed list of what to test for a module, critique it -- what is missing, what is redundant, what would survive a deliberate bug -- and write those tests once he agrees. Use this whenever he says what he thinks should be tested, asks whether his test plan is any good, asks what else is worth testing, asks whether the tests are strong enough, or asks for a mutation pass. Do not use it to invent a test plan from nothing; the units and the intent come from him.
---

# Testing feedback

Eviatar names the units and says what he wants asserted about each. This skill
critiques that plan and, once agreed, writes the tests.

The constraint exists for a reason he stated: deciding *what is worth asserting*
is the part that teaches; typing out `assert_close` is not. So never arrive with
a finished plan he did not ask for. If he has not said what to test, the answer
is a question or a prose sketch, not code.

## Critiquing the plan

Work through his list and report on four things.

**What a trivial implementation would do.** This is the highest-yield question in
the whole workflow. For every assertion of the form "these two things are equal",
ask what degenerate implementation makes everything equal -- an identity
function, a function ignoring its input, a constant. If such an implementation
passes, the test needs a second clause that rules it out.

The mirror case matters just as much and is easier to miss: when an assertion
says "these two things differ", ask what property guarantees they do not. A test
here compared RoPE at positions 0-4 against 100-104 expecting a difference and
got byte-identical output, because RoPE depends only on *relative* position by
construction. The test was asserting an invariance it thought it was refuting.

**What is missing.** Look for the stages of the module his list does not mention
at all, and for the project's recurring failure mode -- right shape, wrong
numbers, no exception. Also look for the cases where a normally-negligible term
becomes significant: `eps` inside versus outside a square root is invisible at
unit scale and only separable when `mean(x^2)` is comparable to `eps`.

**What is redundant.** If two tests catch exactly the same set of bugs, say so
and recommend keeping the one that states the rule more directly. Redundant tests
are not free -- they make the suite slower to read and imply coverage that is not
there.

**Detection versus localization.** When a strong oracle exists, the oracle detects
almost everything, so the other tests are not earning their place through
detection. They earn it by *localizing* -- naming which stage broke -- or by
asserting something the oracle cannot express, like causality or a design
decision such as "the caller supplies the positions". Say which job each proposed
test is doing. If a whole class of bugs is detectable only end to end, the
finding is usually about the *module* being too coarse, and belongs in
implementation feedback instead: extract the unit, then it becomes testable.

## Mutation testing

This project's method, and the way to answer "is this test plan strong enough"
with evidence rather than opinion. Break the implementation deliberately, one
change at a time, run the suite, record which tests go red, restore.

Run it when he asks, and offer it once a new suite is green.

Mechanics that matter:

- Always restore the file, in a `finally` block. Back it up to the scratchpad
  first.
- Assert the suite is green before starting, or every result is meaningless.
- Report a matrix: mutation down the side, catching tests across. A mutation
  nothing catches is a hole and is the entire point of the exercise.
- Separate the oracle tests from the rest in the report. "Caught, but only by the
  parity test" is a different result from "caught by its own unit test", and the
  difference is exactly the localization point above.

Favour mutations that keep the shape valid, since those are the ones that fail
silently: tiling instead of interleaving, reducing over the wrong axis, a
dropped scale factor, swapped weight tensors, an off-by-one in a mask.

Be honest that the matrix is only as good as the mutations you thought to write.
The first pass on `attention` missed two of the most important ones. Say what you
did not try.

## Writing the tests

Only after agreement, and only the units and assertions he agreed to. Do not add
extra tests because they seemed worthwhile -- if something is missing, that is a
finding to raise, not a liberty to take.

Match the conventions already in `tests/`: existing fixtures, tolerances from the
project's testing convention rather than invented ones, shapes derived from
`config` rather than literals, and a docstring that says *why the test exists and
what bug it catches* rather than restating the assertion.

Prefer toy sizes with hand-computed expected values where a unit permits it --
they need no model, run instantly, and pin the arithmetic exactly. Fall back to
the oracle where hand computation is impractical.

Then run them. A test that has never failed is not known to work: check each new
test by deliberately breaking the code it covers and confirming it goes red. A
discarded return value is a test that always passes -- `torch.equal(a, b)`
without `assert` is legal Python and silently green.

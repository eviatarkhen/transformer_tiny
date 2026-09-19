---
name: implementation-feedback
description: Review a module Eviatar has written and give structured feedback on correctness, performance, modularity and pythonic style, then apply the changes he agrees to. Use this whenever he asks for a review, says he does not like his implementation, asks "is this right", asks what could be better or cleaner, or finishes writing a function and wants a second opinion -- even if he does not use the word "review". Do not use it for writing new code from scratch, for tests (use testing-feedback), or for explaining theory.
---

# Implementation feedback

Eviatar wrote the code. The job here is to help him see it more clearly, not to
rewrite it out from under him. He is a strong software engineer and a CS /
engineering manager; he is not fluent in ML terminology. Assume he will
understand any software argument immediately and will want the domain reasoning
spelled out.

Nothing changes on disk until he agrees. Feedback first, then a decision, then
edits.

## Verify before claiming

The single most useful thing in this workflow is running code instead of
asserting things. This is not ceremony -- it repeatedly turns a vague style note
into a concrete finding.

Real examples from this project:

- "Don't forget the `.T` on weights" was a generic warning. Running it across all
  four projections showed `q_proj` and `o_proj` *raise*, while `k_proj` and
  `v_proj` are square and silently return the right shape with numbers off by
  7.7. That reframed the advice entirely.
- A claim that two tensors would differ turned out to be false, because RoPE is
  invariant to a uniform position shift by construction. Running it caught a
  wrong claim before it reached him.

So: before reporting a finding, run the thing. Small scripts in the scratchpad
directory, not in the project. Report the measured number, not the adjective.
If a claim cannot be checked cheaply, say it is unverified rather than implying
it was.

## The four axes

Report on each axis, in this order. Say explicitly when an axis is clean --
"performance: nothing to report" is useful information, and padding it with
invented findings wastes his time.

**Correctness.** Does it do what it claims? Run the test suite. Run the module
against its oracle if one exists. Look hardest at the failure mode this project
keeps producing: *right shape, wrong numbers, no exception*. Transposes,
axis numbers, reductions over the wrong dimension, broadcast that silently
succeeds. A correctness finding should come with the input that breaks it.

**Performance.** Only report what matters at this scale, and measure before
claiming. Note that `rms_norm` measured identical for rsqrt vs divide because it
is memory-bandwidth bound -- instruction-count arguments that do not reduce data
movement are noise here. Allocation, redundant recomputation across 28 layers,
and unnecessary `.contiguous()` are worth flagging; micro-arithmetic is not.

**Modularity.** The question that pays off is: *can this piece be tested on its
own?* If a bug in it can only be caught end to end, the unit is too big. Ask
whether each extracted function has a name from the problem domain -- `repeat_kv`
and `causal_mask` do, `state_reshape_view` did not, and that is the difference
between an abstraction and a mechanic. Also flag the reverse: wrappers around a
single builtin call that add a layer without adding meaning.

**Pythonic.** Naming, shadowing, dead parameters, repeated string keys,
suffix-accumulating variable names (`q_view` -> `q_norm` -> `q_trans`), type
hints, docstrings that say what rather than why. Keep this axis short; it is the
least important of the four and he will notice most of it himself.

## Reporting

Lead with a one-line verdict, then the axes. For each finding:

- where it is, as `file:line`
- what is wrong, in one sentence
- the concrete input or measurement that demonstrates it
- what you would change, and what it costs

Rank within each axis by how much it matters. Distinguish "this is a bug" from
"this is a preference" -- conflating them makes him distrust the whole report.
If a finding contradicts a decision already recorded in `CLAUDE.md` or
`docs/PROGRESS.md`, say so and argue the case rather than quietly ignoring the
record; those decisions were made for reasons and the reasons may still hold.

Close by asking which findings he wants applied. Do not bundle them -- he may
want two of five.

## Applying

Only after he says which ones. Then:

- back the file up to the scratchpad first, so a bad refactor is one command
  from being undone
- make exactly the agreed changes, nothing adjacent
- re-run the full suite, and the oracle comparison if the module has one
- report what moved, including numbers that improved or degraded

If applying a change breaks his tests, say so plainly and hand the fix back to
him rather than silently rewriting the tests to match -- test changes go through
the `testing-feedback` workflow.

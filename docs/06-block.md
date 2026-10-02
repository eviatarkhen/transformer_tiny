# Testing
the hidden layer can be got from

expected = hf_block(x, hf_model, layer)

## HF has two attention implementations:
- sdpa, the default: a fused kernel that does the arithmetic in its own order.
- eager: plain matmul, softmax, matmul, which is the same arithmetic we wrote.

Compared against eager, your block is bit-for-bit identical everywhere:

┌─────────────────────────────────────────────┬───────────────────┬──────────┐
│                 comparison                  │ vs sdpa (default) │ vs eager │
├─────────────────────────────────────────────┼───────────────────┼──────────┤
│ random input, layer 27, (2, 7)              │ 3.9e-4            │ 0.0      │
├─────────────────────────────────────────────┼───────────────────┼──────────┤
│ random input, layer 27, (1, 32)             │ 1.2e-3            │ 0.0      │
├─────────────────────────────────────────────┼───────────────────┼──────────┤
│ real stream, each of 28 layers alone, worst │ 3.0e-5            │ 0.0      │
├─────────────────────────────────────────────┼───────────────────┼──────────┤
│ real stream, all 28 layers chained          │ 4.4e-4            │ 0.0      │
└─────────────────────────────────────────────┴───────────────────┴──────────┘

Why sdpa shows a diff, and why it's harmless

Float32 addition isn't associative: (a + b) + c can differ from a + (b + c) in the last bit. sdpa sums in a different order than a plain matmul does, so its attention output differs from ours by about 2.5e-5. That's rounding noise.

At layer 27, the norm in front of the MLP has weights as large as 192, and the MLP then multiplies again. That stage turns 2.5e-5 into about 1e-3 on random input. Compared with the size of the output, it's still tiny: 5e-7 relative, which is about float32's precision limit. On real input it stays around 3e-5, and chained through all 28 layers it's 3e-7 relative.

So the tests are right to use real input at deep layers. The 1e-4 tolerance is an absolute number, and absolute tolerances break when values get large.

A side finding: the eager attention path isn't causal by default

My first eager run gave nonsense diffs in the thousands, because I called the layer with attention_mask=None. Measured: under eager, mask=None means no mask at all. The output differs from the masked version by 0.49, and changing the last token moves position 0. Under sdpa, None means causal.
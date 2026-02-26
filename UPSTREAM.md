# Upstream: TMU (Tsetlin Machine Unified)

## Source

- **Repository:** https://github.com/cair/tmu
- **License:** MIT
- **Author:** Ole-Christoffer Granmo and contributors at CAIR (Centre for Artificial Intelligence Research, University of Agder)
- **Pinned tag:** v0.8.3

## What we use

The C core from `tmu/lib/`:

- `src/ClauseBank.c` -- clause evaluation, Type I/II/III feedback
- `src/WeightBank.c` -- clause weight management
- `src/Tools.c` -- binary encoding utilities
- `src/random/pcg32_fast.c` -- PCG32 PRNG
- `include/` -- corresponding headers

## What we add

`csrc/wl_tm_api.c` is a C glue layer (~550 lines) that wraps the TMU core and adds:

- **Training loop** -- multiclass one-vs-all orchestration with contiguous-block polarity
- **Binarization** -- quantile-based thresholds for continuous features
- **Regression** -- target scaling to `[0, T]` range
- **Serialization** -- TM01 binary format for model persistence
- **Stable C ABI** -- exported functions for WASM/JS interop

The TMU C core handles clause evaluation and feedback at the bit level. Our glue layer handles the training orchestration, data preprocessing, and serialization that the upstream Python code provides.

## Update policy

- Upstream is tracked as a git submodule at `upstream/tmu/`
- Version bumps are visible as submodule pointer changes
- Local patches: none (clean wrapper around unmodified upstream)
- Security fixes: merge upstream tag, rebuild WASM, run tests

## Differences from upstream Python

The upstream TMU Python package provides training via its own Python orchestration code. This WASM port reimplements the training loop in C to run in browsers and Node.js without Python dependencies. The clause-level operations (feedback, evaluation) use the upstream C code directly.

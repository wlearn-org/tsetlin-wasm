# Changelog

## 0.1.0

Initial release.

- Tsetlin Machine compiled to WebAssembly via Emscripten
- sklearn-style API: `create()`, `fit()`, `predict()`, `score()`, `save()`, `load()`, `dispose()`
- Binary and multiclass classification (one-vs-all)
- Regression with automatic target scaling
- `predictProba()` for classification (softmax over clause votes)
- Automatic binarization of continuous features via quantile-based thresholds
- TM01 binary serialization format wrapped in WLRN v1 bundles
- Configurable: clauses, threshold, specificity, state bits, boost TPF, epochs, seed
- AutoML integration: `getParams()`, `setParams()`, `defaultSearchSpace()`
- FinalizationRegistry safety net for leak detection
- Upstream: TMU (Tsetlin Machine Unified) v0.8.3 C core by Ole-Christoffer Granmo (MIT)

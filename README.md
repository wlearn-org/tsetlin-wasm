# @wlearn/tsetlin

Tsetlin Machine compiled to WebAssembly -- interpretable ML in browsers and Node.js.

Wraps the C core from [TMU](https://github.com/cair/tmu) (Tsetlin Machine Unified) by Ole-Christoffer Granmo. ESM module, zero native dependencies.

## Install

```bash
npm install @wlearn/tsetlin
```

## Quick start

```js
import { TsetlinModel } from '@wlearn/tsetlin'

// Create model (async -- loads WASM)
const model = await TsetlinModel.create({
  nClauses: 200,
  threshold: 50,
  s: 3.0,
  nEpochs: 100,
})

// Train
model.fit(X_train, y_train)

// Predict
const predictions = model.predict(X_test)
const accuracy = model.score(X_test, y_test)

// Probabilities (classification only)
const proba = model.predictProba(X_test)

// Save / load
const bytes = model.save()
const loaded = await TsetlinModel.load(bytes)

// Release WASM memory
model.dispose()
```

## API

### `TsetlinModel.create(params?)`

Async factory. Returns a `Promise<TsetlinModel>`.

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | `string` | `'classification'` | `'classification'` or `'regression'` |
| `nClauses` | `number` | `100` | Number of clauses (half positive, half negative) |
| `threshold` | `number` | `50` | T -- voting threshold |
| `s` | `number` | `3.0` | Specificity parameter |
| `stateBits` | `number` | `8` | Bits per Tsetlin automaton |
| `boostTruePositiveFeedback` | `boolean` | `false` | Boost Type I feedback on true positives |
| `nThresholdsPerFeature` | `number` | `10` | Max binary thresholds per continuous feature |
| `nEpochs` | `number` | `100` | Training epochs |
| `seed` | `number` | `42` | PRNG seed for reproducibility |

### `model.fit(X, y)`

Train on data. Accepts `number[][]` or `{ data: Float64Array, rows, cols }`. Continuous features are automatically binarized using quantile-based thresholds. Returns `this`.

### `model.predict(X)`

Returns `Float64Array` of predictions.

- Classification: integer class labels
- Regression: continuous values scaled to original range

### `model.predictProba(X)`

Returns `Float64Array` of class probabilities (flat, row-major). Classification only. Uses softmax over raw clause vote sums.

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.save()`

Returns `Uint8Array` -- a WLRN v1 bundle containing the TM01 binary blob.

### `TsetlinModel.load(bytes)`

Async. Loads a saved bundle. Returns `Promise<TsetlinModel>`.

### `model.dispose()`

Frees WASM memory. Must be called when done. Safe to call multiple times.

### `model.getParams()` / `model.setParams(p)`

Get or update hyperparameters. For AutoML compatibility.

### `TsetlinModel.defaultSearchSpace()`

Returns a search space object for hyperparameter optimization.

## How it works

Tsetlin Machines are interpretable ML models based on propositional logic. They learn conjunctive clauses (AND of binary features) that vote for or against each class. Training uses learning automata (Tsetlin automata) to include or exclude literals from each clause.

Continuous input features are automatically binarized: for each feature, quantile-based thresholds are computed, producing binary features of the form `x > threshold`. Both the original and negated literals are available to the clauses.

For classification, clauses are organized as one-vs-all: each class gets its own set of clauses. Half the clauses have positive polarity (vote for the class) and half have negative polarity (vote against). The class with the highest total vote wins.

For regression, clause votes are summed and scaled back to the original target range.

## Resource management

WASM linear memory is not garbage collected. Always call `dispose()` when done. A `FinalizationRegistry` safety net logs a warning if a model is garbage collected without being disposed.

## Build from source

Requires [Emscripten](https://emscripten.org/).

```bash
git clone --recurse-submodules https://github.com/wlearn-org/tsetlin-wasm
cd tsetlin-wasm
npm install
bash scripts/build-wasm.sh
bash scripts/verify-exports.sh
node test/test.js
```

## License

MIT. Upstream TMU is also MIT licensed.

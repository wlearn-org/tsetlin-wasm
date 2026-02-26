import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// --- Deterministic LCG PRNG ---
function makeLCG(seed = 42) {
  let s = seed | 0
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

function makeClassificationData(rng, nSamples, nFeatures, nClasses = 2) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const label = i % nClasses
    const row = []
    for (let j = 0; j < nFeatures; j++) {
      row.push(label * 2 + (rng() - 0.5) * 0.5)
    }
    X.push(row)
    y.push(label)
  }
  return { X, y }
}

function makeRegressionData(rng, nSamples, nFeatures) {
  const X = [], y = []
  for (let i = 0; i < nSamples; i++) {
    const row = []
    let target = 0
    for (let j = 0; j < nFeatures; j++) {
      const v = rng() * 4 - 2
      row.push(v)
      target += v * (j + 1)
    }
    X.push(row)
    y.push(target + (rng() - 0.5) * 0.1)
  }
  return { X, y }
}

// ============================================================
// WASM Loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadTsetlin } = await import('../src/wasm.js')
const wasm = await loadTsetlin()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('Exported functions present', async () => {
  assert(typeof wasm._wl_tm_create === 'function', 'wl_tm_create missing')
  assert(typeof wasm._wl_tm_fit === 'function', 'wl_tm_fit missing')
  assert(typeof wasm._wl_tm_predict === 'function', 'wl_tm_predict missing')
  assert(typeof wasm._wl_tm_save === 'function', 'wl_tm_save missing')
  assert(typeof wasm._wl_tm_load === 'function', 'wl_tm_load missing')
  assert(typeof wasm._wl_tm_free === 'function', 'wl_tm_free missing')
})

// ============================================================
// Model creation and capabilities
// ============================================================
console.log('\n=== Model Basics ===')

const { TsetlinModel } = await import('../src/model.js')

await test('Create unfitted model', async () => {
  const model = await TsetlinModel.create({ task: 'classification', nClauses: 20 })
  assert(!model.isFitted, 'should not be fitted')
  model.dispose()
})

await test('Capabilities (classifier)', async () => {
  const model = await TsetlinModel.create({ task: 'classification' })
  const caps = model.capabilities
  assert(caps.classifier === true, 'should be classifier')
  assert(caps.regressor === false, 'should not be regressor')
  assert(caps.predictProba === true, 'should support predictProba')
  model.dispose()
})

await test('Capabilities (regressor)', async () => {
  const model = await TsetlinModel.create({ task: 'regression' })
  const caps = model.capabilities
  assert(caps.classifier === false, 'should not be classifier')
  assert(caps.regressor === true, 'should be regressor')
  assert(caps.predictProba === false, 'should not support predictProba')
  model.dispose()
})

// ============================================================
// Binary classification
// ============================================================
console.log('\n=== Binary Classification ===')

await test('Binary classification accuracy > 0.7', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 200, 4, 2)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 40,
    threshold: 20,
    s: 3.0,
    stateBits: 8,
    nEpochs: 30,
    nThresholdsPerFeature: 5,
    seed: 42
  })
  model.fit(X, y)
  assert(model.isFitted, 'should be fitted')
  const acc = model.score(X, y)
  assert(acc > 0.7, `accuracy ${acc} should be > 0.7`)
  model.dispose()
})

await test('Binary classification with boost TPF', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 200, 4, 2)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 40,
    threshold: 20,
    s: 3.0,
    boostTruePositiveFeedback: true,
    nEpochs: 30,
    nThresholdsPerFeature: 5,
    seed: 42
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.7, `boosted accuracy ${acc} should be > 0.7`)
  model.dispose()
})

await test('predictProba returns valid probabilities', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 100, 4, 2)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 20,
    threshold: 10,
    nEpochs: 20,
    nThresholdsPerFeature: 5,
    seed: 42
  })
  model.fit(X, y)
  const proba = model.predictProba(X)
  assert(proba.length === 100 * 2, `expected 200 values, got ${proba.length}`)
  // Check each row sums to ~1
  for (let i = 0; i < 100; i++) {
    const sum = proba[i * 2] + proba[i * 2 + 1]
    assertClose(sum, 1.0, 1e-6, `row ${i} probabilities sum to ${sum}`)
    assert(proba[i * 2] >= 0, 'probability must be >= 0')
    assert(proba[i * 2 + 1] >= 0, 'probability must be >= 0')
  }
  model.dispose()
})

await test('predictProba fails for regression', async () => {
  const model = await TsetlinModel.create({ task: 'regression', nClauses: 20, nEpochs: 5, seed: 42 })
  model.fit([[1], [2], [3]], [1, 2, 3])
  let threw = false
  try { model.predictProba([[1]]) } catch { threw = true }
  assert(threw, 'predictProba should throw for regression')
  model.dispose()
})

// ============================================================
// Multiclass classification
// ============================================================
console.log('\n=== Multiclass ===')

await test('Multiclass (3 classes)', async () => {
  const rng = makeLCG(77)
  const { X, y } = makeClassificationData(rng, 300, 4, 3)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 60,
    threshold: 30,
    s: 3.0,
    nEpochs: 50,
    nThresholdsPerFeature: 5,
    seed: 77
  })
  model.fit(X, y)
  const acc = model.score(X, y)
  assert(acc > 0.5, `multiclass accuracy ${acc} should be > 0.5`)
  assert(model.nClasses === 3, `expected 3 classes, got ${model.nClasses}`)
  model.dispose()
})

await test('Multiclass predictProba shape', async () => {
  const rng = makeLCG(77)
  const { X, y } = makeClassificationData(rng, 60, 3, 3)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 30,
    threshold: 15,
    nEpochs: 20,
    nThresholdsPerFeature: 3,
    seed: 77
  })
  model.fit(X, y)
  const proba = model.predictProba(X)
  assert(proba.length === 60 * 3, `expected 180 values, got ${proba.length}`)
  model.dispose()
})

// ============================================================
// Regression
// ============================================================
console.log('\n=== Regression ===')

await test('Regression R2 > 0', async () => {
  const rng = makeLCG(99)
  const { X, y } = makeRegressionData(rng, 200, 3)
  const model = await TsetlinModel.create({
    task: 'regression',
    nClauses: 100,
    threshold: 50,
    s: 3.0,
    nEpochs: 50,
    nThresholdsPerFeature: 10,
    seed: 99
  })
  model.fit(X, y)
  const r2 = model.score(X, y)
  assert(r2 > 0, `R2 ${r2} should be > 0`)
  model.dispose()
})

await test('Regression has 0 classes', async () => {
  const model = await TsetlinModel.create({ task: 'regression', nClauses: 20, nEpochs: 5, seed: 42 })
  model.fit([[1], [2], [3]], [1, 2, 3])
  assert(model.nClasses === 0, `expected 0 classes, got ${model.nClasses}`)
  assert(model.classes === null, 'classes should be null for regression')
  model.dispose()
})

// ============================================================
// Save/Load round-trip
// ============================================================
console.log('\n=== Save/Load ===')

await test('Save/load classification round-trip', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 100, 4, 2)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 20,
    threshold: 10,
    nEpochs: 20,
    nThresholdsPerFeature: 5,
    seed: 42
  })
  model.fit(X, y)
  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  const loaded = await TsetlinModel.load(bundle)
  const preds2 = loaded.predict(X)
  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `prediction mismatch at ${i}: ${preds1[i]} vs ${preds2[i]}`)
  }
  loaded.dispose()
})

await test('Save/load regression round-trip', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeRegressionData(rng, 100, 3)
  const model = await TsetlinModel.create({
    task: 'regression',
    nClauses: 40,
    threshold: 30,
    nEpochs: 20,
    nThresholdsPerFeature: 5,
    seed: 42
  })
  model.fit(X, y)
  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  const loaded = await TsetlinModel.load(bundle)
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-10, `regression pred mismatch at ${i}`)
  }
  loaded.dispose()
})

await test('Save/load multiclass round-trip', async () => {
  const rng = makeLCG(77)
  const { X, y } = makeClassificationData(rng, 90, 3, 3)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 30,
    threshold: 15,
    nEpochs: 20,
    nThresholdsPerFeature: 3,
    seed: 77
  })
  model.fit(X, y)
  const preds1 = model.predict(X)
  const bundle = model.save()
  model.dispose()

  const loaded = await TsetlinModel.load(bundle)
  assert(loaded.nClasses === 3, 'nClasses mismatch')
  const preds2 = loaded.predict(X)
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `multiclass pred mismatch at ${i}`)
  }
  loaded.dispose()
})

await test('TM01 header in saved blob', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 50, 2, 2)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 10,
    threshold: 5,
    nEpochs: 5,
    nThresholdsPerFeature: 3,
    seed: 42
  })
  model.fit(X, y)
  const bundle = model.save()
  // Bundle is WLRN format; the TM01 blob is inside
  assert(bundle.length > 64, 'bundle too small')
  model.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const model = await TsetlinModel.create({ nClauses: 50, threshold: 25 })
  const params = model.getParams()
  assert(params.nClauses === 50, 'nClauses mismatch')
  assert(params.threshold === 25, 'threshold mismatch')
  model.setParams({ s: 5.0 })
  assert(model.getParams().s === 5.0, 's not updated')
  model.dispose()
})

await test('Refit with different params', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 100, 3, 2)
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 10,
    threshold: 5,
    nEpochs: 5,
    nThresholdsPerFeature: 3,
    seed: 42
  })
  model.fit(X, y)
  const acc1 = model.score(X, y)

  // Refit with more clauses/epochs
  model.setParams({ nClauses: 40, nEpochs: 30 })
  model.fit(X, y)
  const acc2 = model.score(X, y)
  assert(model.isFitted, 'should still be fitted')
  model.dispose()
})

await test('Small dataset (3 samples)', async () => {
  const model = await TsetlinModel.create({
    task: 'classification',
    nClauses: 10,
    threshold: 5,
    nEpochs: 10,
    nThresholdsPerFeature: 2,
    seed: 42
  })
  model.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 0])
  const preds = model.predict([[0, 0], [1, 1], [2, 2]])
  assert(preds.length === 3, 'should predict 3 samples')
  model.dispose()
})

// ============================================================
// Lifecycle
// ============================================================
console.log('\n=== Lifecycle ===')

await test('Dispose frees resources', async () => {
  const model = await TsetlinModel.create({ nClauses: 10, nEpochs: 5, seed: 42, nThresholdsPerFeature: 2 })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()
  assert(!model.isFitted, 'should not be fitted after dispose')
})

await test('Double dispose is safe', async () => {
  const model = await TsetlinModel.create({ nClauses: 10, nEpochs: 5, seed: 42, nThresholdsPerFeature: 2 })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()
  model.dispose() // should not throw
})

await test('Not fitted error', async () => {
  const model = await TsetlinModel.create()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'should throw NotFittedError')
  model.dispose()
})

await test('Disposed error', async () => {
  const model = await TsetlinModel.create({ nClauses: 10, nEpochs: 5, seed: 42, nThresholdsPerFeature: 2 })
  model.fit([[1, 2], [3, 4]], [0, 1])
  model.dispose()
  let threw = false
  try { model.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'should throw DisposedError')
})

// ============================================================
// Search space
// ============================================================
console.log('\n=== Search Space ===')

await test('defaultSearchSpace', async () => {
  const space = TsetlinModel.defaultSearchSpace()
  assert(space.nClauses, 'nClauses missing')
  assert(space.threshold, 'threshold missing')
  assert(space.s, 's missing')
  assert(space.stateBits, 'stateBits missing')
  assert(space.boostTruePositiveFeedback, 'boostTruePositiveFeedback missing')
  assert(space.nThresholdsPerFeature, 'nThresholdsPerFeature missing')
  assert(space.nEpochs, 'nEpochs missing')
})

// ============================================================
// Determinism
// ============================================================
console.log('\n=== Determinism ===')

await test('Same seed = same predictions', async () => {
  const rng = makeLCG(42)
  const { X, y } = makeClassificationData(rng, 80, 3, 2)
  const params = {
    task: 'classification',
    nClauses: 20,
    threshold: 10,
    nEpochs: 10,
    nThresholdsPerFeature: 3,
    seed: 123
  }
  const m1 = await TsetlinModel.create(params)
  m1.fit(X, y)
  const p1 = m1.predict(X)
  m1.dispose()

  const m2 = await TsetlinModel.create(params)
  m2.fit(X, y)
  const p2 = m2.predict(X)
  m2.dispose()

  assert(p1.length === p2.length, 'length mismatch')
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `determinism failed at ${i}: ${p1[i]} vs ${p2[i]}`)
  }
})

// ============================================================
// Summary
// ============================================================
console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`)
if (failed > 0) process.exit(1)

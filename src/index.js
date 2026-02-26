export { loadTsetlin, getWasm } from './wasm.js'
export { TsetlinModel } from './model.js'

// Convenience: create, fit, return fitted model
export async function train(params, X, y) {
  const model = await TsetlinModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
export async function predict(bundleBytes, X) {
  const model = await TsetlinModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

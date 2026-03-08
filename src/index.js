const { loadTsetlin, getWasm } = require('./wasm.js')
const { TsetlinModel } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await TsetlinModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await TsetlinModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadTsetlin, getWasm, TsetlinModel, train, predict }

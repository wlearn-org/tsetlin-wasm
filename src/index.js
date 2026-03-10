const { loadTsetlin, getWasm } = require('./wasm.js')
const { TsetlinModel: TsetlinModelImpl } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const TsetlinModel = createModelClass(TsetlinModelImpl, TsetlinModelImpl, { name: 'TsetlinModel', load: loadTsetlin })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await TsetlinModel.create(params)
  await model.fit(X, y)
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

// WASM loader -- loads the Tsetlin Machine WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadTsetlin(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createTsetlin = require('../wasm/tsetlin.cjs')
    wasmModule = await createTsetlin(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadTsetlin() first')
  return wasmModule
}

module.exports = { loadTsetlin, getWasm }

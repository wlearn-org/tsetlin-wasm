// WASM loader -- loads the Tsetlin Machine WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadTsetlin(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const require = createRequire(import.meta.url)
    const createTsetlin = require('../wasm/tsetlin.cjs')
    wasmModule = await createTsetlin(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadTsetlin() first')
  return wasmModule
}

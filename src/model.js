import { getWasm, loadTsetlin } from './wasm.js'
import {
  normalizeX, normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/tsetlin: Model was not disposed -- calling free() automatically.')
      freeFn(ptr[0])
    }
  })
  : null

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_tm_get_last_error', 'string', [], [])
}

const LOAD_SENTINEL = Symbol('load')

export class TsetlinModel {
  #handle = null
  #freed = false
  #ptrRef = null
  #params = {}
  #fitted = false
  #nClasses = 0
  #classes = null

  constructor(handle, params, extra) {
    if (handle === LOAD_SENTINEL) {
      this.#handle = params
      this.#params = extra.params || {}
      this.#nClasses = extra.nClasses || 0
      this.#classes = extra.classes || null
      this.#fitted = true
    } else {
      this.#handle = null
      this.#params = handle || {}
    }

    this.#freed = false
    if (this.#handle) {
      this.#registerLeak()
    }
  }

  static async create(params = {}) {
    await loadTsetlin()
    return new TsetlinModel(params)
  }

  fit(X, y) {
    this.#ensureFitted(false)
    const wasm = getWasm()

    // Dispose previous model if refitting
    if (this.#handle) {
      wasm._wl_tm_free(this.#handle)
      this.#handle = null
      if (this.#ptrRef) this.#ptrRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    const yData = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    if (yData.length !== rows) {
      throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)
    }

    const task = this.#taskEnum()
    const nClauses = this.#params.nClauses ?? 100
    const threshold = this.#params.threshold ?? 50
    const s = this.#params.s ?? 3.0
    const stateBits = this.#params.stateBits ?? 8
    const boost = this.#params.boostTruePositiveFeedback ? 1 : 0
    const nThresh = this.#params.nThresholdsPerFeature ?? 10
    const nEpochs = this.#params.nEpochs ?? 100
    const seed = this.#params.seed ?? 42

    // Extract class info before training (for classification)
    if (task === 0) {
      const classSet = new Set()
      for (let i = 0; i < yData.length; i++) classSet.add(yData[i] | 0)
      this.#classes = new Int32Array([...classSet].sort((a, b) => a - b))
      this.#nClasses = this.#classes.length
    } else {
      this.#classes = null
      this.#nClasses = 0
    }

    // Create model in WASM
    const modelPtr = wasm._wl_tm_create(
      nClauses, threshold, s, stateBits, boost, nThresh, task, seed
    )
    if (!modelPtr) {
      throw new Error(`Create failed: ${getLastError()}`)
    }

    // Copy X to WASM heap
    const xBytes = xData.length * 8
    const xPtr = wasm._malloc(xBytes)
    wasm.HEAPF64.set(xData, xPtr / 8)

    // Copy y to WASM heap
    const yBytes = yData.length * 8
    const yPtr = wasm._malloc(yBytes)
    wasm.HEAPF64.set(yData, yPtr / 8)

    // Train
    const ret = wasm._wl_tm_fit(modelPtr, xPtr, rows, cols, yPtr, nEpochs)

    wasm._free(xPtr)
    wasm._free(yPtr)

    if (ret !== 0) {
      wasm._wl_tm_free(modelPtr)
      throw new Error(`Training failed: ${getLastError()}`)
    }

    this.#handle = modelPtr
    this.#fitted = true

    // Update class info from model
    if (task === 0) {
      this.#nClasses = wasm._wl_tm_get_n_classes(modelPtr)
    }

    this.#registerLeak()
    return this
  }

  predict(X) {
    this.#ensureFitted()
    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * 4)

    const ret = wasm._wl_tm_predict(this.#handle, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const task = this.#taskEnum()

    if (task === 0) {
      // Classification: int32 labels
      const result = new Float64Array(rows)
      for (let i = 0; i < rows; i++) {
        result[i] = wasm.HEAP32[(outPtr / 4) + i]
      }
      wasm._free(outPtr)
      return result
    } else {
      // Regression: raw votes -> scale back
      const T = wasm._wl_tm_get_threshold(this.#handle)
      const yMin = wasm._wl_tm_get_y_min(this.#handle)
      const yMax = wasm._wl_tm_get_y_max(this.#handle)
      const result = new Float64Array(rows)
      for (let i = 0; i < rows; i++) {
        const raw = wasm.HEAP32[(outPtr / 4) + i]
        result[i] = yMin + (raw / T) * (yMax - yMin)
      }
      wasm._free(outPtr)
      return result
    }
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#taskEnum() !== 0) {
      throw new Error('predictProba is only available for classification')
    }

    const wasm = getWasm()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const nClasses = this.#nClasses

    const xPtr = wasm._malloc(xData.length * 8)
    wasm.HEAPF64.set(xData, xPtr / 8)

    const outPtr = wasm._malloc(rows * nClasses * 4)

    const ret = wasm._wl_tm_predict_votes(this.#handle, xPtr, rows, cols, outPtr)

    wasm._free(xPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`predictProba failed: ${getLastError()}`)
    }

    // Convert raw votes to probabilities via softmax
    const result = new Float64Array(rows * nClasses)
    for (let i = 0; i < rows; i++) {
      let maxVote = -Infinity
      for (let c = 0; c < nClasses; c++) {
        const vote = wasm.HEAP32[(outPtr / 4) + i * nClasses + c]
        result[i * nClasses + c] = vote
        if (vote > maxVote) maxVote = vote
      }
      // Softmax
      let expSum = 0
      for (let c = 0; c < nClasses; c++) {
        result[i * nClasses + c] = Math.exp(result[i * nClasses + c] - maxVote)
        expSum += result[i * nClasses + c]
      }
      for (let c = 0; c < nClasses; c++) {
        result[i * nClasses + c] /= expSum
      }
    }

    wasm._free(outPtr)
    return result
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#taskEnum() === 1) {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    } else {
      // Accuracy
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if (preds[i] === yArr[i]) correct++
      }
      return correct / preds.length
    }
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const rawBytes = this.#saveRaw()
    const task = this.#params.task || 'classification'
    const typeId = task === 'regression'
      ? 'wlearn.tsetlin.regressor@1'
      : 'wlearn.tsetlin.classifier@1'

    const metadata = {}
    if (task !== 'regression') {
      metadata.nClasses = this.#nClasses
      metadata.classes = Array.from(this.#classes)
    }

    return encodeBundle(
      { typeId, params: this.getParams(), metadata },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return TsetlinModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadTsetlin()
    const wasm = getWasm()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const params = manifest.params || {}

    const bufPtr = wasm._malloc(raw.length)
    wasm.HEAPU8.set(raw, bufPtr)
    const modelPtr = wasm._wl_tm_load(bufPtr, raw.length)
    wasm._free(bufPtr)

    if (!modelPtr) {
      throw new Error(`load failed: ${getLastError()}`)
    }

    const metadata = manifest.metadata || {}
    const nClasses = metadata.nClasses || wasm._wl_tm_get_n_classes(modelPtr)
    const classes = metadata.classes
      ? new Int32Array(metadata.classes)
      : null

    return new TsetlinModel(LOAD_SENTINEL, modelPtr, {
      params, nClasses, classes
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_tm_free(this.#handle)
    }

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      nClauses: { type: 'categorical', values: [20, 50, 100, 200, 500, 1000, 2000] },
      threshold: { type: 'int_uniform', low: 10, high: 200 },
      s: { type: 'log_uniform', low: 1.0, high: 20.0 },
      stateBits: { type: 'int_uniform', low: 4, high: 12 },
      boostTruePositiveFeedback: { type: 'categorical', values: [true, false] },
      nThresholdsPerFeature: { type: 'int_uniform', low: 2, high: 20 },
      nEpochs: { type: 'int_uniform', low: 20, high: 200 }
    }
  }

  // --- Inspection ---

  get nClasses() {
    return this.#nClasses
  }

  get classes() {
    return this.#classes ? new Int32Array(this.#classes) : null
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const isClassifier = this.#taskEnum() === 0
    return {
      classifier: isClassifier,
      regressor: !isClassifier,
      predictProba: isClassifier,
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  // --- Private helpers ---

  #taskEnum() {
    return (this.#params.task || 'classification') === 'regression' ? 1 : 0
  }

  #normalizeX(X) {
    return normalizeX(X, 'auto')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('TsetlinModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('TsetlinModel is not fitted. Call fit() first.')
  }

  #registerLeak() {
    this.#ptrRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_tm_free(h)
      }, this)
    }
  }

  #saveRaw() {
    const wasm = getWasm()

    const outBufPtr = wasm._malloc(4)
    const outLenPtr = wasm._malloc(4)

    const ret = wasm._wl_tm_save(this.#handle, outBufPtr, outLenPtr)

    if (ret !== 0) {
      wasm._free(outBufPtr)
      wasm._free(outLenPtr)
      throw new Error(`save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.getValue(outBufPtr, 'i32')
    const bufLen = wasm.getValue(outLenPtr, 'i32')

    const result = new Uint8Array(bufLen)
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + bufLen))

    wasm._wl_tm_free_buffer(bufPtr)
    wasm._free(outBufPtr)
    wasm._free(outLenPtr)

    return result
  }
}

// Register loaders with @wlearn/core
register('wlearn.tsetlin.classifier@1', (m, t, b) => TsetlinModel._fromBundle(m, t, b))
register('wlearn.tsetlin.regressor@1', (m, t, b) => TsetlinModel._fromBundle(m, t, b))

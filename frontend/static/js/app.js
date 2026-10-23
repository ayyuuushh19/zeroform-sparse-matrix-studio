/**
 * ZEROFORM — Sparse Matrix Studio
 * Full Frontend Integration with C and Flask (Stage 6)
 *
 * NOTE: Mathematical algorithms, sparse compression, matrix arithmetic,
 * sorting, and theoretical memory analyses are strictly delegated to the C11 engine
 * via the Flask REST API. JavaScript manages presentation, input drafting,
 * coordinate translation, state validation, and communication.
 */

(function () {
  'use strict';

  // ── 1. Application State ──────────────────────────────────────────────────
  const state = {
    matrixA: {
      rows: 3,
      cols: 3,
      cells: {}, // Key: `${r}_${c}` (1-based), Value: float
      view: 'grid' // 'grid' | 'triplets'
    },
    matrixB: {
      rows: 3,
      cols: 3,
      cells: {},
      visible: false,
      view: 'grid'
    },
    lastOperation: null, // { type, result, timestamp, isStale }
    lastAnalysis: null,  // { result, timestamp, isStale }
    inFlight: false,
    activePasteTarget: 'A', // 'A' or 'B'
    engineHealthy: false,
    renderThreshold: 2500 // Max cells to render in DOM table
  };

  // ── 2. DOM Elements Cache ─────────────────────────────────────────────────
  const getEl = id => (typeof document !== 'undefined' ? document.getElementById(id) : null);
  const getAll = sel => (typeof document !== 'undefined' ? document.querySelectorAll(sel) : []);
  const getOne = sel => (typeof document !== 'undefined' ? document.querySelector(sel) : null);

  const elements = {
    // Health Badge
    statusBadge: getEl('status-badge'),
    statusDot: getEl('status-dot'),
    statusText: getEl('status-text'),

    // Top Preset Strip & Matrix B Toggle
    samplePresetSelect: getEl('sample-preset-select'),
    btnLoadPreset: getEl('btn-load-preset'),
    btnToggleMatrixB: getEl('btn-toggle-matrix-b'),

    // Matrix A Controls
    dimARows: getEl('dim-a-rows'),
    dimACols: getEl('dim-a-cols'),
    btnResizeA: getEl('btn-resize-a'),
    btnPasteA: getEl('btn-paste-a'),
    btnViewADense: getEl('btn-view-a-dense'),
    btnViewATriplets: getEl('btn-view-a-triplets'),
    tableMatrixA: getEl('table-matrix-a'),
    containerAGrid: getEl('container-a-grid'),
    containerATriplets: getEl('container-a-triplets'),
    tbodyATriplets: getEl('tbody-matrix-a-triplets'),

    // Matrix B Controls
    panelMatrixB: getEl('panel-matrix-b'),
    dimBRows: getEl('dim-b-rows'),
    dimBCols: getEl('dim-b-cols'),
    btnResizeB: getEl('btn-resize-b'),
    btnPasteB: getEl('btn-paste-b'),
    btnViewBDense: getEl('btn-view-b-dense'),
    btnViewBTriplets: getEl('btn-view-b-triplets'),
    tableMatrixB: getEl('table-matrix-b'),
    containerBGrid: getEl('container-b-grid'),
    containerBTriplets: getEl('container-b-triplets'),
    tbodyBTriplets: getEl('tbody-matrix-b-triplets'),

    // Command Deck Operations
    opToSparse: getEl('op-to-sparse'),
    opToDense: getEl('op-to-dense'),
    opTranspose: getEl('op-transpose'),
    opAdd: getEl('op-add'),
    opMultiply: getEl('op-multiply'),
    opAnalyze: getEl('op-analyze'),
    opReset: getEl('op-reset'),

    // Result Inspector
    resultBadge: getEl('result-badge'),
    resultStaleBadge: getEl('result-stale-badge'),
    btnExportDense: getEl('btn-export-dense'),
    btnExportTriplets: getEl('btn-export-triplets'),
    btnCopyTriplets: getEl('btn-copy-triplets'),
    tabButtons: getAll('.tab-btn'),
    tabContents: getAll('.tab-content'),
    tabDense: getEl('tab-dense'),
    tabTriplets: getEl('tab-triplets'),
    tabStats: getEl('tab-stats'),
    tabMemory: getEl('tab-memory'),


    // Theoretical Memory Analysis Sidebar
    telemetryStaleBadge: getEl('telemetry-stale-badge'),
    metricDims: getEl('metric-dims'),
    metricTotalCells: getEl('metric-total-cells'),
    metricNnz: getEl('metric-nnz'),
    metricZeros: getEl('metric-zeros'),
    metricDensity: getEl('metric-density'),
    metricSparsity: getEl('metric-sparsity'),
    metricDenseBytes: getEl('metric-dense-bytes'),
    metricSparseBytes: getEl('metric-sparse-bytes'),
    metricSavingsBadge: getEl('metric-savings-badge'),
    scaleBarTrack: getOne('.scale-bar-track'),
    scaleBarFill: getEl('scale-bar-fill'),

    // Bottom Activity Bar
    activityPill: getEl('activity-pill'),
    activityMessage: getEl('activity-message'),
    activityTimestamp: getEl('activity-timestamp'),

    // Dialogs
    helpDialog: getEl('help-dialog'),
    btnOpenHelp: getEl('btn-open-help'),
    btnCloseHelp: getEl('btn-close-help'),
    btnDismissHelp: getEl('btn-dismiss-help'),

    resetDialog: getEl('reset-dialog'),
    btnCloseReset: getEl('btn-close-reset'),
    btnCancelReset: getEl('btn-cancel-reset'),
    btnConfirmReset: getEl('btn-confirm-reset'),

    pasteDialog: getEl('paste-dialog'),
    pasteTargetLabel: getEl('paste-target-label'),
    pasteTextarea: getEl('paste-textarea'),
    pasteErrorMsg: getEl('paste-error-msg'),
    btnClosePaste: getEl('btn-close-paste'),
    btnCancelPaste: getEl('btn-cancel-paste'),
    btnApplyPaste: getEl('btn-apply-paste')
  };

  // ── 3. Activity Logger & Inline Feedback ──────────────────────────────────
  function updateActivity(status, message) {
    const s = String(status).toUpperCase();
    elements.activityPill.className = `status-pill ${s.toLowerCase()}`;
    elements.activityPill.textContent = s;
    elements.activityMessage.textContent = message;
    const now = new Date();
    elements.activityTimestamp.textContent = now.toTimeString().split(' ')[0];
  }

  // ── 4. Coordinate Translation & Adapters ──────────────────────────────────
  // Wire format: 0-based indices [0..R-1, 0..C-1] for C engine protocol.
  // Display format: 1-based indices [1..R, 1..C] for mathematician/engineer UI.

  function displayToWire(r1, c1) {
    return {
      row: parseInt(r1, 10) - 1,
      col: parseInt(c1, 10) - 1
    };
  }

  function wireToDisplay(r0, c0) {
    return {
      row: parseInt(r0, 10) + 1,
      col: parseInt(c0, 10) + 1
    };
  }

  function matrixToDensePayload(mat) {
    const grid = [];
    for (let r = 1; r <= mat.rows; r++) {
      const row = [];
      for (let c = 1; c <= mat.cols; c++) {
        const val = mat.cells[`${r}_${c}`];
        row.push(val !== undefined ? val : 0.0);
      }
      grid.push(row);
    }
    return {
      format: 'dense',
      rows: mat.rows,
      cols: mat.cols,
      data: grid
    };
  }

  function matrixToSparsePayload(mat) {
    const entries = [];
    for (let r = 1; r <= mat.rows; r++) {
      for (let c = 1; c <= mat.cols; c++) {
        const val = mat.cells[`${r}_${c}`];
        if (val !== undefined && val !== 0.0) {
          const wire = displayToWire(r, c);
          entries.push({
            row: wire.row,
            col: wire.col,
            value: val
          });
        }
      }
    }
    return {
      format: 'sparse',
      rows: mat.rows,
      cols: mat.cols,
      nnz: entries.length,
      entries: entries
    };
  }

  // ── 4b. Matrix Dimension Validation & Safety Limits ──────────────────────────
  // Conforms with backend and C-engine limits (ZF_MAX_DIM=100000, ZF_MAX_CELLS=10000000)
  const LIMITS = {
    MAX_DIM: 100000,
    MAX_CELLS: 10000000
  };

  /**
   * Strictly validates a single dimension input string (row or column).
   * Rejects:
   *  - fractional values (e.g. '2.5', '3.7', '0.5')
   *  - zero ('0') or negative ('-2')
   *  - empty strings and whitespace-only input
   *  - malformed values ('3abc', '2.5xyz', 'Infinity', 'NaN')
   *  - scientific notation ('1e3')
   *  - values exceeding MAX_DIM (100,000)
   *
   * @param {string|number} raw - The raw input value
   * @param {HTMLInputElement} [inputElem] - Optional DOM input element to check validityState
   * @returns {{ ok: boolean, value?: number, reason: string }}
   */
  function parseStrictDimension(raw, inputElem) {
    if (inputElem && inputElem.validity && inputElem.validity.badInput) {
      return { ok: false, reason: 'malformed' };
    }

    if (raw === null || raw === undefined) {
      return { ok: false, reason: 'empty' };
    }

    const str = String(raw).trim();
    if (str === '') {
      return { ok: false, reason: 'empty' };
    }

    // Must be a positive integer without leading zeros, decimals, exponents, or signs
    if (!/^[1-9]\d*$/.test(str)) {
      return { ok: false, reason: 'invalid_format' };
    }

    const num = Number(str);
    if (!Number.isSafeInteger(num) || num <= 0) {
      return { ok: false, reason: 'not_safe_integer' };
    }

    if (num > LIMITS.MAX_DIM) {
      return { ok: false, reason: 'exceeds_max_dim', value: num };
    }

    return { ok: true, value: num };
  }

  /**
   * Validates both dimensions of a matrix against positive integer rules and the
   * universal 10,000,000 cell ceiling using overflow-safe division.
   *
   * @param {string|number} rawRows
   * @param {string|number} rawCols
   * @param {HTMLInputElement} [rowElem]
   * @param {HTMLInputElement} [colElem]
   * @param {string} [matrixName='Matrix']
   * @returns {{ ok: boolean, rows: number|null, cols: number|null, rowError: boolean, colError: boolean, message: string }}
   */
  function validateMatrixDimensions(rawRows, rawCols, rowElem, colElem, matrixName = 'Matrix') {
    const rRes = parseStrictDimension(rawRows, rowElem);
    const cRes = parseStrictDimension(rawCols, colElem);

    const rowError = !rRes.ok;
    const colError = !cRes.ok;

    if (rowError || colError) {
      let msg = '';
      if (rowError && colError) {
        msg = `${matrixName} rows and columns must be valid positive integers between 1 and ${LIMITS.MAX_DIM.toLocaleString()}.`;
      } else if (rowError) {
        if (rRes.reason === 'exceeds_max_dim') {
          msg = `${matrixName} rows (${rRes.value.toLocaleString()}) exceeds maximum allowed dimension of ${LIMITS.MAX_DIM.toLocaleString()}.`;
        } else {
          msg = `${matrixName} rows must be a valid positive integer between 1 and ${LIMITS.MAX_DIM.toLocaleString()}.`;
        }
      } else {
        if (cRes.reason === 'exceeds_max_dim') {
          msg = `${matrixName} columns (${cRes.value.toLocaleString()}) exceeds maximum allowed dimension of ${LIMITS.MAX_DIM.toLocaleString()}.`;
        } else {
          msg = `${matrixName} columns must be a valid positive integer between 1 and ${LIMITS.MAX_DIM.toLocaleString()}.`;
        }
      }

      return {
        ok: false,
        rows: null,
        cols: null,
        rowError,
        colError,
        message: msg
      };
    }

    const r = rRes.value;
    const c = cRes.value;

    // Universal cell ceiling check: rows * cols <= MAX_CELLS (overflow-safe)
    if (c > 0 && r > Math.floor(LIMITS.MAX_CELLS / c)) {
      return {
        ok: false,
        rows: r,
        cols: c,
        rowError: true,
        colError: true,
        message: `${matrixName} dimensions (${r.toLocaleString()}×${c.toLocaleString()} = ${(r * c).toLocaleString()} cells) exceed the universal limit of ${LIMITS.MAX_CELLS.toLocaleString()} cells.`
      };
    }

    return {
      ok: true,
      rows: r,
      cols: c,
      rowError: false,
      colError: false,
      message: ''
    };
  }

  function clearDimensionErrors(targetKey) {
    if (targetKey === 'A' || targetKey === 'both') {
      if (elements.dimARows) {
        elements.dimARows.classList.remove('invalid');
        elements.dimARows.removeAttribute('aria-invalid');
      }
      if (elements.dimACols) {
        elements.dimACols.classList.remove('invalid');
        elements.dimACols.removeAttribute('aria-invalid');
      }
    }
    if (targetKey === 'B' || targetKey === 'both') {
      if (elements.dimBRows) {
        elements.dimBRows.classList.remove('invalid');
        elements.dimBRows.removeAttribute('aria-invalid');
      }
      if (elements.dimBCols) {
        elements.dimBCols.classList.remove('invalid');
        elements.dimBCols.removeAttribute('aria-invalid');
      }
    }
  }

  function applyMatrixResize(targetKey) {
    const isA = targetKey === 'A';
    const matrixName = isA ? 'Matrix A' : 'Matrix B';
    const mat = isA ? state.matrixA : state.matrixB;
    const rowElem = isA ? elements.dimARows : elements.dimBRows;
    const colElem = isA ? elements.dimACols : elements.dimBCols;
    const tableElem = isA ? elements.tableMatrixA : elements.tableMatrixB;

    const val = validateMatrixDimensions(rowElem.value, colElem.value, rowElem, colElem, matrixName);

    if (!val.ok) {
      if (val.rowError) {
        rowElem.classList.add('invalid');
        rowElem.setAttribute('aria-invalid', 'true');
      } else {
        rowElem.classList.remove('invalid');
        rowElem.removeAttribute('aria-invalid');
      }

      if (val.colError) {
        colElem.classList.add('invalid');
        colElem.setAttribute('aria-invalid', 'true');
      } else {
        colElem.classList.remove('invalid');
        colElem.removeAttribute('aria-invalid');
      }

      updateActivity('ERROR', val.message);
      return;
    }

    // Success: clear invalid markers
    rowElem.classList.remove('invalid');
    rowElem.removeAttribute('aria-invalid');
    colElem.classList.remove('invalid');
    colElem.removeAttribute('aria-invalid');

    const newRows = val.rows;
    const newCols = val.cols;

    // Prune entries outside new boundaries using O(NNZ) filtering
    const newCells = {};
    for (const [key, v] of Object.entries(mat.cells)) {
      const parts = key.split('_');
      const ri = Number(parts[0]);
      const ci = Number(parts[1]);
      if (ri <= newRows && ci <= newCols) {
        newCells[key] = v;
      }
    }

    mat.rows = newRows;
    mat.cols = newCols;
    mat.cells = newCells;

    // Ensure input fields display the clean parsed integer
    rowElem.value = newRows;
    colElem.value = newCols;

    renderMatrixGrid(tableElem, mat, matrixName);
    renderTripletsTable(matrixName);
    markStale();
    updateActivity('IDLE', `${matrixName} resized to ${newRows}×${newCols}.`);
  }

  // Synchronize Matrix A input drafting cells from C-returned triplets
  function syncMatrixAFromTriplets(triplets, rows, cols) {
    if (rows !== undefined) state.matrixA.rows = rows;
    if (cols !== undefined) state.matrixA.cols = cols;
    elements.dimARows.value = state.matrixA.rows;
    elements.dimACols.value = state.matrixA.cols;
    clearDimensionErrors('A');

    state.matrixA.cells = {};
    if (Array.isArray(triplets)) {
      triplets.forEach(t => {
        const disp = wireToDisplay(t.row, t.col);
        if (t.value !== 0.0) {
          state.matrixA.cells[`${disp.row}_${disp.col}`] = t.value;
        }
      });
    }

    renderMatrixGrid(elements.tableMatrixA, state.matrixA, 'Matrix A');
    renderTripletsTable('Matrix A');
  }

  // ── 5. Stale State Tracking ───────────────────────────────────────────────
  function markStale() {
    if (state.lastOperation) {
      state.lastOperation.isStale = true;
      if (elements.resultStaleBadge) elements.resultStaleBadge.style.display = 'inline-block';
    }
    if (state.lastAnalysis) {
      state.lastAnalysis.isStale = true;
      if (elements.telemetryStaleBadge) elements.telemetryStaleBadge.style.display = 'inline-block';
    }
    // Refresh current result tab if open to reflect stale badge
    renderActiveResultTab();
  }

  function clearStale() {
    if (elements.resultStaleBadge) elements.resultStaleBadge.style.display = 'none';
    if (elements.telemetryStaleBadge) elements.telemetryStaleBadge.style.display = 'none';
  }

  // ── 6. Reliable API Client ────────────────────────────────────────────────
  class ApiError extends Error {
    constructor(code, message, status) {
      super(message);
      this.name = 'ApiError';
      this.code = code || 'UNKNOWN_ERROR';
      this.status = status || 500;
    }
  }

  const apiClient = {
    async request(url, options = {}) {
      if (state.inFlight) {
        throw new ApiError('IN_FLIGHT', 'Another operation is currently processing. Please wait.');
      }

      setInFlight(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs || 20000);

      try {
        const fetchOptions = {
          method: options.method || 'GET',
          headers: {
            'Content-Type': 'application/json',
            ...(options.headers || {})
          },
          signal: controller.signal
        };

        if (options.body) {
          fetchOptions.body = JSON.stringify(options.body);
        }

        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);

        let data = null;
        const text = await response.text();
        if (text && text.trim().length > 0) {
          try {
            data = JSON.parse(text);
          } catch (e) {
            throw new ApiError('INVALID_JSON', 'Server returned non-JSON payload.', response.status);
          }
        }

        if (!response.ok) {
          const errCode = data?.error?.code || `HTTP_${response.status}`;
          const errMessage = data?.error?.message || response.statusText || 'Request failed';
          throw new ApiError(errCode, errMessage, response.status);
        }

        return data;
      } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
          throw new ApiError('TIMEOUT', 'Execution request timed out after 20 seconds.', 504);
        }
        if (err instanceof ApiError) {
          throw err;
        }
        throw new ApiError('NETWORK_ERROR', `Network failure: Unable to communicate with server (${err.message}).`, 0);
      } finally {
        setInFlight(false);
      }
    },

    getHealth() {
      // Health check bypasses inFlight guard to allow concurrent heartbeat
      return fetch('/api/health')
        .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)));
    },

    getSamples() {
      return this.request('/api/samples', { method: 'GET' });
    },

    postAnalyze(matrixPayload) {
      return this.request('/api/analyze', {
        method: 'POST',
        body: { matrix: matrixPayload }
      });
    },

    postOperation(operation, matrixA, matrixB) {
      const body = {
        operation: operation,
        matrix_a: matrixA
      };
      if (matrixB) {
        body.matrix_b = matrixB;
      }
      return this.request('/api/operation', {
        method: 'POST',
        body: body
      });
    }
  };

  function setInFlight(inFlight) {
    state.inFlight = inFlight;
    const actionButtons = [
      elements.opToSparse, elements.opToDense, elements.opTranspose,
      elements.opAdd, elements.opMultiply, elements.opAnalyze, elements.opReset,
      elements.btnLoadPreset, elements.btnResizeA, elements.btnResizeB
    ];

    actionButtons.forEach(btn => {
      if (btn) {
        btn.disabled = inFlight;
        if (inFlight) btn.classList.add('in-flight');
        else btn.classList.remove('in-flight');
      }
    });

    if (inFlight) {
      updateActivity('PROCESSING', 'Dispatching request to C11 engine via Flask subprocess...');
    }
  }

  // ── 7. Health Check Management ────────────────────────────────────────────
  async function checkEngineHealth() {
    try {
      const data = await apiClient.getHealth();
      if (data && data.status === 'healthy') {
        state.engineHealthy = true;
        elements.statusBadge.className = 'health-badge online';
        elements.statusText.textContent = `C-ENGINE ONLINE (v${data.c_engine_version})`;
        return;
      }
      throw new Error(data?.error || 'Unhealthy');
    } catch (err) {
      state.engineHealthy = false;
      elements.statusBadge.className = 'health-badge offline';
      elements.statusText.textContent = 'C-ENGINE OFFLINE';
    }
  }

  // ── 8. Grid and Triplet View Renderers ─────────────────────────────────────
  function renderMatrixGrid(tableElem, mat, matrixName) {
    tableElem.innerHTML = '';
    const totalCells = mat.rows * mat.cols;

    if (totalCells > state.renderThreshold) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = 5;
      td.style.padding = '24px';
      td.style.textAlign = 'center';
      td.style.color = 'var(--text-secondary)';
      td.textContent = `Grid size (${mat.rows}×${mat.cols} = ${totalCells} cells) exceeds interactive rendering limit of ${state.renderThreshold.toLocaleString()} cells. Switch to TRIPLETS view to inspect non-zero entries.`;
      tr.appendChild(td);
      tableElem.appendChild(tr);
      return;
    }

    // Header row with column coordinates (1-based display)
    const headerRow = document.createElement('tr');
    const cornerTh = document.createElement('th');
    cornerTh.textContent = 'R\\C';
    cornerTh.className = 'row-idx';
    headerRow.appendChild(cornerTh);

    for (let c = 1; c <= mat.cols; c++) {
      const th = document.createElement('th');
      th.textContent = c;
      th.setAttribute('scope', 'col');
      headerRow.appendChild(th);
    }
    tableElem.appendChild(headerRow);

    // Grid rows with row coordinates (1-based display)
    for (let r = 1; r <= mat.rows; r++) {
      const tr = document.createElement('tr');
      const rowTh = document.createElement('th');
      rowTh.textContent = r;
      rowTh.className = 'row-idx';
      rowTh.setAttribute('scope', 'row');
      tr.appendChild(rowTh);

      for (let c = 1; c <= mat.cols; c++) {
        const td = document.createElement('td');
        const key = `${r}_${c}`;
        const val = mat.cells[key] !== undefined ? mat.cells[key] : 0.0;

        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.className = 'matrix-cell-input ' + (val === 0.0 ? 'cell-zero' : 'cell-nonzero');
        input.value = val === 0.0 ? '0' : String(val);
        input.setAttribute('aria-label', `${matrixName} Row ${r}, Column ${c}`);

        input.addEventListener('change', (e) => {
          const num = parseFloat(e.target.value);
          let parsed = 0.0;
          if (isNaN(num) || !isFinite(num)) {
            parsed = 0.0;
            e.target.value = '0';
          } else {
            parsed = num;
          }

          if (parsed === 0.0) {
            delete mat.cells[key];
          } else {
            mat.cells[key] = parsed;
          }

          input.value = parsed === 0.0 ? '0' : String(parsed);
          input.className = 'matrix-cell-input ' + (parsed === 0.0 ? 'cell-zero' : 'cell-nonzero');
          renderTripletsTable(matrixName);
          markStale();

        });

        td.appendChild(input);
        tr.appendChild(td);
      }
      tableElem.appendChild(tr);
    }
  }

  function renderTripletsTable(matrixName) {
    const isA = matrixName === 'Matrix A';
    const tbody = isA ? elements.tbodyATriplets : elements.tbodyBTriplets;
    const mat = isA ? state.matrixA : state.matrixB;

    tbody.innerHTML = '';
    let count = 0;

    for (let r = 1; r <= mat.rows; r++) {
      for (let c = 1; c <= mat.cols; c++) {
        const v = mat.cells[`${r}_${c}`];
        if (v !== undefined && v !== 0.0) {
          count++;
          const tr = document.createElement('tr');
          const wire = displayToWire(r, c);
          tr.innerHTML = `
            <td>${count}</td>
            <td><strong>${r}</strong> <span style="font-size: 10px; color: var(--text-muted);">(wire: ${wire.row})</span></td>
            <td><strong>${c}</strong> <span style="font-size: 10px; color: var(--text-muted);">(wire: ${wire.col})</span></td>
            <td style="color: ${isA ? 'var(--accent-cobalt)' : 'var(--accent-orange-dark)'}; font-weight: 700;">${v}</td>
          `;
          tbody.appendChild(tr);
        }
      }
    }

    if (count === 0) {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--text-muted); padding: 12px;">All entries are implicit zero (NNZ: 0).</td>`;
      tbody.appendChild(tr);
    }
  }

  // ── 9. Result Inspector Presentation ──────────────────────────────────────
  function getActiveTabId() {
    const activeBtn = document.querySelector('.tab-btn.active');
    return activeBtn ? activeBtn.getAttribute('data-tab') : 'tab-dense';
  }

  function renderActiveResultTab() {
    const activeTab = getActiveTabId();
    if (activeTab === 'tab-dense') renderResultDense();
    else if (activeTab === 'tab-triplets') renderResultTriplets();
    else if (activeTab === 'tab-stats') renderResultStats();
    else if (activeTab === 'tab-memory') renderResultMemory();
  }

  function renderResultDense() {
    const container = elements.tabDense;
    container.innerHTML = '';

    if (!state.lastOperation || !state.lastOperation.result) {
      container.innerHTML = `
        <div class="empty-state">
          <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2"></rect>
            <line x1="3" y1="9" x2="21" y2="9"></line>
            <line x1="9" y1="3" x2="9" y2="21"></line>
          </svg>
          <span class="empty-state-title">AWAITING OPERATION STREAM</span>
          <p class="empty-state-desc">Select an operation from the Command Deck above. The compiled C11 CLI engine will execute the mathematics and render the output matrix here.</p>
        </div>
      `;
      return;
    }

    const res = state.lastOperation.result;
    const rows = res.rows;
    const cols = res.cols;
    const totalCells = rows * cols;

    if (totalCells > state.renderThreshold) {
      container.innerHTML = `
        <div class="empty-state" style="padding: 24px;">
          <span class="empty-state-title" style="color: var(--accent-orange);">LARGE RESULT MATRIX</span>
          <p class="empty-state-desc">
            Output dimension (${rows}×${cols} = ${totalCells.toLocaleString()} cells) exceeds the visual drafting limit (${state.renderThreshold.toLocaleString()} cells).
            All mathematical values are preserved in memory. Switch to the <strong>TRIPLET STREAM</strong> tab to view entries.
          </p>
        </div>
      `;
      return;
    }

    if (state.lastOperation.isStale) {
      const banner = document.createElement('div');
      banner.className = 'result-stale-banner';
      banner.setAttribute('role', 'alert');
      banner.innerHTML = `<span>⚠️</span> <div><strong>STALE RESULT:</strong> Input matrices were modified since this calculation was executed. Recalculate to refresh.</div>`;
      container.appendChild(banner);
    }

    // Build grid from returned result (either res.matrix from to_dense, or res.triplets from sparse ops)
    const scrollWrapper = document.createElement('div');
    scrollWrapper.className = 'matrix-scroll-wrapper';
    scrollWrapper.tabIndex = 0;
    scrollWrapper.setAttribute('role', 'region');
    scrollWrapper.setAttribute('aria-label', 'Operation result matrix grid');
    const table = document.createElement('table');
    table.className = 'matrix-grid-table';

    // Header row
    const hr = document.createElement('tr');
    const corner = document.createElement('th');
    corner.textContent = 'R\\C';
    corner.className = 'row-idx';
    hr.appendChild(corner);
    for (let c = 1; c <= cols; c++) {
      const th = document.createElement('th');
      th.textContent = c;
      hr.appendChild(th);
    }
    table.appendChild(hr);

    // Fast lookup for cells
    let cellLookup = null;
    if (res.matrix && Array.isArray(res.matrix)) {
      cellLookup = (r0, c0) => res.matrix[r0] ? res.matrix[r0][c0] : 0.0;
    } else {
      const map = new Map();
      if (Array.isArray(res.triplets)) {
        res.triplets.forEach(t => map.set(`${t.row}_${t.col}`, t.value));
      }
      cellLookup = (r0, c0) => map.get(`${r0}_${c0}`) || 0.0;
    }

    for (let r = 1; r <= rows; r++) {
      const tr = document.createElement('tr');
      const rowTh = document.createElement('th');
      rowTh.textContent = r;
      rowTh.className = 'row-idx';
      tr.appendChild(rowTh);

      for (let c = 1; c <= cols; c++) {
        const td = document.createElement('td');
        const wire = displayToWire(r, c);
        const val = cellLookup(wire.row, wire.col);

        const input = document.createElement('input');
        input.type = 'text';
        input.readOnly = true;
        input.className = 'matrix-cell-input ' + (val === 0.0 ? 'cell-zero' : 'cell-nonzero');
        input.value = val === 0.0 ? '0' : String(val);
        input.title = `Result [${r}, ${c}] = ${val}`;

        td.appendChild(input);
        tr.appendChild(td);
      }
      table.appendChild(tr);
    }

    scrollWrapper.appendChild(table);
    container.appendChild(scrollWrapper);
  }

  function renderResultTriplets() {
    const container = elements.tabTriplets;
    container.innerHTML = '';

    if (!state.lastOperation || !state.lastOperation.result) {
      container.innerHTML = `
        <div class="empty-state">
          <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <line x1="8" y1="6" x2="21" y2="6"></line>
            <line x1="8" y1="12" x2="21" y2="12"></line>
            <line x1="8" y1="18" x2="21" y2="18"></line>
            <line x1="3" y1="6" x2="3.01" y2="6"></line>
            <line x1="3" y1="12" x2="3.01" y2="12"></line>
            <line x1="3" y1="18" x2="3.01" y2="18"></line>
          </svg>
          <span class="empty-state-title">NO TRIPLETS TO DISPLAY</span>
          <p class="empty-state-desc">When a sparse matrix operation completes, canonical row-major sorted (row, col, value) triplets will appear here.</p>
        </div>
      `;
      return;
    }

    const res = state.lastOperation.result;
    let triplets = [];

    if (Array.isArray(res.triplets)) {
      triplets = res.triplets;
    } else if (res.cells && Array.isArray(res.cells)) {
      // For to_dense: filter nonzero cells
      triplets = res.cells.filter(c => c.value !== 0.0);
    }

    if (triplets.length === 0) {
      container.innerHTML = `
        <div class="empty-state" style="padding: 24px;">
          <span class="empty-state-title">ALL-ZERO MATRIX (NNZ: 0)</span>
          <p class="empty-state-desc">The result matrix has dimensions ${res.rows}×${res.cols} with zero non-zero entries. No triplets are stored in memory.</p>
        </div>
      `;
      return;
    }

    if (state.lastOperation.isStale) {
      const banner = document.createElement('div');
      banner.className = 'result-stale-banner';
      banner.setAttribute('role', 'alert');
      banner.innerHTML = `<span>⚠️</span> <div><strong>STALE RESULT:</strong> Input matrices were modified since this calculation was executed. Recalculate to refresh.</div>`;
      container.appendChild(banner);
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'triplet-table-wrapper';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', 'Operation result triplets');
    const table = document.createElement('table');
    table.className = 'triplet-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>#</th>
          <th>ROW (1-BASED)</th>
          <th>COL (1-BASED)</th>
          <th>WIRE INDEX (0-BASED)</th>
          <th>VALUE</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');
    triplets.forEach((t, i) => {
      const disp = wireToDisplay(t.row, t.col);
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><strong>${disp.row}</strong></td>
        <td><strong>${disp.col}</strong></td>
        <td style="color: var(--text-muted); font-family: var(--font-mono); font-size: 10px;">[${t.row}, ${t.col}]</td>
        <td style="color: var(--accent-cobalt); font-weight: 700;">${t.value}</td>
      `;
      tbody.appendChild(tr);
    });

    wrapper.appendChild(table);
    container.appendChild(wrapper);
  }

  function renderResultStats() {
    const container = elements.tabStats;
    container.innerHTML = '';

    if (!state.lastOperation || !state.lastOperation.result) {
      container.innerHTML = `
        <div class="empty-state">
          <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          <span class="empty-state-title">DIAGNOSTICS IDLE</span>
          <p class="empty-state-desc">Dimensions, non-zero element count, zero count, and execution diagnostics will be recorded upon computation.</p>
        </div>
      `;
      return;
    }

    const op = state.lastOperation;
    const res = op.result;
    const totalCells = res.rows * res.cols;
    let nnz = res.nnz;
    if (nnz === undefined) {
      if (Array.isArray(res.triplets)) nnz = res.triplets.length;
      else if (Array.isArray(res.cells)) nnz = res.cells.filter(c => c.value !== 0.0).length;
      else nnz = 0;
    }
    const zeros = totalCells - nnz;
    const density = totalCells > 0 ? (nnz / totalCells) * 100 : 0;
    const sparsity = totalCells > 0 ? (zeros / totalCells) * 100 : 0;

    const isStale = op.isStale;

    container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: var(--space-md);">
        ${isStale ? `
          <div style="background-color: var(--accent-orange-subtle); border: 1px solid var(--accent-orange); color: var(--accent-orange); padding: 8px 12px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 11px;">
            ⚠️ <strong>INPUT MODIFIED:</strong> The matrix input has been edited since this operation was executed. Recalculate to refresh.
          </div>
        ` : ''}

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: var(--space-xs);">
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">OPERATION</span>
            <span class="metric-value" style="color: var(--text-ink); font-size: 12px;">${op.type.toUpperCase()}</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">DIMENSIONS</span>
            <span class="metric-value" style="color: var(--text-ink);">${res.rows} × ${res.cols}</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">TOTAL CELLS</span>
            <span class="metric-value" style="color: var(--text-ink);">${totalCells.toLocaleString()}</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">NONZERO (NNZ)</span>
            <span class="metric-value" style="color: var(--accent-cobalt);">${nnz.toLocaleString()}</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">ZERO CELLS</span>
            <span class="metric-value" style="color: var(--text-ink);">${zeros.toLocaleString()}</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">DENSITY</span>
            <span class="metric-value" style="color: var(--text-ink);">${density.toFixed(2)}%</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">SPARSITY</span>
            <span class="metric-value" style="color: var(--text-ink);">${sparsity.toFixed(2)}%</span>
          </div>
          <div class="metric-card" style="background: var(--surface-subtle); border-color: var(--border-rule);">
            <span class="metric-label" style="color: var(--text-secondary);">COMPUTED BY</span>
            <span class="metric-value" style="color: var(--accent-green); font-size: 11px;">C11 ENGINE</span>
          </div>
        </div>
      </div>
    `;
  }

  function renderResultMemory() {
    const container = elements.tabMemory;
    container.innerHTML = '';

    if (!state.lastAnalysis || !state.lastAnalysis.result) {
      container.innerHTML = `
        <div class="empty-state">
          <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect>
            <rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect>
            <line x1="6" y1="6" x2="6.01" y2="6"></line>
            <line x1="6" y1="18" x2="6.01" y2="18"></line>
          </svg>
          <span class="empty-state-title">MEMORY COMPARISON PENDING</span>
          <p class="empty-state-desc">Click <strong>ANALYZE MEMORY</strong> on the Command Deck to compute theoretical memory payloads and savings in the C engine.</p>
        </div>
      `;
      return;
    }

    const ma = state.lastAnalysis.result;
    const isStale = state.lastAnalysis.isStale;

    container.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: var(--space-md);">
        ${isStale ? `
          <div style="background-color: var(--accent-orange-subtle); border: 1px solid var(--accent-orange); color: var(--accent-orange); padding: 8px 12px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 11px;">
            ⚠️ <strong>STALE METRICS:</strong> Input matrices were modified. Click <strong>ANALYZE MEMORY</strong> to recalculate in C.
          </div>
        ` : ''}

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-md);">
          <div style="background: var(--surface-subtle); border: 1px solid var(--border-rule); border-radius: var(--radius-sm); padding: var(--space-sm);">
            <h4 style="font-family: var(--font-mono); font-size: 11px; margin-bottom: 8px; color: var(--text-secondary);">DENSE PAYLOAD MODEL</h4>
            <p style="font-size: 12px; margin-bottom: 4px;"><strong>${ma.total_cells.toLocaleString()} cells × 8 bytes</strong> (IEEE 754)</p>
            <p style="font-family: var(--font-mono); font-size: 18px; font-weight: 700; color: var(--text-ink);">${ma.dense_payload_bytes.toLocaleString()} Bytes</p>
          </div>

          <div style="background: var(--surface-subtle); border: 1px solid var(--border-rule); border-radius: var(--radius-sm); padding: var(--space-sm);">
            <h4 style="font-family: var(--font-mono); font-size: 11px; margin-bottom: 8px; color: var(--text-secondary);">SPARSE TRIPLET MODEL</h4>
            <p style="font-size: 12px; margin-bottom: 4px;"><strong>${ma.nnz.toLocaleString()} entries × 16 bytes</strong> (2×32-bit + double)</p>
            <p style="font-family: var(--font-mono); font-size: 18px; font-weight: 700; color: var(--accent-cobalt);">${ma.sparse_payload_bytes.toLocaleString()} Bytes</p>
          </div>
        </div>

        <div style="background: var(--surface-sheet); border: 1px solid var(--border-strong); border-radius: var(--radius-sm); padding: var(--space-sm); font-family: var(--font-mono); font-size: 12px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span>VERDICT:</span>
            <span style="font-weight: 700; color: ${ma.sparse_is_smaller ? 'var(--accent-green)' : (ma.savings_bytes === 0 ? 'var(--text-secondary)' : 'var(--accent-orange)')};">
              ${ma.sparse_is_smaller ? `SPARSE WINS (${ma.percent_saved.toFixed(2)}% SAVINGS)` : (ma.savings_bytes === 0 ? 'IDENTICAL FOOTPRINT' : 'DENSE IS SMALLER')}
            </span>
          </div>
          <div style="font-size: 11px; color: var(--text-secondary);">
            Breakeven threshold is exactly 50% density. Current density is <strong>${ma.density_percent.toFixed(2)}%</strong>.
          </div>
        </div>
      </div>
    `;
  }

  // Update the dark telemetry panel in the right sidebar
  function updateTelemetryPanel(ma) {
    if (!ma) return;
    elements.metricDims.textContent = `${ma.rows} × ${ma.cols}`;
    elements.metricTotalCells.textContent = ma.total_cells.toLocaleString();
    elements.metricNnz.textContent = ma.nnz.toLocaleString();
    elements.metricZeros.textContent = ma.zero_count.toLocaleString();
    elements.metricDensity.textContent = `${ma.density_percent.toFixed(2)}%`;
    elements.metricSparsity.textContent = `${ma.sparsity_percent.toFixed(2)}%`;
    elements.metricDenseBytes.textContent = `${ma.dense_payload_bytes.toLocaleString()} B`;
    elements.metricSparseBytes.textContent = `${ma.sparse_payload_bytes.toLocaleString()} B`;

    if (ma.sparse_is_smaller) {
      elements.metricSavingsBadge.style.color = '#38E090';
      elements.metricSavingsBadge.textContent = `SAVINGS: ${ma.savings_bytes.toLocaleString()} B (${ma.percent_saved.toFixed(2)}%)`;
      elements.scaleBarFill.style.backgroundColor = 'var(--accent-cobalt)';
    } else if (ma.savings_bytes === 0) {
      elements.metricSavingsBadge.style.color = 'var(--text-dark-secondary)';
      elements.metricSavingsBadge.textContent = `IDENTICAL: 0 B SAVINGS`;
      elements.scaleBarFill.style.backgroundColor = 'var(--text-dark-secondary)';
    } else {
      elements.metricSavingsBadge.style.color = 'var(--accent-orange)';
      elements.metricSavingsBadge.textContent = `EXTRA OVERHEAD: +${Math.abs(ma.savings_bytes).toLocaleString()} B`;
      elements.scaleBarFill.style.backgroundColor = 'var(--accent-orange)';
    }

    const ratio = ma.dense_payload_bytes > 0 ? (ma.sparse_payload_bytes / ma.dense_payload_bytes) * 100 : 0;
    const clampedRatio = Math.min(100, Math.max(0, ratio));
    elements.scaleBarFill.style.width = `${clampedRatio}%`;

    if (elements.scaleBarTrack) {
      elements.scaleBarTrack.setAttribute('aria-valuenow', Math.round(clampedRatio));
      elements.scaleBarTrack.setAttribute('aria-valuetext', `${Math.round(ratio)}% of dense footprint`);
    }
  }

  // ── 10. Operational Command Deck Handlers ──────────────────────────────────
  async function runOperation(opName) {
    // 1. Basic UI validation
    if (opName === 'add' || opName === 'multiply') {
      if (!state.matrixB.visible) {
        setMatrixBVisibility(true);
      }
      if (opName === 'add') {
        if (state.matrixA.rows !== state.matrixB.rows || state.matrixA.cols !== state.matrixB.cols) {
          updateActivity('ERROR', `Addition dimension mismatch: Matrix A (${state.matrixA.rows}×${state.matrixA.cols}) cannot be added to Matrix B (${state.matrixB.rows}×${state.matrixB.cols}).`);
          return;
        }
      } else if (opName === 'multiply') {
        if (state.matrixA.cols !== state.matrixB.rows) {
          updateActivity('ERROR', `Multiplication dimension mismatch: Matrix A columns (${state.matrixA.cols}) must match Matrix B rows (${state.matrixB.rows}).`);
          return;
        }
      }
    }

    try {
      const matAPayload = matrixToDensePayload(state.matrixA);
      const matBPayload = (opName === 'add' || opName === 'multiply') ? matrixToDensePayload(state.matrixB) : null;

      const data = await apiClient.postOperation(opName, matAPayload, matBPayload);

      state.lastOperation = {
        type: opName,
        result: data,
        timestamp: new Date(),
        isStale: false
      };

      if (elements.resultBadge) {
        elements.resultBadge.textContent = `${opName.toUpperCase()} RESULT`;
      }
      if (elements.resultStaleBadge) {
        elements.resultStaleBadge.style.display = 'none';
      }

      // Switch to relevant result tab if appropriate
      if (opName === 'to_dense') {
        switchResultTab('tab-dense');
      } else {
        switchResultTab('tab-triplets');
      }

      renderActiveResultTab();
      updateActivity('SUCCESS', `Operation '${opName.toUpperCase()}' completed by C11 engine (NNZ: ${data.nnz !== undefined ? data.nnz : (data.cells ? data.cells.filter(c => c.value !== 0).length : 'N/A')}).`);
    } catch (err) {
      updateActivity('ERROR', `Operation '${opName}' failed: ${err.message} (${err.code})`);
    }
  }

  async function runMemoryAnalysis() {
    try {
      const matAPayload = matrixToDensePayload(state.matrixA);
      const data = await apiClient.postAnalyze(matAPayload);

      state.lastAnalysis = {
        result: data,
        timestamp: new Date(),
        isStale: false
      };

      if (elements.telemetryStaleBadge) {
        elements.telemetryStaleBadge.style.display = 'none';
      }

      updateTelemetryPanel(data);
      if (getActiveTabId() === 'tab-memory') {
        renderResultMemory();
      }

      updateActivity('SUCCESS', `Theoretical memory analysis computed by C engine (Density: ${data.density_percent.toFixed(2)}%).`);
    } catch (err) {
      updateActivity('ERROR', `Memory analysis failed: ${err.message} (${err.code})`);
    }
  }

  // ── 12. Preset Sample Loading ─────────────────────────────────────────────
  async function loadPresetSample() {
    const presetKey = elements.samplePresetSelect.value;
    try {
      clearDimensionErrors('both');
      const samples = await apiClient.getSamples();

      if (presetKey === 'sparse_sample' && samples.sparse_sample) {
        const sample = samples.sparse_sample;
        state.matrixA.rows = sample.rows;
        state.matrixA.cols = sample.cols;
        state.matrixA.cells = {};
        sample.entries.forEach(e => {
          const disp = wireToDisplay(e.row, e.col);
          state.matrixA.cells[`${disp.row}_${disp.col}`] = e.value;
        });

        elements.dimARows.value = sample.rows;
        elements.dimACols.value = sample.cols;
        renderMatrixGrid(elements.tableMatrixA, state.matrixA, 'Matrix A');
        renderTripletsTable('Matrix A');

        markStale();
        updateActivity('SUCCESS', `Loaded Sparse Sample (${sample.rows}×${sample.cols}, NNZ: ${sample.entries.length}) from C catalog.`);
      } else if (presetKey === 'addition_pair' && samples.addition_pair) {
        const pair = samples.addition_pair;
        // Matrix A
        state.matrixA.rows = pair.matrix_a.rows;
        state.matrixA.cols = pair.matrix_a.cols;
        state.matrixA.cells = {};
        pair.matrix_a.data.forEach((rowVals, r) => {
          rowVals.forEach((val, c) => {
            if (val !== 0.0) state.matrixA.cells[`${r + 1}_${c + 1}`] = val;
          });
        });
        elements.dimARows.value = pair.matrix_a.rows;
        elements.dimACols.value = pair.matrix_a.cols;

        // Matrix B
        state.matrixB.rows = pair.matrix_b.rows;
        state.matrixB.cols = pair.matrix_b.cols;
        state.matrixB.cells = {};
        pair.matrix_b.data.forEach((rowVals, r) => {
          rowVals.forEach((val, c) => {
            if (val !== 0.0) state.matrixB.cells[`${r + 1}_${c + 1}`] = val;
          });
        });
        elements.dimBRows.value = pair.matrix_b.rows;
        elements.dimBCols.value = pair.matrix_b.cols;

        setMatrixBVisibility(true);
        renderMatrixGrid(elements.tableMatrixA, state.matrixA, 'Matrix A');
        renderMatrixGrid(elements.tableMatrixB, state.matrixB, 'Matrix B');
        renderTripletsTable('Matrix A');
        renderTripletsTable('Matrix B');

        markStale();
        updateActivity('SUCCESS', `Loaded Addition Compatible Pair (A: 2×3, B: 2×3). Ready for A + B.`);
      } else if (presetKey === 'multiplication_pair' && samples.multiplication_pair) {
        const pair = samples.multiplication_pair;
        state.matrixA.rows = pair.matrix_a.rows;
        state.matrixA.cols = pair.matrix_a.cols;
        state.matrixA.cells = {};
        pair.matrix_a.data.forEach((rowVals, r) => {
          rowVals.forEach((val, c) => {
            if (val !== 0.0) state.matrixA.cells[`${r + 1}_${c + 1}`] = val;
          });
        });
        elements.dimARows.value = pair.matrix_a.rows;
        elements.dimACols.value = pair.matrix_a.cols;

        state.matrixB.rows = pair.matrix_b.rows;
        state.matrixB.cols = pair.matrix_b.cols;
        state.matrixB.cells = {};
        pair.matrix_b.data.forEach((rowVals, r) => {
          rowVals.forEach((val, c) => {
            if (val !== 0.0) state.matrixB.cells[`${r + 1}_${c + 1}`] = val;
          });
        });
        elements.dimBRows.value = pair.matrix_b.rows;
        elements.dimBCols.value = pair.matrix_b.cols;

        setMatrixBVisibility(true);
        renderMatrixGrid(elements.tableMatrixA, state.matrixA, 'Matrix A');
        renderMatrixGrid(elements.tableMatrixB, state.matrixB, 'Matrix B');
        renderTripletsTable('Matrix A');
        renderTripletsTable('Matrix B');

        markStale();
        updateActivity('SUCCESS', `Loaded Rectangular Multiplication Pair (A: 2×3, B: 3×2). Inner dim matches.`);
      }
    } catch (err) {
      updateActivity('ERROR', `Failed to load preset: ${err.message}`);
    }
  }

  function setMatrixBVisibility(visible) {
    state.matrixB.visible = visible;
    elements.panelMatrixB.style.display = visible ? 'flex' : 'none';
    elements.btnToggleMatrixB.setAttribute('aria-pressed', visible ? 'true' : 'false');
    elements.btnToggleMatrixB.textContent = visible ? '- HIDE MATRIX B' : '+ SHOW MATRIX B';
  }

  // ── 13. Paste Support ─────────────────────────────────────────────────────
  let lastFocusedTrigger = null;

  function openPasteModal(target) {
    lastFocusedTrigger = target === 'A' ? elements.btnPasteA : elements.btnPasteB;
    state.activePasteTarget = target;
    elements.pasteTargetLabel.textContent = `MATRIX ${target}`;
    elements.pasteTextarea.value = '';
    elements.pasteErrorMsg.style.display = 'none';
    elements.pasteDialog.showModal();
    elements.pasteTextarea.focus();
  }

  function handleApplyPaste() {
    const rawText = elements.pasteTextarea.value.trim();
    if (!rawText) {
      elements.pasteErrorMsg.textContent = 'Paste text cannot be empty.';
      elements.pasteErrorMsg.style.display = 'block';
      return;
    }

    const lines = rawText.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) {
      elements.pasteErrorMsg.textContent = 'No non-empty lines found.';
      elements.pasteErrorMsg.style.display = 'block';
      return;
    }

    const parsedRows = [];
    let expectedCols = -1;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      // Delimiter: tab if present, otherwise comma or whitespace
      let tokens = [];
      if (line.includes('\t')) {
        tokens = line.split('\t').map(t => t.trim()).filter(t => t.length > 0);
      } else if (line.includes(',')) {
        tokens = line.split(',').map(t => t.trim()).filter(t => t.length > 0);
      } else {
        tokens = line.split(/\s+/).filter(t => t.length > 0);
      }

      if (tokens.length === 0) continue;

      if (expectedCols === -1) {
        expectedCols = tokens.length;
      } else if (tokens.length !== expectedCols) {
        elements.pasteErrorMsg.textContent = `Ragged row detected at line ${i + 1}: expected ${expectedCols} columns, found ${tokens.length}.`;
        elements.pasteErrorMsg.style.display = 'block';
        return;
      }

      const rowNums = [];
      for (let j = 0; j < tokens.length; j++) {
        const n = parseFloat(tokens[j]);
        if (isNaN(n) || !isFinite(n)) {
          elements.pasteErrorMsg.textContent = `Invalid numeric value '${tokens[j]}' at row ${i + 1}, column ${j + 1}.`;
          elements.pasteErrorMsg.style.display = 'block';
          return;
        }
        rowNums.push(n);
      }
      parsedRows.push(rowNums);
    }

    const newRows = parsedRows.length;
    const newCols = expectedCols;

    if (newRows < 1 || newCols < 1 || newRows > 100 || newCols > 100) {
      elements.pasteErrorMsg.textContent = `Dimensions (${newRows}×${newCols}) must be between 1×1 and 100×100.`;
      elements.pasteErrorMsg.style.display = 'block';
      return;
    }

    const mat = state.activePasteTarget === 'A' ? state.matrixA : state.matrixB;
    mat.rows = newRows;
    mat.cols = newCols;
    mat.cells = {};

    for (let r = 0; r < newRows; r++) {
      for (let c = 0; c < newCols; c++) {
        const val = parsedRows[r][c];
        if (val !== 0.0) {
          mat.cells[`${r + 1}_${c + 1}`] = val;
        }
      }
    }

    if (state.activePasteTarget === 'A') {
      elements.dimARows.value = newRows;
      elements.dimACols.value = newCols;
      renderMatrixGrid(elements.tableMatrixA, mat, 'Matrix A');
      renderTripletsTable('Matrix A');
    } else {
      elements.dimBRows.value = newRows;
      elements.dimBCols.value = newCols;
      renderMatrixGrid(elements.tableMatrixB, mat, 'Matrix B');
      renderTripletsTable('Matrix B');
    }

    clearDimensionErrors(state.activePasteTarget);
    elements.pasteDialog.close();
    markStale();
    updateActivity('SUCCESS', `Pasted ${newRows}×${newCols} matrix into Matrix ${state.activePasteTarget}.`);
  }

  // ── 14. Export & Copy Handlers ────────────────────────────────────────────
  function exportDenseCsv() {
    let rows = 0;
    let cols = 0;
    let getVal = null;

    if (state.lastOperation && state.lastOperation.result) {
      const res = state.lastOperation.result;
      rows = res.rows;
      cols = res.cols;
      if (res.matrix) {
        getVal = (r0, c0) => res.matrix[r0] ? res.matrix[r0][c0] : 0.0;
      } else {
        const map = new Map();
        if (Array.isArray(res.triplets)) res.triplets.forEach(t => map.set(`${t.row}_${t.col}`, t.value));
        getVal = (r0, c0) => map.get(`${r0}_${c0}`) || 0.0;
      }
    } else {
      rows = state.matrixA.rows;
      cols = state.matrixA.cols;
      getVal = (r0, c0) => state.matrixA.cells[`${r0 + 1}_${c0 + 1}`] || 0.0;
    }

    const lines = [];
    for (let r0 = 0; r0 < rows; r0++) {
      const rowVals = [];
      for (let c0 = 0; c0 < cols; c0++) {
        rowVals.push(getVal(r0, c0));
      }
      lines.push(rowVals.join(','));
    }

    downloadFile(lines.join('\n'), 'zeroform_dense_matrix.csv', 'text/csv');
    updateActivity('SUCCESS', `Exported dense matrix (${rows}×${cols}) as CSV.`);
  }

  function exportTripletsCsv() {
    let triplets = [];
    if (state.lastOperation && state.lastOperation.result && Array.isArray(state.lastOperation.result.triplets)) {
      triplets = state.lastOperation.result.triplets;
    } else {
      // Export Matrix A triplets
      for (let r = 1; r <= state.matrixA.rows; r++) {
        for (let c = 1; c <= state.matrixA.cols; c++) {
          const v = state.matrixA.cells[`${r}_${c}`];
          if (v !== undefined && v !== 0.0) {
            const w = displayToWire(r, c);
            triplets.push({ row: w.row, col: w.col, value: v });
          }
        }
      }
    }

    const lines = ['row_display,col_display,row_wire,col_wire,value'];
    triplets.forEach(t => {
      const disp = wireToDisplay(t.row, t.col);
      lines.push(`${disp.row},${disp.col},${t.row},${t.col},${t.value}`);
    });

    downloadFile(lines.join('\n'), 'zeroform_sparse_triplets.csv', 'text/csv');
    updateActivity('SUCCESS', `Exported ${triplets.length} triplets as CSV.`);
  }

  function copyTripletsToClipboard() {
    let triplets = [];
    if (state.lastOperation && state.lastOperation.result && Array.isArray(state.lastOperation.result.triplets)) {
      triplets = state.lastOperation.result.triplets;
    } else {
      for (let r = 1; r <= state.matrixA.rows; r++) {
        for (let c = 1; c <= state.matrixA.cols; c++) {
          const v = state.matrixA.cells[`${r}_${c}`];
          if (v !== undefined && v !== 0.0) {
            const w = displayToWire(r, c);
            triplets.push({ row: w.row, col: w.col, value: v });
          }
        }
      }
    }

    if (triplets.length === 0) {
      updateActivity('IDLE', 'No nonzero triplets to copy (all zeros).');
      return;
    }

    const lines = [
      '# ZEROFORM Sparse Triplet Stream (Canonical Row-Major)',
      '# ROW(1-based)\tCOL(1-based)\tWIRE[r,c]\tVALUE'
    ];
    triplets.forEach(t => {
      const disp = wireToDisplay(t.row, t.col);
      lines.push(`${disp.row}\t${disp.col}\t[${t.row}, ${t.col}]\t${t.value}`);
    });

    const text = lines.join('\n');
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        updateActivity('SUCCESS', `Copied ${triplets.length} triplets to clipboard.`);
      }).catch(() => {
        fallbackCopy(text);
      });
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      updateActivity('SUCCESS', 'Copied triplets to clipboard (fallback).');
    } catch (e) {
      updateActivity('ERROR', 'Unable to access system clipboard.');
    }
    document.body.removeChild(ta);
  }

  function downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ── 15. Reset Workspace ───────────────────────────────────────────────────
  function openResetModal() {
    lastFocusedTrigger = elements.opReset;
    elements.resetDialog.showModal();
  }

  function executeResetWorkspace() {
    state.matrixA = {
      rows: 3,
      cols: 3,
      cells: {},
      view: 'grid'
    };
    state.matrixB = {
      rows: 3,
      cols: 3,
      cells: {},
      visible: false,
      view: 'grid'
    };
    state.lastOperation = null;
    state.lastAnalysis = null;

    elements.dimARows.value = 3;
    elements.dimACols.value = 3;
    elements.dimBRows.value = 3;
    elements.dimBCols.value = 3;
    clearDimensionErrors('both');

    setMatrixBVisibility(false);
    renderMatrixGrid(elements.tableMatrixA, state.matrixA, 'Matrix A');
    renderMatrixGrid(elements.tableMatrixB, state.matrixB, 'Matrix B');
    renderTripletsTable('Matrix A');
    renderTripletsTable('Matrix B');

    // Reset Result Inspector
    clearStale();
    if (elements.resultBadge) elements.resultBadge.textContent = 'ENGINE STDOUT STREAM';
    renderActiveResultTab();

    // Reset Memory Telemetry
    elements.metricDims.textContent = '3 × 3';
    elements.metricTotalCells.textContent = '9';
    elements.metricNnz.textContent = '0';
    elements.metricZeros.textContent = '9';
    elements.metricDensity.textContent = '0.00%';
    elements.metricSparsity.textContent = '100.00%';
    elements.metricDenseBytes.textContent = '72 B';
    elements.metricSparseBytes.textContent = '0 B';
    elements.metricSavingsBadge.style.color = '#38E090';
    elements.metricSavingsBadge.textContent = 'SAVINGS: 72 B (100.00%)';
    elements.scaleBarFill.style.width = '0%';

    updateActivity('IDLE', 'Workspace reset to default zero matrices.');
    elements.resetDialog.close();
  }

  // ── 16. Tab Management ────────────────────────────────────────────────────
  function switchResultTab(targetTabId) {
    elements.tabButtons.forEach(btn => {
      if (btn.getAttribute('data-tab') === targetTabId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    elements.tabContents.forEach(content => {
      if (content.id === targetTabId) {
        content.classList.add('active');
      } else {
        content.classList.remove('active');
      }
    });

    renderActiveResultTab();
  }

  function initTabs() {
    elements.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-tab');
        switchResultTab(targetId);
      });
    });
  }

  // ── 17. View Switchers & Resizing ─────────────────────────────────────────
  function initViewSwitchers() {
    // Matrix A View Switchers
    elements.btnViewADense.addEventListener('click', () => {
      state.matrixA.view = 'grid';
      elements.containerAGrid.style.display = 'block';
      elements.containerATriplets.style.display = 'none';
      elements.btnViewADense.classList.add('active');
      elements.btnViewADense.setAttribute('aria-pressed', 'true');
      elements.btnViewATriplets.classList.remove('active');
      elements.btnViewATriplets.setAttribute('aria-pressed', 'false');
    });

    elements.btnViewATriplets.addEventListener('click', () => {
      state.matrixA.view = 'triplets';
      elements.containerAGrid.style.display = 'none';
      elements.containerATriplets.style.display = 'block';
      elements.btnViewATriplets.classList.add('active');
      elements.btnViewATriplets.setAttribute('aria-pressed', 'true');
      elements.btnViewADense.classList.remove('active');
      elements.btnViewADense.setAttribute('aria-pressed', 'false');
      renderTripletsTable('Matrix A');
    });

    // Matrix B View Switchers
    elements.btnViewBDense.addEventListener('click', () => {
      state.matrixB.view = 'grid';
      elements.containerBGrid.style.display = 'block';
      elements.containerBTriplets.style.display = 'none';
      elements.btnViewBDense.classList.add('active');
      elements.btnViewBDense.setAttribute('aria-pressed', 'true');
      elements.btnViewBTriplets.classList.remove('active');
      elements.btnViewBTriplets.setAttribute('aria-pressed', 'false');
    });

    elements.btnViewBTriplets.addEventListener('click', () => {
      state.matrixB.view = 'triplets';
      elements.containerBGrid.style.display = 'none';
      elements.containerBTriplets.style.display = 'block';
      elements.btnViewBTriplets.classList.add('active');
      elements.btnViewBTriplets.setAttribute('aria-pressed', 'true');
      elements.btnViewBDense.classList.remove('active');
      elements.btnViewBDense.setAttribute('aria-pressed', 'false');
      renderTripletsTable('Matrix B');
    });

    // Toggle Matrix B
    elements.btnToggleMatrixB.addEventListener('click', () => {
      setMatrixBVisibility(!state.matrixB.visible);
    });

    // Resize Matrix A & Keyboard / Input Handlers
    elements.btnResizeA.addEventListener('click', () => applyMatrixResize('A'));
    elements.dimARows.addEventListener('input', () => {
      elements.dimARows.classList.remove('invalid');
      elements.dimARows.removeAttribute('aria-invalid');
    });
    elements.dimACols.addEventListener('input', () => {
      elements.dimACols.classList.remove('invalid');
      elements.dimACols.removeAttribute('aria-invalid');
    });
    elements.dimARows.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        applyMatrixResize('A');
      }
    });
    elements.dimACols.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        applyMatrixResize('A');
      }
    });

    // Resize Matrix B & Keyboard / Input Handlers
    elements.btnResizeB.addEventListener('click', () => applyMatrixResize('B'));
    elements.dimBRows.addEventListener('input', () => {
      elements.dimBRows.classList.remove('invalid');
      elements.dimBRows.removeAttribute('aria-invalid');
    });
    elements.dimBCols.addEventListener('input', () => {
      elements.dimBCols.classList.remove('invalid');
      elements.dimBCols.removeAttribute('aria-invalid');
    });
    elements.dimBRows.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        applyMatrixResize('B');
      }
    });
    elements.dimBCols.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        applyMatrixResize('B');
      }
    });

    // Paste Buttons
    elements.btnPasteA.addEventListener('click', () => openPasteModal('A'));
    elements.btnPasteB.addEventListener('click', () => openPasteModal('B'));
  }

  // ── 18. Dialogs and Modals Management ─────────────────────────────────────
  function initDialogs() {
    // Help Spec Dialog
    if (elements.btnOpenHelp && elements.helpDialog) {
      elements.btnOpenHelp.addEventListener('click', () => {
        lastFocusedTrigger = elements.btnOpenHelp;
        elements.helpDialog.showModal();
      });
      elements.helpDialog.addEventListener('close', () => {
        if (lastFocusedTrigger) {
          lastFocusedTrigger.focus();
          lastFocusedTrigger = null;
        }
      });
    }
    if (elements.btnCloseHelp && elements.helpDialog) {
      elements.btnCloseHelp.addEventListener('click', () => elements.helpDialog.close());
    }
    if (elements.btnDismissHelp && elements.helpDialog) {
      elements.btnDismissHelp.addEventListener('click', () => elements.helpDialog.close());
    }

    // Reset Dialog
    if (elements.resetDialog) {
      elements.resetDialog.addEventListener('close', () => {
        if (lastFocusedTrigger) {
          lastFocusedTrigger.focus();
          lastFocusedTrigger = null;
        }
      });
    }
    if (elements.btnCloseReset) {
      elements.btnCloseReset.addEventListener('click', () => elements.resetDialog.close());
    }
    if (elements.btnCancelReset) {
      elements.btnCancelReset.addEventListener('click', () => elements.resetDialog.close());
    }
    if (elements.btnConfirmReset) {
      elements.btnConfirmReset.addEventListener('click', executeResetWorkspace);
    }

    // Paste Dialog
    if (elements.pasteDialog) {
      elements.pasteDialog.addEventListener('close', () => {
        if (lastFocusedTrigger) {
          lastFocusedTrigger.focus();
          lastFocusedTrigger = null;
        }
      });
    }
    if (elements.btnClosePaste) {
      elements.btnClosePaste.addEventListener('click', () => elements.pasteDialog.close());
    }
    if (elements.btnCancelPaste) {
      elements.btnCancelPaste.addEventListener('click', () => elements.pasteDialog.close());
    }
    if (elements.btnApplyPaste) {
      elements.btnApplyPaste.addEventListener('click', handleApplyPaste);
    }
  }

  // ── 19. Bind Command Deck Listeners ────────────────────────────────────────
  function initCommandDeck() {
    // Command Deck Operations
    elements.opToSparse.addEventListener('click', () => runOperation('to_sparse'));
    elements.opToDense.addEventListener('click', () => runOperation('to_dense'));
    elements.opTranspose.addEventListener('click', () => runOperation('transpose'));
    elements.opAdd.addEventListener('click', () => runOperation('add'));
    elements.opMultiply.addEventListener('click', () => runOperation('multiply'));
    elements.opAnalyze.addEventListener('click', runMemoryAnalysis);
    elements.opReset.addEventListener('click', openResetModal);

    // Presets
    elements.btnLoadPreset.addEventListener('click', loadPresetSample);

    // Exports
    elements.btnExportDense.addEventListener('click', exportDenseCsv);
    elements.btnExportTriplets.addEventListener('click', exportTripletsCsv);
    elements.btnCopyTriplets.addEventListener('click', copyTripletsToClipboard);
  }

  // ── 20. Initialization Entrypoint ─────────────────────────────────────────
  function init() {
    // Initial sample grid values: 3x3 with (1,1)=1.0, (2,3)=2.0
    state.matrixA.cells['1_1'] = 1.0;
    state.matrixA.cells['2_3'] = 2.0;

    renderMatrixGrid(elements.tableMatrixA, state.matrixA, 'Matrix A');
    renderMatrixGrid(elements.tableMatrixB, state.matrixB, 'Matrix B');
    renderTripletsTable('Matrix A');
    renderTripletsTable('Matrix B');

    initTabs();
    initDialogs();
    initViewSwitchers();
    initCommandDeck();

    // Check engine health immediately
    checkEngineHealth();

    // Periodic health heartbeat every 25 seconds
    setInterval(checkEngineHealth, 25000);

    updateActivity('IDLE', 'ZEROFORM Studio shell ready. C11 engine connected via Flask subprocess.');
  }

  // Launch on DOM ready
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
    } else {
      init();
    }
  }

  // Expose validation helpers for automated regression testing
  if (typeof window !== 'undefined') {
    window.ZEROFORM_VALIDATORS = {
      LIMITS,
      parseStrictDimension,
      validateMatrixDimensions
    };
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      LIMITS,
      parseStrictDimension,
      validateMatrixDimensions
    };
  }

})();

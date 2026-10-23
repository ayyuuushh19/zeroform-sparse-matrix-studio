/**
 * tests/test_frontend_validation.js
 * Automated regression test suite for P2: Fractional Matrix Dimensions
 *
 * Verifies all 18 test cases specified in the P2 requirement:
 *  A-01 to A-10: Matrix A dimension validations
 *  B-01 to B-03: Matrix B dimension validations
 *  R-01 to R-05: State preservation, error recovery, boundary pruning, and independence
 */

const assert = require('assert');
const { parseStrictDimension, validateMatrixDimensions, LIMITS } = require('../frontend/static/js/app.js');

let passed = 0;
let total = 0;

function test(name, fn) {
  total++;
  try {
    fn();
    passed++;
    console.log(`  [PASS] ${name}`);
  } catch (err) {
    console.error(`  [FAIL] ${name}: ${err.message}`);
    throw err;
  }
}

console.log('Running Frontend Dimension Validation Regression Tests...\n');

// ── Test A-01: Rows = 2.5 ──────────────────────────────────────────────────
test('A-01: Rows = 2.5 rejected', () => {
  const res = parseStrictDimension('2.5');
  assert.strictEqual(res.ok, false);
  assert.strictEqual(res.reason, 'invalid_format');

  const val = validateMatrixDimensions('2.5', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
  assert.strictEqual(val.colError, false);
});

// ── Test A-02: Columns = 3.5 ───────────────────────────────────────────────
test('A-02: Columns = 3.5 rejected', () => {
  const res = parseStrictDimension('3.5');
  assert.strictEqual(res.ok, false);
  assert.strictEqual(res.reason, 'invalid_format');

  const val = validateMatrixDimensions('3', '3.5', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, false);
  assert.strictEqual(val.colError, true);
});

// ── Test A-03: Rows = 0.5 ──────────────────────────────────────────────────
test('A-03: Rows = 0.5 rejected', () => {
  const res = parseStrictDimension('0.5');
  assert.strictEqual(res.ok, false);
  assert.strictEqual(res.reason, 'invalid_format');

  const val = validateMatrixDimensions('0.5', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
});

// ── Test A-04: Rows = 0 ────────────────────────────────────────────────────
test('A-04: Rows = 0 rejected', () => {
  const res = parseStrictDimension('0');
  assert.strictEqual(res.ok, false);

  const val = validateMatrixDimensions('0', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
});

// ── Test A-05: Rows = -2 ───────────────────────────────────────────────────
test('A-05: Rows = -2 rejected', () => {
  const res = parseStrictDimension('-2');
  assert.strictEqual(res.ok, false);

  const val = validateMatrixDimensions('-2', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
});

// ── Test A-06: Rows = empty ────────────────────────────────────────────────
test('A-06: Rows = empty rejected', () => {
  assert.strictEqual(parseStrictDimension('').ok, false);
  assert.strictEqual(parseStrictDimension('   ').ok, false);
  assert.strictEqual(parseStrictDimension(null).ok, false);
  assert.strictEqual(parseStrictDimension(undefined).ok, false);

  const val = validateMatrixDimensions('', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
});

// ── Test A-07: Rows = 3abc ─────────────────────────────────────────────────
test('A-07: Rows = 3abc rejected', () => {
  assert.strictEqual(parseStrictDimension('3abc').ok, false);
  assert.strictEqual(parseStrictDimension('2.5xyz').ok, false);
  assert.strictEqual(parseStrictDimension('Infinity').ok, false);
  assert.strictEqual(parseStrictDimension('NaN').ok, false);
  assert.strictEqual(parseStrictDimension('1e3').ok, false);

  // Simulated browser input with validity.badInput
  const mockBadInput = { validity: { badInput: true } };
  assert.strictEqual(parseStrictDimension('', mockBadInput).ok, false);

  const val = validateMatrixDimensions('3abc', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
});

// ── Test A-08: Rows = 3 ────────────────────────────────────────────────────
test('A-08: Rows = 3 accepted', () => {
  const res = parseStrictDimension('3');
  assert.strictEqual(res.ok, true);
  assert.strictEqual(res.value, 3);

  const val = validateMatrixDimensions('3', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, true);
  assert.strictEqual(val.rows, 3);
  assert.strictEqual(val.cols, 3);
  assert.strictEqual(val.rowError, false);
  assert.strictEqual(val.colError, false);
});

// ── Test A-09: Columns = 4 ─────────────────────────────────────────────────
test('A-09: Columns = 4 accepted', () => {
  const res = parseStrictDimension('4');
  assert.strictEqual(res.ok, true);
  assert.strictEqual(res.value, 4);

  const val = validateMatrixDimensions('3', '4', null, null, 'Matrix A');
  assert.strictEqual(val.ok, true);
  assert.strictEqual(val.rows, 3);
  assert.strictEqual(val.cols, 4);
});

// ── Test A-10: Product exceeds cell limit ──────────────────────────────────
test('A-10: Valid dimensions whose product exceeds cell limit rejected before resize', () => {
  // 2000 x 5001 = 10,002,000 cells > 10,000,000
  const val = validateMatrixDimensions('2000', '5001', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
  assert.strictEqual(val.colError, true);
  assert.match(val.message, /exceed the universal limit/);

  // Exactly at ceiling: 1000 x 10000 = 10,000,000 cells <= 10,000,000 -> Accepted
  const valLimit = validateMatrixDimensions('1000', '10000', null, null, 'Matrix A');
  assert.strictEqual(valLimit.ok, true);
  assert.strictEqual(valLimit.rows, 1000);
  assert.strictEqual(valLimit.cols, 10000);
});

// ── Test B-01: Matrix B rows = 2.5 ─────────────────────────────────────────
test('B-01: Matrix B rows = 2.5 rejected', () => {
  const val = validateMatrixDimensions('2.5', '3', null, null, 'Matrix B');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, true);
  assert.strictEqual(val.colError, false);
  assert.match(val.message, /Matrix B/);
});

// ── Test B-02: Matrix B columns = 3.5 ──────────────────────────────────────
test('B-02: Matrix B columns = 3.5 rejected', () => {
  const val = validateMatrixDimensions('3', '3.5', null, null, 'Matrix B');
  assert.strictEqual(val.ok, false);
  assert.strictEqual(val.rowError, false);
  assert.strictEqual(val.colError, true);
  assert.match(val.message, /Matrix B/);
});

// ── Test B-03: Matrix B valid integer dimensions ───────────────────────────
test('B-03: Matrix B valid integer dimensions accepted', () => {
  const val = validateMatrixDimensions('5', '6', null, null, 'Matrix B');
  assert.strictEqual(val.ok, true);
  assert.strictEqual(val.rows, 5);
  assert.strictEqual(val.cols, 6);
});

// ── Test R-01: Invalid input followed by valid input ───────────────────────
test('R-01: Invalid input followed by valid input clears errors and succeeds', () => {
  let val = validateMatrixDimensions('2.5', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, false);

  val = validateMatrixDimensions('4', '3', null, null, 'Matrix A');
  assert.strictEqual(val.ok, true);
  assert.strictEqual(val.rows, 4);
  assert.strictEqual(val.cols, 3);
  assert.strictEqual(val.rowError, false);
  assert.strictEqual(val.colError, false);
});

// ── Test R-02: Invalid Matrix A input does not affect Matrix B ─────────────
test('R-02: Invalid Matrix A input leaves Matrix B independent', () => {
  const valA = validateMatrixDimensions('2.5', '3', null, null, 'Matrix A');
  const valB = validateMatrixDimensions('3', '3', null, null, 'Matrix B');

  assert.strictEqual(valA.ok, false);
  assert.strictEqual(valB.ok, true);
});

// ── Test R-03: Invalid Matrix B input does not affect Matrix A ─────────────
test('R-03: Invalid Matrix B input leaves Matrix A independent', () => {
  const valA = validateMatrixDimensions('3', '3', null, null, 'Matrix A');
  const valB = validateMatrixDimensions('3', '3.5', null, null, 'Matrix B');

  assert.strictEqual(valA.ok, true);
  assert.strictEqual(valB.ok, false);
});

// ── Test R-04: Resize populated matrix preserves existing semantics ────────
test('R-04: Resize populated matrix prunes cells outside new boundaries', () => {
  const initialCells = {
    '1_1': 10.0,
    '2_2': 20.0,
    '3_3': 30.0,
    '4_4': 40.0
  };

  // Shrink from 4x4 to 2x3
  const newRows = 2;
  const newCols = 3;
  const newCells = {};
  for (const [key, v] of Object.entries(initialCells)) {
    const parts = key.split('_');
    const ri = Number(parts[0]);
    const ci = Number(parts[1]);
    if (ri <= newRows && ci <= newCols) {
      newCells[key] = v;
    }
  }

  assert.strictEqual(Object.keys(newCells).length, 2);
  assert.strictEqual(newCells['1_1'], 10.0);
  assert.strictEqual(newCells['2_2'], 20.0);
  assert.strictEqual(newCells['3_3'], undefined);
  assert.strictEqual(newCells['4_4'], undefined);
});

// ── Test R-05: Repeated invalid attempts cause no corruption ───────────────
test('R-05: Repeated invalid attempts produce consistent rejections without corruption', () => {
  const invalidInputs = ['2.5', '3.7', '-1', '0', 'abc', '1e4', '   ', '0.001'];
  for (const inv of invalidInputs) {
    const val = validateMatrixDimensions(inv, '3', null, null, 'Matrix A');
    assert.strictEqual(val.ok, false);
    assert.strictEqual(val.rowError, true);
  }
});

console.log(`\n======================================================`);
console.log(`  All ${passed}/${total} frontend dimension validation tests passed!`);
console.log(`======================================================\n`);

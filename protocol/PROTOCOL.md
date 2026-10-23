# ZEROFORM — Binary/Text Inter-Process Protocol Specification

Version: **v1**  
Scope: Communication between Python (Flask backend) and the compiled C engine (`zeroform.exe`).

---

## 1. Architectural Principles

1. **Strict Process Separation**: The C engine executes mathematical algorithms and data structures exclusively. Python handles HTTP routing, JSON marshaling, and process coordination.
2. **Standard Stream I/O**: Requests are streamed to the C engine via standard input (`stdin`); responses are written to standard output (`stdout`). Diagnostics and logging go to `stderr`.
3. **No HTTP in C**: The C engine is completely decoupled from HTTP, WebSockets, or network semantics.
4. **No Third-Party JSON Parser in C**: Communication uses a deterministic, line-oriented key-value protocol that is safely parsed in standard C11 without external dependencies.
5. **No Python Fallback Calculation**: If the C engine encounters an error or crashes, Python reports the error to the client. Python never calculates fallback matrix results.

---

## 2. Framing Rules

### Request Framing:
```text
CMD <VERB> v1
D <KEY> <VALUE...>
...
END
```

- Every request begins with a header: `CMD <VERB> v1`.
- If the version token is not `v1` or the verb is unknown, the engine responds with `ERR`.
- Every data line begins with `D ` followed by an uppercase key and one or more space-separated values.
- The request terminates with a single line containing `END`.
- Any trailing data after `END` is rejected.

### Response Framing:
Success response:
```text
OK
D <KEY> <VALUE...>
...
END
```

Error response:
```text
ERR
D CODE <ZF_STATUS_CODE>
D MESSAGE <HUMAN_READABLE_MESSAGE>
END
```

Error codes map 1:1 to the C `ZF_Status` enumeration defined in `zeroform.h`:
- `ZF_OK`
- `ZF_ERR_NOMEM`
- `ZF_ERR_INDEX`
- `ZF_ERR_EMPTY`
- `ZF_ERR_PARAM`
- `ZF_ERR_OVERFLOW`
- `ZF_ERR_DUPLICATE`
- `ZF_ERR_NOT_FOUND`
- `ZF_ERR_VALUE`
- `ZF_ERR_DIMENSION`

---

## 3. Supported Verbs

| Command Verb | Required Parameters | Purpose |
|---|---|---|
| `PING` | None | Health check; responds with `D STATUS PONG` and `D VERSION 1.0.0`. |
| `ANALYZE` | Matrix A (Dense or Sparse) | Calculates dimensions, NNZ, density, sparsity, and theoretical payload estimates. |
| `TO_SPARSE` | Matrix A (Dense or Sparse) | Converts dense matrix to canonical row-major sorted Triplet list. |
| `TO_DENSE` | Matrix A (Sparse or Dense) | Converts sparse matrix to full flat dense row-major matrix. |
| `ADD` | Matrix A, Matrix B | Computes $C = A + B$ in $O(\text{nnzA} + \text{nnzB})$ time. |
| `MULTIPLY` | Matrix A, Matrix B | Computes $C = A \times B$ using row-indexed accumulator algorithm. |
| `TRANSPOSE` | Matrix A | Computes $C = A^T$ in $O(\text{cols} + \text{nnz})$ time. |
| `INSERT` | Matrix A, `ROW`, `COL`, `VALUE` | Inserts a new nonzero entry; rejects duplicates with `ZF_ERR_DUPLICATE`. |
| `SET` | Matrix A, `ROW`, `COL`, `VALUE` | Sets coordinate value; updates existing, inserts if absent, deletes if 0.0. |
| `DELETE` | Matrix A, `ROW`, `COL` | Deletes coordinate; returns `ZF_ERR_NOT_FOUND` if absent. |
| `SEARCH` | Matrix A, `ROW`, `COL` | Binary search coordinate; returns `D FOUND 0\|1`, `D VALUE <v>`, `D POS <p>`. |
| `TRAVERSE` | Matrix A | Visits all entries in row-major order and returns triplets. |
| `SORT` | Matrix A | Sorts triplets in canonical row-major order. |

---

## 4. Matrix Serialization Formats

### Dense Matrix Format (`DENSE`)
Used to submit complete matrices cell-by-cell.
```text
D A_ROWS 2
D A_COLS 3
D A_FORMAT DENSE
D A_CELL 0 0 1.0
D A_CELL 0 1 0.0
D A_CELL 0 2 2.0
D A_CELL 1 0 0.0
D A_CELL 1 1 3.0
D A_CELL 1 2 0.0
```
- Every cell $(r, c)$ in $[0, \text{rows}-1] \times [0, \text{cols}-1]$ must appear exactly once.
- Missing or duplicate cells are rejected with `ZF_ERR_PARAM`.

### Sparse Matrix Format (`SPARSE`)
Used to submit nonzero triplets.
```text
D A_ROWS 2
D A_COLS 3
D A_FORMAT SPARSE
D A_NNZ 2
D A_ENTRY 0 0 1.0
D A_ENTRY 0 2 2.0
```
- Number of `A_ENTRY` lines must match `A_NNZ`.
- Entries may arrive unordered; the C engine validates and sorts them.
- Explicit zeros (`0`, `0.0`, `-0.0`), out-of-bounds coordinates, and non-finite values are rejected.
- Implicit zeros are valid and represented by omitted coordinates (they are not stored).
- Canonical sparse matrices contain only nonzero stored values; dense matrices may contain zero cells, and dense-to-sparse conversion omits them.

For commands taking a second matrix $B$ (such as `ADD` or `MULTIPLY`), the prefix `B_` is used (`B_ROWS`, `B_COLS`, `B_FORMAT`, `B_CELL`, `B_ENTRY`).

---

## 5. Parameter Fields for Mutation/Search Commands

- `D ROW <r>` : Target row index (zero-based integer).
- `D COL <c>` : Target column index (zero-based integer).
- `D VALUE <v>` : Numeric value (finite IEEE 754 double).
- `D ACTION <act>` : Optional action descriptor (`INSERT`, `SET`, `DELETE`, `SEARCH`).

---

## 6. Limits and Safety Bounds

- Maximum line length: **1,024 bytes** (overlong lines rejected).
- Maximum total cells ($rows \times cols$): **10,000,000**.
- Maximum nonzero entries ($NNZ$): **1,000,000**.
- Maximum dimension ($rows, cols$): **100,000**.
- Subprocess timeout: **5.0 seconds** (enforced by Python caller).

---

## 7. Error Codes and HTTP Status Mapping

The C engine communicates machine-readable errors via `D CODE <ZF_Status>`:

| C Status Code | Enum Value | Meaning | HTTP Status Code | Description |
|---|---|---|---|---|
| `ZF_OK` | 0 | Operation succeeded | `200 OK` | Successful response |
| `ZF_ERR_PARAM` | 4 | Invalid parameter / missing field | `400 Bad Request` | Malformed request syntax |
| `ZF_ERR_NOT_FOUND` | 7 | Coordinate absent in sparse matrix | `404 Not Found` | Coordinate not found on delete |
| `ZF_ERR_OVERFLOW` | 5 | Dimension or cell count exceeds limit | `413 Payload Too Large` | Limit exceeded |
| `ZF_ERR_INDEX` | 2 | Coordinate out of matrix bounds | `422 Unprocessable Entity` | Invalid coordinate index |
| `ZF_ERR_DUPLICATE` | 6 | Duplicate coordinate on insert | `422 Unprocessable Entity` | Duplicate coordinate conflict |
| `ZF_ERR_DIMENSION` | 9 | Dimension mismatch in Add/Multiply | `422 Unprocessable Entity` | Incompatible matrix shapes |
| `ZF_ERR_VALUE` | 8 | Non-finite number (NaN, Inf) | `422 Unprocessable Entity` | Floating-point validation error |
| `ZF_ERR_NOMEM` | 1 | Out of memory | `500 Internal Server Error` | Allocation failure |
| `ZF_ERR_INTERNAL` | 10 | Unhandled internal error | `500 Internal Server Error` | Unexpected error |

Subprocess / Engine level mappings in Flask:
- `EngineNotFoundError`: `503 Service Unavailable`
- `EngineTimeoutError`: `504 Gateway Timeout`
- `EngineCrashError`: `500 Internal Server Error`
- `ProtocolFramingError` / `ProtocolParseError`: `500 Internal Server Error`

---

## 8. Example Transactions

### 8.1 Matrix Addition with Cancellation
**Request:**
```text
CMD ADD v1
D A_ROWS 2
D A_COLS 3
D A_FORMAT DENSE
D A_CELL 0 0 1
D A_CELL 0 1 0
D A_CELL 0 2 2
D A_CELL 1 0 0
D A_CELL 1 1 3
D A_CELL 1 2 0
D B_ROWS 2
D B_COLS 3
D B_FORMAT DENSE
D B_CELL 0 0 -1
D B_CELL 0 1 4
D B_CELL 0 2 0
D B_CELL 1 0 5
D B_CELL 1 1 -3
D B_CELL 1 2 7
END
```

**Response:**
```text
OK
D ROWS 2
D COLS 3
D NNZ 4
D ENTRY 0 1 4
D ENTRY 0 2 2
D ENTRY 1 0 5
D ENTRY 1 2 7
END
```

### 8.2 Theoretical Memory Analysis
**Request:**
```text
CMD ANALYZE v1
D A_ROWS 3
D A_COLS 4
D A_FORMAT DENSE
D A_CELL 0 0 0
D A_CELL 0 1 0
D A_CELL 0 2 0
D A_CELL 0 3 0
D A_CELL 1 0 0
D A_CELL 1 1 0
D A_CELL 1 2 0
D A_CELL 1 3 0
D A_CELL 2 0 0
D A_CELL 2 1 0
D A_CELL 2 2 0
D A_CELL 2 3 0
END
```

**Response:**
```text
OK
D ROWS 3
D COLS 4
D TOTAL_CELLS 12
D NNZ 0
D ZERO_COUNT 12
D DENSITY 0.0000000000
D SPARSITY 100.0000000000
D DENSE_BYTES 96
D SPARSE_BYTES 0
D SPARSE_IS_SMALLER 1
D SAVINGS_BYTES 96
D PERCENT_SAVED 100.0000000000
END
```

### 8.3 Theoretical Memory Model & Fixed Payload Assumptions

The `ANALYZE` command evaluates idealized academic theoretical storage footprints:
- **Dense Theoretical Payload**: $\text{total\_cells} \times 8\text{ bytes}$ (one 64-bit IEEE 754 double per cell).
- **Sparse Theoretical Payload**: $\text{nnz} \times 16\text{ bytes}$ (one 64-bit double [8 bytes] + two 32-bit integer indices [4 bytes row + 4 bytes col]).

**Architectural Distinction from Runtime C Struct Layout:**
In the compiled 64-bit C engine (`x86_64-w64-mingw32`), the runtime in-memory `Triplet` structure uses 64-bit `size_t` coordinates:
```c
typedef struct {
    size_t row;    /* 8 bytes on x64 */
    size_t col;    /* 8 bytes on x64 */
    double value;  /* 8 bytes */
} Triplet;         /* sizeof(Triplet) == 24 bytes in RAM */
```

In the compiled C engine, the actual runtime Triplet backing allocation is `capacity * sizeof(Triplet)` (where `sizeof(Triplet)` is 24 bytes on 64-bit platforms), accompanied by `DynamicArray` and `SparseMatrix` struct metadata. Actual process memory also entails OS allocator headers and runtime page overhead.
In contrast, the theoretical payload model uses fixed constants (`THEORETICAL_DENSE_BYTES_PER_CELL = 8ULL`, `THEORETICAL_SPARSE_BYTES_PER_ENTRY = 16ULL`) per the academic course specification, evaluating idealized algorithmic payload sizes without allocation overhead.



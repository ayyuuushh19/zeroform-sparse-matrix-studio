# ZEROFORM — Sparse Matrix Studio

**ZEROFORM** is a high-performance Sparse Matrix Studio developed for B.Tech CSE Semester 3 Data Structures. It demonstrates standard C11 dynamic memory management, canonical Triplet-form sparse matrix compression, fast matrix arithmetic, memory footprint analysis, and an isolated Flask REST API backend.

---

## Architecture & Separation of Concerns

The project adheres to a strict multi-tier architecture:

```text
[ Client (Browser / HTTP) ]
            │  JSON / HTTP
            ▼
[ Python Flask Backend ]   (app.py, validator.py, protocol.py, engine.py)
            │
            │  subprocess (shell=False, stdin/stdout text protocol v1)
            ▼
[ C CLI Engine ]           (zeroform.exe: DynamicArray, SparseMatrix, DenseMatrix)
```

1. **C Engine (`zeroform.exe`)**:
   - Implements all data structures (`DynamicArray`, `DenseMatrix`, `SparseMatrix`).
   - Implements all matrix operations (addition, multiplication, transpose, search, mutation, sort).
   - Computes theoretical memory payload metrics and savings.
   - Absolutely zero HTTP or JSON parsing code.
2. **Python Flask Backend (`backend/`)**:
   - Validates incoming HTTP payloads, dimensions, coordinates, and finiteness.
   - Serializes requests into deterministic line-oriented v1 protocol.
   - Coordinates subprocess execution using `subprocess.run(shell=False)` with a 5.0s default timeout.
   - Deserializes C engine output and returns clean, uniform JSON responses.
   - **Zero matrix math fallback in Python**: Python never calculates mathematical operations. If C errors or fails, an appropriate error code (e.g. 422, 503, 504) is returned to the client.

---

## 1. C Engine Build Instructions

### Prerequisites
- **OS**: Windows 11 x64
- **Compiler**: GCC 15.2.0 (MSYS2 / mingw64)
- **Make Tool**: GNU Make 4.4.1 (invoked as `mingw32-make`)

### Building the C Engine
From the repository root:
```powershell
cd D:\DSA_MINI_PROJECT\c_engine
mingw32-make clean
mingw32-make build
```

This compiles all library objects and links the CLI engine executable:
```
D:\DSA_MINI_PROJECT\c_engine\zeroform.exe
```

Compiler flags enforced:
```text
-std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2
```

### Running C Unit Tests
```powershell
# Run all 4 C test suites (DynamicArray, Dense/Sparse Matrix, Operations, Memory Analysis)
mingw32-make test

# Run all test suites with -ftrapv signed integer overflow detection
mingw32-make sanitize
```
**Test Results**: 670/670 C checks passed across all 4 suites with 0 failures (DynamicArray: 257, Matrices: 210, Operations: 113, Memory Analysis: 90).

---

## 2. Python Virtual Environment & Flask Setup

### Prerequisites
- Python 3.14 x64
- Flask 3.1.3
- pytest 9.1.1

### Setup & Activation
From repository root (`D:\DSA_MINI_PROJECT`):
```powershell
# Create project-local virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

### Starting the Flask Application
```powershell
.\.venv\Scripts\python.exe -m backend.app
```
The server binds to `http://127.0.0.1:5000/`.

### Running Python Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
**Test Results**: 78/78 pytest tests passed across `test_protocol.py`, `test_engine.py`, `test_api.py`, and `test_frontend_integration.py`.

---

## 3. Protocol Specification (`v1`)

The communication between Flask and `zeroform.exe` is deterministic, line-oriented, and UTF-8 encoded.

### Framing Grammar
```text
CMD <VERB> v1
D <KEY> <VALUE...>
...
END
```

Responses:
```text
OK
D <KEY> <VALUE...>
...
END
```
or on error:
```text
ERR
D CODE <ZF_STATUS_CODE>
D MESSAGE <TEXT>
END
```

### Supported Command Verbs
- `PING` — Engine liveness check.
- `ANALYZE` — Compute dimensions, NNZ, density, sparsity, and memory payload bytes.
- `TO_SPARSE` — Convert dense matrix into canonical row-major sorted triplets.
- `TO_DENSE` — Convert sparse triplets into flat 2D cell grid.
- `ADD` — Sparse matrix addition $C = A + B$.
- `MULTIPLY` — Sparse matrix multiplication $C = A \times B$.
- `TRANSPOSE` — Sparse matrix transpose $C = A^T$.
- `INSERT` — Insert new nonzero entry (rejects duplicates).
- `SET` — Set value (updates existing, inserts if absent, deletes if 0.0).
- `DELETE` — Remove coordinate (returns `ZF_ERR_NOT_FOUND` if absent).
- `SEARCH` — Binary search coordinate (`D FOUND 1|0`).
- `TRAVERSE` — In-order traversal of all stored triplets.
- `SORT` — Canonical row-major sorting of triplets.

### System Safety Limits
- Max line length: `1,024` bytes
- Max dimensions: `100,000` rows/cols
- Max total cells: `10,000,000`
- Max non-zero entries (NNZ): `1,000,000`
- Subprocess timeout: `5.0` seconds

---

## 4. API Endpoints & Status Code Mapping

### HTTP Status Code Mapping

| Status Code | Cause / Meaning | C Status Code |
|---|---|---|
| `200 OK` | Successful execution | `ZF_OK` |
| `400 Bad Request` | Missing JSON body, ragged matrix, negative dimension | `ZF_ERR_PARAM` |
| `404 Not Found` | Coordinate absent during deletion | `ZF_ERR_NOT_FOUND` |
| `413 Payload Too Large` | Dimensions or cell count exceed limits | `ZF_ERR_OVERFLOW` |
| `422 Unprocessable Entity` | Dimension mismatch, duplicate coord, NaN/Inf value | `ZF_ERR_DIMENSION`, `ZF_ERR_DUPLICATE`, `ZF_ERR_INDEX`, `ZF_ERR_VALUE` |
| `500 Internal Server Error` | C process crash, memory exhaustion, protocol syntax failure | `ZF_ERR_NOMEM`, `ZF_ERR_INTERNAL` |
| `503 Service Unavailable` | C executable missing or fails PING | `EngineNotFoundError` |
| `504 Gateway Timeout` | C engine process exceeded execution timeout | `EngineTimeoutError` |

---

### REST Routes Summary

#### `GET /api/health`
Checks if `zeroform.exe` responds to `PING`.
**Response (200 OK):**
```json
{
  "status": "healthy",
  "engine": "connected",
  "c_engine_version": "1.0.0",
  "executable": "zeroform.exe"
}
```

#### `GET /api/samples`
Returns sample inputs for sparse matrices, addition pairs, and multiplication pairs.

#### `POST /api/analyze`
Computes theoretical memory analysis using the C core.

**Theoretical Payload Model**:
- **Dense Theoretical Payload**: $\text{total\_cells} \times 8\text{ bytes}$ (one 64-bit IEEE 754 double per cell).
- **Sparse Theoretical Payload**: $\text{nnz} \times 16\text{ bytes}$ (fixed theoretical assumption: 8-byte double + 4-byte 32-bit row + 4-byte 32-bit col index).
- **Architectural Distinction**: In 64-bit Windows x64 compiled C memory, the Triplet backing allocation is `capacity * sizeof(Triplet)` (where `sizeof(Triplet)` is 24 bytes due to 64-bit `size_t` coordinates), together with matrix and dynamic array metadata. Actual process memory additionally involves heap allocation headers and OS runtime overhead that the theoretical model does not measure. The theoretical model evaluates idealized algorithmic payload sizes per the academic course specification (`total_cells * 8` vs `nnz * 16`).

**Request:**
```json
{
  "matrix": [
    [1.0, 0.0, 2.0],
    [0.0, 3.0, 0.0]
  ]
}
```
**Response (200 OK):**
```json
{
  "rows": 2,
  "cols": 3,
  "total_cells": 6,
  "nnz": 3,
  "zero_count": 3,
  "density_percent": 50.0,
  "sparsity_percent": 50.0,
  "dense_payload_bytes": 48,
  "sparse_payload_bytes": 48,
  "sparse_is_smaller": false,
  "savings_bytes": 0,
  "percent_saved": 0.0,
  "triplets": [
    {"row": 0, "col": 0, "value": 1.0},
    {"row": 0, "col": 2, "value": 2.0},
    {"row": 1, "col": 1, "value": 3.0}
  ]
}
```

#### `POST /api/operation`
Executes matrix operations (`to_sparse`, `to_dense`, `add`, `multiply`, `transpose`).
**Request (Addition):**
```json
{
  "operation": "add",
  "matrix_a": {
    "rows": 2,
    "cols": 3,
    "format": "dense",
    "data": [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]
  },
  "matrix_b": {
    "rows": 2,
    "cols": 3,
    "format": "dense",
    "data": [[-1.0, 4.0, 0.0], [5.0, -3.0, 7.0]]
  }
}
```
**Response (200 OK):**
```json
{
  "operation": "add",
  "rows": 2,
  "cols": 3,
  "nnz": 4,
  "triplets": [
    {"row": 0, "col": 1, "value": 4.0},
    {"row": 0, "col": 2, "value": 2.0},
    {"row": 1, "col": 0, "value": 5.0},
    {"row": 1, "col": 2, "value": 7.0}
  ]
}
```

#### `POST /api/entry`
Executes single-element mutations (`insert`, `set`, `delete`, `search`, `traverse`, `sort`).
**Request (Search):**
```json
{
  "action": "search",
  "matrix": {
    "rows": 3,
    "cols": 3,
    "format": "sparse",
    "entries": [{"row": 1, "col": 2, "value": 4.5}]
  },
  "row": 1,
  "col": 2
}
```
**Response (200 OK):**
```json
{
  "action": "search",
  "found": true,
  "row": 1,
  "col": 2,
  "value": 4.5,
  "pos": 0
}
```

---

## 5. Stage 5 Frontend Design System & Application Shell

### Architecture & Directory Structure
The frontend application shell is housed exclusively under `frontend/`:
```text
frontend/
├── templates/
│   └── index.html      # Main engineering workbench shell
└── static/
    ├── css/
    │   └── style.css   # Engineering design system & token definitions
    └── js/
        └── app.js      # Presentation logic, matrix drafting grid, and health polling
```

### Visual Identity & Design Tokens
Inspired by precision scientific instruments, architectural graph paper, and modern developer workbenches:
- **Canvas / Surfaces**: Warm ivory (`#F4F1E8`), white drafting sheets (`#FFFFFF`), pale technical grey (`#EAE7DD`), and deep technical telemetry dark panels (`#171B22`).
- **Typography & Rules**: Deep graphite primary ink (`#20252D`), technical slate (`#586070`), and 1px drafting rule borders (`rgba(32, 37, 45, 0.14)`).
- **Accents**: Electric cobalt (`#2855E8`) for primary commands, signal orange (`#E9783D`) for highlights, verified online green (`#1E8E5A`), and alert red (`#D9383A`).
- **Coordinate Conventions**: 1-based display $(r \in [1, R], c \in [1, C])$ for user convenience, converted automatically to 0-based wire indices for C engine protocol requests.
- **Visual Grid Rendering Threshold**: 2,500 cells (e.g. 50×50 interactive table in `frontend/static/js/app.js`). Matrices exceeding 2,500 cells display a guard notification directing the user to the Triplet List view, preventing browser DOM performance degradation while allowing the C engine and backend to process up to 10,000,000 cells.
- **Zero Remote Dependencies**: Self-contained CSS and vanilla JS; uses native system fonts and clean inline SVG icons with no external CDNs or Google Fonts.

### Application Shell Components
1. **Header**: Brand mark, wordmark **ZEROFORM**, subtitle **SPARSE MATRIX STUDIO**, live health check badge connected to `GET /api/health`, and accessible architecture specification modal.
2. **Preset Control Strip**: Quick-load sample datasets from `GET /api/samples` (Mostly-Zero 4×5, Addition Pair 2×3, Rectangular Multiply Pair 2×3 × 3×2).
3. **Matrix Workspace**: Interactive drafting grid with 1-based row and column coordinate headers, inline cell editing, and format toggling between Dense Grid and Triplet List views. Dual-matrix mode toggles Matrix B for binary arithmetic.
4. **Operation Command Deck**: Purpose-grouped toolbars for Transformations (`Dense → Sparse`, `Sparse → Dense`, `Transpose`), Binary Arithmetic (`A + B`, `A × B`), and Analysis (`Analyze Memory`).
5. **Result Inspector**: Switchable tabs for `Dense Matrix`, `Triplet Stream`, `Statistics`, and `Memory Metrics`.
6. **Telemetry & Activity Drawer**: Live status pill (`IDLE`, `PROCESSING`, `SUCCESS`, `ERROR`), real-time diagnostic stream, and system timestamps.

---

## 6. Production Docker & Render Deployment

ZEROFORM is fully containerized and production-ready for deployment on **Render** (or any Docker/Kubernetes container cloud) directly from a GitHub repository.

### 6.1 Container Architecture
- **Base Image**: Official `python:3.12-slim-bookworm` (lightweight Debian Linux).
- **Native C Compilation**: The C11 source files (`zeroform.c`, `dense_matrix.c`, `sparse_matrix.c`, `matrix_operations.c`, `memory_analysis.c`, `src/main.c`) are compiled inside the Linux container during image build using `gcc` with strict flags (`-std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 -lm`). The Windows `.exe` is never copied or executed.
- **WSGI Production Server**: Gunicorn multi-worker multi-threaded server (`2 workers, 4 threads, 30s timeout`), replacing the Flask development server.
- **Least-Privilege Security**: Non-root system user `appuser` (UID 10001) executes the process.
- **Port Flexibility**: Binds to `0.0.0.0:${PORT:-10000}` automatically conforming to Render's dynamic `$PORT` assignment.

### 6.2 Local Docker Build & Verification (Optional)
If Docker Desktop is running locally on your workstation:
```bash
# 1. Build the production Docker image
docker build -t zeroform-studio .

# 2. Run the container locally (mapped to port 10000)
docker run -d --name zeroform-app -p 10000:10000 -e PORT=10000 zeroform-studio

# 3. Test container health
curl http://localhost:10000/api/health

# 4. Open in browser: http://localhost:10000/
```

### 6.3 Step-by-Step Render Public Deployment
Deploying to Render takes under 3 minutes via GitHub:

1. **Push to GitHub**:
   Ensure all tracked files (including `Dockerfile`, `.dockerignore`, `c_engine/`, `backend/`, `frontend/`, `protocol/`) are committed and pushed to your GitHub repository.
2. **Log into Render**:
   Open [dashboard.render.com](https://dashboard.render.com/) and click **New +** → **Web Service**.
3. **Connect Repository**:
   Select your GitHub repository (`DSA_MINI_PROJECT` or `zeroform`).
4. **Configure Service Settings**:
   - **Name**: `zeroform-studio` (or your choice).
   - **Region**: Choose the region closest to you (e.g., *Singapore*, *Oregon*, *Frankfurt*).
   - **Language / Runtime**: Select **Docker** (Render will detect the root `Dockerfile` automatically).
   - **Instance Type**: **Free** or **Starter**.
5. **Configure Health Check & Networking**:
   - Expand **Advanced Settings**.
   - Set **Health Check Path**: `/api/health`.
   - Render automatically sets `$PORT` (typically `10000`), which Gunicorn automatically binds to.
6. **Deploy**:
   Click **Create Web Service**. Render pulls the repository, builds the Docker image, compiles `zeroform` for Linux via GCC, and launches Gunicorn.
7. **Verify Public URL**:
   Once the build completes and the health check responds `200 OK`, Render provides your public HTTPS URL (e.g., `https://zeroform-studio.onrender.com`).

### 6.4 Platform-Aware Executable Resolution
`backend/engine.py` dynamically resolves the C computational engine:
- **Windows Development**: Defaults to `c_engine/zeroform.exe`.
- **Linux / Docker Production**: Defaults to `c_engine/zeroform`.
- **Environment Override**: Setting `ZEROFORM_EXECUTABLE` (or `ZEROFORM_EXE`) forces an absolute path override if configured in cloud environment variables.
- **Path Isolation**: Paths are resolved relative to the project root, immune to working directory shifts, and user HTTP input can never influence the executable path.

### 6.5 Health Check Endpoint
Render monitors the container using:
```http
GET /api/health
```
- **Live C Engine Verification**: This endpoint does not merely test if the file exists on disk; it executes `CMD PING v1` through the C subprocess.
- **HTTP 200 OK**: Engine is connected, executable is verified, version is returned.
- **HTTP 503 Service Unavailable**: Engine crashed, binary missing, or protocol failed.

### 6.6 Troubleshooting Guide
| Issue | Common Cause | Resolution |
|---|---|---|
| **Render Health Check Timeout** | Health check path was left blank or misconfigured | Set **Health Check Path** to `/api/health` in Render Dashboard settings. |
| **Port Binding Error** | Hardcoded port instead of `$PORT` | Dockerfile uses `0.0.0.0:${PORT:-10000}` which respects Render's dynamic port injection. |
| **C Engine Not Found (503)** | File permissions or compilation failure | Dockerfile executes `chmod 755 c_engine/zeroform` during image build. Ensure build logs show `gcc ... -o zeroform` succeeded. |
| **Subprocess Timeout (504)** | Massive input exceeding memory bounds | Input limits enforce 10M cells and 1M NNZ; C subprocess timeout is 5.0s. Keep operations within limits. |
| **Out of Memory on Free Tier** | Excessive Gunicorn worker processes | Container uses `--workers 2 --threads 4` which remains well within Render's 512MB RAM free tier limit. |


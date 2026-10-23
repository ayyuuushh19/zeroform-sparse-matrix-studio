"""
test_api.py — Flask REST API and end-to-end integration test suite.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.
"""

import pytest
import json
from unittest.mock import patch
from backend.app import create_app
from backend.validator import (
    validate_number,
    ValidationError,
    validate_coordinate,
    validate_matrix_dict,
    validate_entry_payload
)
from backend.engine import (
    execute_engine,
    EngineNotFoundError,
    EngineTimeoutError,
    EngineCrashError
)
from backend.protocol import parse_response
from tests.oracle import (
    oracle_dense_add,
    oracle_dense_multiply,
    oracle_dense_transpose,
    oracle_dense_to_triplets,
    oracle_memory_analysis
)


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestApiHealthAndSamples:
    """Test health check and sample matrices endpoints."""

    def test_health_with_working_engine(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "healthy"
        assert data["engine"] == "connected"
        assert data["c_engine_version"] == "1.0.0"

    def test_health_engine_unavailable(self, client):
        with patch("backend.app.ping_engine", side_effect=EngineNotFoundError("Executable missing")):
            res = client.get("/api/health")
            assert res.status_code == 503
            data = res.get_json()
            assert data["status"] == "unhealthy"
            assert data["engine"] == "unavailable"

    def test_samples_endpoint(self, client):
        res = client.get("/api/samples")
        assert res.status_code == 200
        data = res.get_json()
        assert "sparse_sample" in data
        assert "addition_pair" in data
        assert "multiplication_pair" in data

    def test_homepage_renders_index_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"ZEROFORM" in res.data
        assert b"Sparse Matrix Studio" in res.data
        assert b"matrix-workspace" in res.data

    def test_static_css_and_js_served(self, client):
        res_css = client.get("/static/css/style.css")
        assert res_css.status_code == 200
        assert b"--surface-ground" in res_css.data

        res_js = client.get("/static/js/app.js")
        assert res_js.status_code == 200
        assert b"checkEngineHealth" in res_js.data


class TestApiAnalyze:
    """Test POST /api/analyze endpoint."""

    def test_analyze_all_zero_matrix(self, client):
        # 3x4 all-zero matrix
        grid = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
        res = client.post("/api/analyze", json={"matrix": grid})
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 3
        assert data["cols"] == 4
        assert data["total_cells"] == 12
        assert data["nnz"] == 0
        assert data["zero_count"] == 12
        assert data["density_percent"] == 0.0
        assert data["sparsity_percent"] == 100.0
        assert data["dense_payload_bytes"] == 96  # 12 * 8
        assert data["sparse_payload_bytes"] == 0
        assert data["sparse_is_smaller"] is True
        assert data["triplets"] == []

        # Check oracle match
        oracle = oracle_memory_analysis(3, 4, 0)
        assert data["dense_payload_bytes"] == oracle["dense_payload_bytes"]

    def test_analyze_sparse_example(self, client):
        grid = [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0]
        ]
        res = client.post("/api/analyze", json={"matrix": grid})
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 2
        assert data["cols"] == 3
        assert data["nnz"] == 3
        assert data["zero_count"] == 3
        assert len(data["triplets"]) == 3
        assert data["dense_payload_bytes"] == 48  # 6 * 8
        assert data["sparse_payload_bytes"] == 48  # 3 * 16
        assert data["sparse_is_smaller"] is False  # 48 is not strictly smaller than 48
        assert data["savings_bytes"] == 0
        assert data["percent_saved"] == 0.0

    def test_analyze_sparse_larger_footprint(self, client):
        # 2x2 matrix with 3 nonzeros: dense=32B, sparse=48B (sparse is larger)
        grid = [
            [1.0, 2.0],
            [3.0, 0.0]
        ]
        res = client.post("/api/analyze", json={"matrix": grid})
        assert res.status_code == 200
        data = res.get_json()
        assert data["dense_payload_bytes"] == 32  # 4 * 8
        assert data["sparse_payload_bytes"] == 48  # 3 * 16
        assert data["sparse_is_smaller"] is False
        assert data["savings_bytes"] == 16  # 48 - 32 penalty
        assert data["percent_saved"] == 0.0


class TestApiOperations:
    """Test POST /api/operation for all supported verbs."""

    def test_to_sparse_conversion(self, client):
        grid = [
            [0.0, 5.0, 0.0],
            [2.0, 0.0, 0.0]
        ]
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": grid
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["operation"] == "to_sparse"
        assert data["rows"] == 2
        assert data["cols"] == 3
        assert data["nnz"] == 2
        assert data["triplets"] == [
            {"row": 0, "col": 1, "value": 5.0},
            {"row": 1, "col": 0, "value": 2.0}
        ]

    def test_to_dense_conversion(self, client):
        sparse_mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 1, "value": 8.0}]
        }
        res = client.post("/api/operation", json={
            "operation": "to_dense",
            "matrix": sparse_mat
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 2
        assert data["cols"] == 2
        assert data["matrix"] == [[0.0, 8.0], [0.0, 0.0]]

    def test_addition_with_cancellation(self, client):
        mat_a = [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0]
        ]
        mat_b = [
            [-1.0, 4.0, 0.0],
            [5.0, -3.0, 7.0]
        ]
        res = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 2
        assert data["cols"] == 3
        # In (0,0): 1 + (-1) = 0 (cancelled)
        # In (1,1): 3 + (-3) = 0 (cancelled)
        # Remaining: (0,1)=4, (0,2)=2, (1,0)=5, (1,2)=7 -> NNZ = 4
        assert data["nnz"] == 4
        expected_triplets = [
            {"row": 0, "col": 1, "value": 4.0},
            {"row": 0, "col": 2, "value": 2.0},
            {"row": 1, "col": 0, "value": 5.0},
            {"row": 1, "col": 2, "value": 7.0}
        ]
        assert data["triplets"] == expected_triplets

        # Check against test oracle
        oracle_res = oracle_dense_add(mat_a, mat_b)
        assert oracle_dense_to_triplets(oracle_res) == expected_triplets

    def test_rectangular_multiplication(self, client):
        # 2x3 times 3x2 -> 2x2
        mat_a = [
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0]
        ]
        mat_b = [
            [5.0, 0.0],
            [0.0, 6.0],
            [7.0, 8.0]
        ]
        res = client.post("/api/operation", json={
            "operation": "multiply",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 2
        assert data["cols"] == 2

        # Oracle expected:
        # [1*5 + 2*0 + 0*7, 1*0 + 2*6 + 0*8] = [5, 12]
        # [0*5 + 3*0 + 4*7, 0*0 + 3*6 + 4*8] = [28, 50]
        expected_oracle = oracle_dense_multiply(mat_a, mat_b)
        expected_triplets = oracle_dense_to_triplets(expected_oracle)
        assert data["triplets"] == expected_triplets

    def test_transpose_rectangular(self, client):
        mat_a = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        res = client.post("/api/operation", json={
            "operation": "transpose",
            "matrix_a": mat_a
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 3
        assert data["cols"] == 2
        expected_oracle = oracle_dense_transpose(mat_a)
        assert data["triplets"] == oracle_dense_to_triplets(expected_oracle)


class TestApiEntryMutations:
    """Test POST /api/entry for mutation and search actions."""

    def test_entry_insert(self, client):
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }
        res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": mat,
            "row": 1,
            "col": 2,
            "value": 4.5
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["nnz"] == 2
        assert data["triplets"] == [
            {"row": 0, "col": 0, "value": 1.0},
            {"row": 1, "col": 2, "value": 4.5}
        ]

    def test_entry_insert_duplicate_fails(self, client):
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }
        res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": mat,
            "row": 0,
            "col": 0,
            "value": 2.0
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_DUPLICATE"

    def test_entry_set_operations(self, client):
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }
        # Update existing entry
        res_update = client.post("/api/entry", json={
            "action": "set",
            "matrix": mat,
            "row": 0,
            "col": 0,
            "value": 5.5
        })
        assert res_update.status_code == 200
        assert res_update.get_json()["triplets"] == [{"row": 0, "col": 0, "value": 5.5}]

        # Set new entry
        res_insert_via_set = client.post("/api/entry", json={
            "action": "set",
            "matrix": mat,
            "row": 1,
            "col": 1,
            "value": 3.0
        })
        assert res_insert_via_set.status_code == 200
        assert res_insert_via_set.get_json()["nnz"] == 2

        # Set to 0.0 deletes entry
        res_zero = client.post("/api/entry", json={
            "action": "set",
            "matrix": mat,
            "row": 0,
            "col": 0,
            "value": 0.0
        })
        assert res_zero.status_code == 200
        assert res_zero.get_json()["nnz"] == 0

    def test_entry_search_found_and_not_found(self, client):
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 2, "value": 9.9}]
        }
        # Search found
        res_found = client.post("/api/entry", json={
            "action": "search",
            "matrix": mat,
            "row": 1,
            "col": 2
        })
        assert res_found.status_code == 200
        d_found = res_found.get_json()
        assert d_found["found"] is True
        assert d_found["value"] == 9.9
        assert d_found["row"] == 1
        assert d_found["col"] == 2

        # Search not found
        res_not_found = client.post("/api/entry", json={
            "action": "search",
            "matrix": mat,
            "row": 0,
            "col": 0
        })
        assert res_not_found.status_code == 200
        d_not_found = res_not_found.get_json()
        assert d_not_found["found"] is False
        assert d_not_found["value"] == 0.0

    def test_entry_delete_success_and_absent_404(self, client):
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 0, "value": 2.0}]
        }
        # Delete existing
        res_del = client.post("/api/entry", json={
            "action": "delete",
            "matrix": mat,
            "row": 0,
            "col": 0
        })
        assert res_del.status_code == 200
        assert res_del.get_json()["nnz"] == 0

        # Delete absent -> 404 Not Found
        res_absent = client.post("/api/entry", json={
            "action": "delete",
            "matrix": mat,
            "row": 1,
            "col": 1
        })
        assert res_absent.status_code == 404
        err = res_absent.get_json()["error"]
        assert err["code"] == "ZF_ERR_NOT_FOUND"

    def test_entry_sort_and_traverse(self, client):
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [
                {"row": 1, "col": 1, "value": 4.0},
                {"row": 0, "col": 0, "value": 1.0}
            ]
        }
        res_sort = client.post("/api/entry", json={
            "action": "sort",
            "matrix": mat
        })
        assert res_sort.status_code == 200
        # Sorted row-major: (0,0) before (1,1)
        assert res_sort.get_json()["triplets"] == [
            {"row": 0, "col": 0, "value": 1.0},
            {"row": 1, "col": 1, "value": 4.0}
        ]

        res_trav = client.post("/api/entry", json={
            "action": "traverse",
            "matrix": mat
        })
        assert res_trav.status_code == 200
        assert res_trav.get_json()["nnz"] == 2


class TestApiValidationErrors:
    """Test validation errors, HTTP status codes, and security."""

    def test_missing_body_400(self, client):
        res = client.post("/api/analyze", data="")
        assert res.status_code == 400

    def test_ragged_matrix_400(self, client):
        ragged = [
            [1.0, 2.0],
            [3.0]  # Missing second column
        ]
        res = client.post("/api/analyze", json={"matrix": ragged})
        assert res.status_code == 400
        data = res.get_json()
        assert "Ragged matrix" in data["error"]["message"]

    def test_dimension_mismatch_addition_422(self, client):
        mat_a = [[1.0, 2.0]]  # 1x2
        mat_b = [[1.0], [2.0]]  # 2x1
        res = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_DIMENSION"

    def test_dimension_mismatch_multiplication_422(self, client):
        mat_a = [[1.0, 2.0]]  # 1x2 (cols=2)
        mat_b = [[1.0, 2.0]]  # 1x2 (rows=1 != 2)
        res = client.post("/api/operation", json={
            "operation": "multiply",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_DIMENSION"

    def test_duplicate_sparse_coordinate_422(self, client):
        sparse = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 1, "col": 1, "value": 2.0},
                {"row": 1, "col": 1, "value": 5.0}  # duplicate
            ]
        }
        res = client.post("/api/entry", json={
            "action": "traverse",
            "matrix": sparse
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_DUPLICATE"

    def test_coordinate_out_of_bounds_422(self, client):
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": []
        }
        res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": mat,
            "row": 5,  # out of bounds
            "col": 0,
            "value": 1.0
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_unsupported_operation_400(self, client):
        res = client.post("/api/operation", json={
            "operation": "invert",
            "matrix": [[1.0]]
        })
        assert res.status_code == 400
        assert "Unsupported operation" in res.get_json()["error"]["message"]

    def test_oversized_dimension_413(self, client):
        huge = {
            "format": "sparse",
            "rows": 100_001,  # exceeds MAX_DIM
            "cols": 5,
            "entries": []
        }
        res = client.post("/api/entry", json={
            "action": "traverse",
            "matrix": huge
        })
        assert res.status_code == 413
        assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"

    def test_subprocess_timeout_504(self, client):
        with patch("backend.app.execute_engine", side_effect=EngineTimeoutError("Timed out")):
            res = client.post("/api/analyze", json={"matrix": [[1.0]]})
            assert res.status_code == 504
            assert res.get_json()["error"]["code"] == "ENGINE_TIMEOUT"

    def test_subprocess_crash_500(self, client):
        with patch("backend.app.execute_engine", side_effect=EngineCrashError(-11, "SIGSEGV")):
            res = client.post("/api/analyze", json={"matrix": [[1.0]]})
            assert res.status_code == 500
            assert res.get_json()["error"]["code"] == "ENGINE_CRASH"

    def test_zero_dimensions_rejected_400(self, client):
        mat = {
            "format": "dense",
            "rows": 0,
            "cols": 5,
            "data": []
        }
        res = client.post("/api/analyze", json={"matrix": mat})
        assert res.status_code == 400
        assert res.get_json()["error"]["code"] == "ZF_ERR_PARAM"

    def test_non_finite_values_rejected_422(self, client):
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 0, "value": float("nan")}]
        }
        # In JSON, Python's json encoder converts nan if allow_nan=True, which Flask parses as nan
        res = client.post("/api/entry", json={"action": "traverse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_multiplication_accumulator_overflow_rejected_422(self, client):
        import sys
        # Finite operands whose individual products are finite, but accumulated sum overflows to +inf
        val = 0.75 * sys.float_info.max
        mat_a = [[val, val]]
        mat_b = [[1.0], [1.0]]
        res = client.post("/api/operation", json={
            "operation": "multiply",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_addition_intermediate_overflow_rejected_422(self, client):
        import sys
        val = sys.float_info.max
        mat_a = [[val]]
        mat_b = [[val]]
        res = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"


class TestSparseDimensionCeilingRegression:
    """
    P1 Regression Test Suite: Universal 10,000,000-cell ceiling enforcement.
    Ensures sparse matrices cannot bypass the ceiling simply because NNZ is small.
    Covers Tests 1 through 10 as specified in the safety audit.
    """

    def test_01_huge_sparse_dimensions_rejected(self, client):
        """Test 1: rows=100000, cols=100000, nnz=1 rejected before C engine or grid construction."""
        huge_sparse = {
            "format": "sparse",
            "rows": 100_000,
            "cols": 100_000,
            "nnz": 1,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }

        with patch("backend.app.execute_engine") as mock_engine:
            # 1. /api/analyze
            res = client.post("/api/analyze", json={"matrix": huge_sparse})
            assert res.status_code == 413
            err = res.get_json()["error"]
            assert err["code"] == "ZF_ERR_OVERFLOW"
            assert "Total cell count" in err["message"]
            assert "10000000" in err["message"]
            mock_engine.assert_not_called()

            # 2. /api/operation to_dense
            res = client.post("/api/operation", json={
                "operation": "to_dense",
                "matrix_a": huge_sparse
            })
            assert res.status_code == 413
            assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"
            mock_engine.assert_not_called()

            # 3. /api/operation to_sparse
            res = client.post("/api/operation", json={
                "operation": "to_sparse",
                "matrix_a": huge_sparse
            })
            assert res.status_code == 413
            assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"
            mock_engine.assert_not_called()

            # 4. /api/entry
            res = client.post("/api/entry", json={
                "action": "search",
                "matrix": huge_sparse,
                "row": 0,
                "col": 0
            })
            assert res.status_code == 413
            assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"
            mock_engine.assert_not_called()

    def test_02_just_above_limit_rejected(self, client):
        """Test 2: Dimensions exceeding 10,000,000 cells rejected."""
        from backend.validator import validate_cell_limit, ValidationError
        # Direct check on exact product 10,000,001
        with pytest.raises(ValidationError) as exc_info:
            validate_cell_limit(1, 10_000_001)
        assert exc_info.value.status_code == 413
        assert exc_info.value.code == "ZF_ERR_OVERFLOW"

        # Via API: 1000 x 10001 = 10,001,000 cells > 10,000,000
        mat = {
            "format": "sparse",
            "rows": 1000,
            "cols": 10001,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }
        res = client.post("/api/analyze", json={"matrix": mat})
        assert res.status_code == 413
        assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"

    def test_03_exactly_at_limit_accepted(self, client):
        """Test 3: Product exactly 10,000,000 cells accepted."""
        mat = {
            "format": "sparse",
            "rows": 1000,
            "cols": 10000,
            "entries": []
        }
        res = client.post("/api/analyze", json={"matrix": mat})
        assert res.status_code == 200
        data = res.get_json()
        assert data["total_cells"] == 10_000_000
        assert data["rows"] == 1000
        assert data["cols"] == 10000

    def test_04_sparse_matrix_at_limit_accepted(self, client):
        """Test 4: Sparse matrix with rows=1000, cols=10000, NNZ=1 accepted."""
        mat = {
            "format": "sparse",
            "rows": 1000,
            "cols": 10000,
            "nnz": 1,
            "entries": [{"row": 999, "col": 9999, "value": 42.0}]
        }
        # Analyze works
        res = client.post("/api/analyze", json={"matrix": mat})
        assert res.status_code == 200
        assert res.get_json()["total_cells"] == 10_000_000
        assert res.get_json()["nnz"] == 1

        # Search works
        res = client.post("/api/entry", json={
            "action": "search",
            "matrix": mat,
            "row": 999,
            "col": 9999
        })
        assert res.status_code == 200
        assert res.get_json()["found"] is True
        assert res.get_json()["value"] == 42.0

        # to_sparse works
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix_a": mat
        })
        assert res.status_code == 200
        assert res.get_json()["nnz"] == 1

    def test_05_zero_nnz_accepted(self, client):
        """Test 5: Empty sparse matrix with valid dimensions within ceiling accepted."""
        mat = {
            "format": "sparse",
            "rows": 500,
            "cols": 500,
            "nnz": 0,
            "entries": []
        }
        res = client.post("/api/analyze", json={"matrix": mat})
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 500
        assert data["cols"] == 500
        assert data["total_cells"] == 250_000
        assert data["nnz"] == 0
        assert data["zero_count"] == 250_000

    def test_06_large_integer_dimensions_no_overflow(self, client):
        """Test 6: Extremely large dimensions rejected without overflow or crash."""
        from backend.validator import validate_cell_limit, ValidationError

        # Direct overflow-safe check on huge integer sizes
        with pytest.raises(ValidationError) as exc:
            validate_cell_limit(10**18, 10**18)
        assert exc.value.status_code == 413
        assert exc.value.code == "ZF_ERR_OVERFLOW"

        with pytest.raises(ValidationError) as exc:
            validate_cell_limit(2**64 + 1, 2**64 + 1)
        assert exc.value.status_code == 413
        assert exc.value.code == "ZF_ERR_OVERFLOW"

        # Via API: oversized dimension is safely rejected with 413
        mat = {
            "format": "sparse",
            "rows": 2**64,
            "cols": 2**64,
            "entries": []
        }
        res = client.post("/api/analyze", json={"matrix": mat})
        assert res.status_code == 413
        assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"

    def test_07_multiplication_result_exceeds_limit_rejected(self, client):
        """Test 7: Compatible inputs whose product dimensions exceed 10,000,000 rejected before allocation."""
        mat_a = {
            "format": "sparse",
            "rows": 10000,
            "cols": 1,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }
        mat_b = {
            "format": "sparse",
            "rows": 1,
            "cols": 10000,
            "entries": [{"row": 0, "col": 0, "value": 2.0}]
        }

        with patch("backend.app.execute_engine") as mock_engine:
            res = client.post("/api/operation", json={
                "operation": "multiply",
                "matrix_a": mat_a,
                "matrix_b": mat_b
            })
            assert res.status_code == 413
            err = res.get_json()["error"]
            assert err["code"] == "ZF_ERR_OVERFLOW"
            assert "Total cell count" in err["message"]
            mock_engine.assert_not_called()

    def test_08_multiplication_result_at_limit_accepted(self, client):
        """Test 8: Multiplication result containing exactly 10,000,000 cells accepted."""
        mat_a = {
            "format": "sparse",
            "rows": 1000,
            "cols": 1,
            "entries": [{"row": 0, "col": 0, "value": 3.0}]
        }
        mat_b = {
            "format": "sparse",
            "rows": 1,
            "cols": 10000,
            "entries": [{"row": 0, "col": 0, "value": 4.0}]
        }
        res = client.post("/api/operation", json={
            "operation": "multiply",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 1000
        assert data["cols"] == 10000
        assert data["nnz"] == 1
        assert data["triplets"] == [{"row": 0, "col": 0, "value": 12.0}]

    def test_09_dense_regression_remains_intact(self, client):
        """Test 9: Verify dense dimension ceiling behavior still functions identically."""
        # Oversized dense 2D list rejected
        oversized_dense = {
            "format": "dense",
            "rows": 1000,
            "cols": 10001,
            "data": [[0.0] * 10001 for _ in range(1)]  # even partial will fail dimension validation
        }
        res = client.post("/api/analyze", json={"matrix": oversized_dense})
        assert res.status_code == 413
        assert res.get_json()["error"]["code"] == "ZF_ERR_OVERFLOW"

        # Valid dense matrix works
        valid_dense = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        res = client.post("/api/analyze", json={"matrix": valid_dense})
        assert res.status_code == 200
        assert res.get_json()["total_cells"] == 4
        assert res.get_json()["nnz"] == 4

    def test_10_normal_sparse_regression(self, client):
        """Test 10: Ordinary sparse matrices continue supporting analyze, conversion, add, mul, transpose."""
        mat_a = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 1, "value": 2.0},
                {"row": 1, "col": 2, "value": 3.0}
            ]
        }
        mat_b = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 1, "value": 5.0},
                {"row": 2, "col": 0, "value": 4.0}
            ]
        }

        # 1. Analyze
        res = client.post("/api/analyze", json={"matrix": mat_a})
        assert res.status_code == 200
        assert res.get_json()["total_cells"] == 9
        assert res.get_json()["nnz"] == 2

        # 2. to_sparse
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix_a": mat_a})
        assert res.status_code == 200
        assert res.get_json()["nnz"] == 2

        # 3. to_dense
        res = client.post("/api/operation", json={"operation": "to_dense", "matrix_a": mat_a})
        assert res.status_code == 200
        assert res.get_json()["matrix"] == [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0]
        ]

        # 4. add
        res = client.post("/api/operation", json={"operation": "add", "matrix_a": mat_a, "matrix_b": mat_b})
        assert res.status_code == 200
        assert res.get_json()["nnz"] == 3

        # 5. multiply
        res = client.post("/api/operation", json={"operation": "multiply", "matrix_a": mat_a, "matrix_b": mat_b})
        assert res.status_code == 200

        # 6. transpose
        res = client.post("/api/operation", json={"operation": "transpose", "matrix_a": mat_a})
        assert res.status_code == 200
        assert res.get_json()["triplets"] == [
            {"row": 1, "col": 0, "value": 2.0},
            {"row": 2, "col": 1, "value": 3.0}
        ]


class TestNumericValidationRegression:
    """
    P2 Regression Test Suite: Very large JSON integers and safe numeric validation.
    Covers NUM-01 through NUM-15:
    - NUM-01: 400-digit positive integer -> controlled rejection (422), no 500
    - NUM-02: 400-digit negative integer -> controlled rejection (422), no 500
    - NUM-03: Very large integer beyond finite float range -> controlled rejection (422)
    - NUM-04: Normal integer (42) -> accepted
    - NUM-05: Negative integer (-42) -> accepted
    - NUM-06: Finite decimal (3.125) -> accepted
    - NUM-07: Zero -> accepted
    - NUM-08: NaN -> rejected (422)
    - NUM-09: Positive infinity -> rejected (422)
    - NUM-10: Negative infinity -> rejected (422)
    - NUM-11: Oversized value through /api/operation -> controlled rejection; engine not invoked
    - NUM-12: Oversized value through /api/analyze -> controlled rejection; engine not invoked
    - NUM-13: Oversized value in sparse triplet -> controlled rejection (422)
    - NUM-14: Normal matrix add, mul, transpose remain functional
    - NUM-15: Repeated invalid numeric requests -> no process crash or persistent failure
    """

    def test_num_01_large_positive_integer_400_digits_rejected(self, client):
        """NUM-01: A 400-digit positive integer as a matrix value is rejected with controlled 422, not 500."""
        huge_int = int("9" * 400)
        # 1. Via Python dictionary / Flask JSON serializer
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[huge_int]]
        })
        assert res.status_code == 422
        data = res.get_json()
        assert data["error"]["code"] == "ZF_ERR_VALUE"
        assert "must be finite" in data["error"]["message"]

        # 2. Via raw JSON string body to verify the wire protocol parsing path
        raw_body = json.dumps({
            "operation": "to_sparse",
            "matrix": [[huge_int]]
        })
        res_raw = client.post("/api/operation", data=raw_body, content_type="application/json")
        assert res_raw.status_code == 422
        assert res_raw.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_02_large_negative_integer_400_digits_rejected(self, client):
        """NUM-02: A 400-digit negative integer is rejected with controlled 422, not 500."""
        huge_neg_int = -int("9" * 400)
        # 1. Via Python dictionary
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[huge_neg_int]]
        })
        assert res.status_code == 422
        data = res.get_json()
        assert data["error"]["code"] == "ZF_ERR_VALUE"
        assert "must be finite" in data["error"]["message"]

        # 2. Via raw JSON string body
        raw_body = json.dumps({
            "operation": "to_sparse",
            "matrix": [[huge_neg_int]]
        })
        res_raw = client.post("/api/operation", data=raw_body, content_type="application/json")
        assert res_raw.status_code == 422
        assert res_raw.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_03_integer_beyond_finite_float_range_rejected(self, client):
        """NUM-03: An integer beyond IEEE-754 finite float range (e.g. 10**309, 2**1024) is rejected."""
        val = 10**309
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[val]]
        })
        assert res.status_code == 422
        data = res.get_json()
        assert data["error"]["code"] == "ZF_ERR_VALUE"

        # Also verify 2**1024
        res2 = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[2**1024]]
        })
        assert res2.status_code == 422
        assert res2.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_04_normal_integer_accepted(self, client):
        """NUM-04: A normal integer such as 42 is accepted."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[42]]
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["triplets"] == [{"row": 0, "col": 0, "value": 42.0}]

    def test_num_05_negative_integer_accepted(self, client):
        """NUM-05: A negative integer such as -42 is accepted."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[-42]]
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["triplets"] == [{"row": 0, "col": 0, "value": -42.0}]

    def test_num_06_finite_decimal_accepted(self, client):
        """NUM-06: A finite decimal such as 3.125 is accepted."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[3.125]]
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["triplets"] == [{"row": 0, "col": 0, "value": 3.125}]

    def test_num_07_zero_accepted(self, client):
        """NUM-07: Zero is accepted."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[0, 5], [0.0, 0]]
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["nnz"] == 1
        assert data["triplets"] == [{"row": 0, "col": 1, "value": 5.0}]

    def test_num_08_nan_rejected(self, client):
        """NUM-08: NaN is rejected."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[float("nan")]]
        })
        assert res.status_code == 422
        data = res.get_json()
        assert data["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_09_positive_infinity_rejected(self, client):
        """NUM-09: Positive infinity is rejected."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[float("inf")]]
        })
        assert res.status_code == 422
        data = res.get_json()
        assert data["error"]["code"] == "ZF_ERR_VALUE"

        # Also via raw JSON representation
        raw_body = '{"operation": "to_sparse", "matrix": [[1e500]]}'
        res_raw = client.post("/api/operation", data=raw_body, content_type="application/json")
        assert res_raw.status_code == 422
        assert res_raw.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_10_negative_infinity_rejected(self, client):
        """NUM-10: Negative infinity is rejected."""
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[float("-inf")]]
        })
        assert res.status_code == 422
        data = res.get_json()
        assert data["error"]["code"] == "ZF_ERR_VALUE"

        # Also via raw JSON representation
        raw_body = '{"operation": "to_sparse", "matrix": [[-1e500]]}'
        res_raw = client.post("/api/operation", data=raw_body, content_type="application/json")
        assert res_raw.status_code == 422
        assert res_raw.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_11_oversized_value_operation_engine_not_invoked(self, client):
        """NUM-11: Oversized value through /api/operation produces controlled error; engine not invoked."""
        with patch("backend.app.execute_engine") as mock_engine:
            res = client.post("/api/operation", json={
                "operation": "to_sparse",
                "matrix": [[int("9" * 400)]]
            })
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"
            mock_engine.assert_not_called()

    def test_num_12_oversized_value_analyze_engine_not_invoked(self, client):
        """NUM-12: Oversized value through /api/analyze produces controlled error; engine not invoked."""
        with patch("backend.app.execute_engine") as mock_engine:
            res = client.post("/api/analyze", json={
                "matrix": [[int("9" * 400)]]
            })
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"
            mock_engine.assert_not_called()

    def test_num_13_oversized_value_in_sparse_triplet_rejected(self, client):
        """NUM-13: Oversized value in a sparse triplet is rejected."""
        sparse_mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 0, "value": int("9" * 400)}]
        }
        res = client.post("/api/operation", json={
            "operation": "to_dense",
            "matrix": sparse_mat
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"

    def test_num_14_normal_matrix_operations_remain_functional(self, client):
        """NUM-14: Normal matrix addition, multiplication, and transpose continue to work."""
        mat_a = [[1.0, 2.0], [3.0, 4.0]]
        mat_b = [[5.0, 6.0], [7.0, 8.0]]

        # Addition
        res_add = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res_add.status_code == 200
        assert res_add.get_json()["nnz"] == 4

        # Multiplication
        res_mul = client.post("/api/operation", json={
            "operation": "multiply",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res_mul.status_code == 200
        assert res_mul.get_json()["nnz"] == 4

        # Transpose
        res_tr = client.post("/api/operation", json={
            "operation": "transpose",
            "matrix_a": mat_a
        })
        assert res_tr.status_code == 200
        assert res_tr.get_json()["rows"] == 2
        assert res_tr.get_json()["cols"] == 2

    def test_num_15_repeated_invalid_numeric_requests_no_app_failure(self, client):
        """NUM-15: Repeated invalid numeric requests cause no crash or persistent failure."""
        huge_int = int("9" * 400)
        # Send a sequence of invalid requests
        for payload in [
            {"operation": "to_sparse", "matrix": [[huge_int]]},
            {"operation": "to_sparse", "matrix": [[-huge_int]]},
            {"operation": "to_sparse", "matrix": [[float("nan")]]},
            {"operation": "to_sparse", "matrix": [[float("inf")]]},
            {"operation": "to_sparse", "matrix": [[10**350]]},
        ]:
            res = client.post("/api/operation", json=payload)
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_VALUE"

        # Application remains completely healthy for valid subsequent requests
        res_valid = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix": [[42.0]]
        })
        assert res_valid.status_code == 200
        assert res_valid.get_json()["triplets"] == [{"row": 0, "col": 0, "value": 42.0}]

        # Health endpoint reports connected
        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        assert res_health.get_json()["status"] == "healthy"

    def test_num_16_direct_validate_number_unit_suite(self):
        """NUM-16: Direct unit verification of validate_number contract."""
        # Accepted values
        assert validate_number(42, "num") == 42.0
        assert validate_number(-42, "num") == -42.0
        assert validate_number(3.125, "num") == 3.125
        assert validate_number(0, "num") == 0.0
        assert validate_number(0.0, "num") == 0.0
        assert validate_number(1e-5, "num") == 1e-5
        assert validate_number(2.5e10, "num") == 2.5e10

        # Out of range / non-finite values -> 422 ZF_ERR_VALUE
        for val in [int("9" * 400), -int("9" * 400), 10**309, 2**1024, 10**5000, float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(ValidationError) as exc_info:
                validate_number(val, "field")
            assert exc_info.value.status_code == 422
            assert exc_info.value.code == "ZF_ERR_VALUE"

        # Non-numeric types -> 400 VALIDATION_ERROR
        for invalid_type_val in [True, False, "42", "1e-5", None, [1], {"val": 1}]:
            with pytest.raises(ValidationError) as exc_info:
                validate_number(invalid_type_val, "field")
            assert exc_info.value.status_code == 400

    def test_num_17_entry_actions_with_oversized_values(self, client):
        """NUM-17: /api/entry insert and set with 400-digit integers rejected."""
        huge_int = int("9" * 400)
        base_sparse = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": []
        }

        # Insert action
        res_ins = client.post("/api/entry", json={
            "action": "insert",
            "matrix": base_sparse,
            "row": 0,
            "col": 0,
            "value": huge_int
        })
        assert res_ins.status_code == 422
        assert res_ins.get_json()["error"]["code"] == "ZF_ERR_VALUE"

        # Set action
        res_set = client.post("/api/entry", json={
            "action": "set",
            "matrix": base_sparse,
            "row": 0,
            "col": 0,
            "value": huge_int
        })
        assert res_set.status_code == 422
        assert res_set.get_json()["error"]["code"] == "ZF_ERR_VALUE"


class TestBooleanCoordinateValidationRegression:
    """
    P3 Regression Test Suite: Boolean Values Pass Sparse Coordinate Validation.
    Covers BOOL-01 through BOOL-16 (plus direct unit tests and edge cases):
    - BOOL-01: Sparse row = true -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-02: Sparse row = false -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-03: Sparse column = true -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-04: Sparse column = false -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-05: Entry insert row = true -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-06: Entry insert column = true -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-07: Entry set/update row or column = Boolean -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-08: Entry delete row or column = Boolean -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-09: Entry search row or column = Boolean -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-10: Valid integer coordinates -> Accepted (200 OK)
    - BOOL-11: Fractional coordinate (e.g. 1.5) -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-12: String coordinate (e.g. "1") -> Follows existing API contract (422 ZF_ERR_INDEX)
    - BOOL-13: Negative or out-of-range coordinate -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-14: Boolean in Matrix B sparse coordinates -> Rejected (422 ZF_ERR_INDEX)
    - BOOL-15: Invalid Boolean coordinate request -> C engine not invoked
    - BOOL-16: Valid sparse operations after invalid requests -> Continue working normally
    - BOOL-17: Direct validate_coordinate unit suite
    - BOOL-18: Declared nnz with boolean value rejected (400)
    - BOOL-19: Dense cells format with boolean row or col rejected (422 ZF_ERR_INDEX)
    """

    def test_bool_01_sparse_row_true_rejected(self, client):
        """BOOL-01: Sparse row = true is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": True, "col": 1, "value": 2.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

        # Also via raw JSON string body
        raw_body = json.dumps({"operation": "to_sparse", "matrix": mat})
        res_raw = client.post("/api/operation", data=raw_body, content_type="application/json")
        assert res_raw.status_code == 422
        assert res_raw.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_02_sparse_row_false_rejected(self, client):
        """BOOL-02: Sparse row = false is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": False, "col": 1, "value": 2.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

        raw_body = json.dumps({"operation": "to_sparse", "matrix": mat})
        res_raw = client.post("/api/operation", data=raw_body, content_type="application/json")
        assert res_raw.status_code == 422
        assert res_raw.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_03_sparse_col_true_rejected(self, client):
        """BOOL-03: Sparse column = true is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": True, "value": 2.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_04_sparse_col_false_rejected(self, client):
        """BOOL-04: Sparse column = false is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": False, "value": 2.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_05_entry_insert_row_true_rejected(self, client):
        """BOOL-05: Entry insert row = true is rejected."""
        base_mat = {"format": "sparse", "rows": 3, "cols": 3, "entries": []}
        res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": base_mat,
            "row": True,
            "col": 1,
            "value": 5.0
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_06_entry_insert_col_true_rejected(self, client):
        """BOOL-06: Entry insert column = true is rejected."""
        base_mat = {"format": "sparse", "rows": 3, "cols": 3, "entries": []}
        res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": base_mat,
            "row": 1,
            "col": True,
            "value": 5.0
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_07_entry_set_row_or_col_boolean_rejected(self, client):
        """BOOL-07: Entry set/update row or column = Boolean is rejected."""
        base_mat = {"format": "sparse", "rows": 3, "cols": 3, "entries": []}
        # row = True
        res_row = client.post("/api/entry", json={
            "action": "set",
            "matrix": base_mat,
            "row": True,
            "col": 1,
            "value": 5.0
        })
        assert res_row.status_code == 422
        assert res_row.get_json()["error"]["code"] == "ZF_ERR_INDEX"

        # col = False
        res_col = client.post("/api/entry", json={
            "action": "set",
            "matrix": base_mat,
            "row": 1,
            "col": False,
            "value": 5.0
        })
        assert res_col.status_code == 422
        assert res_col.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_08_entry_delete_row_or_col_boolean_rejected(self, client):
        """BOOL-08: Entry delete row or column = Boolean is rejected."""
        base_mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 1, "value": 5.0}]
        }
        res_row = client.post("/api/entry", json={
            "action": "delete",
            "matrix": base_mat,
            "row": True,
            "col": 1
        })
        assert res_row.status_code == 422
        assert res_row.get_json()["error"]["code"] == "ZF_ERR_INDEX"

        res_col = client.post("/api/entry", json={
            "action": "delete",
            "matrix": base_mat,
            "row": 1,
            "col": False
        })
        assert res_col.status_code == 422
        assert res_col.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_09_entry_search_row_or_col_boolean_rejected(self, client):
        """BOOL-09: Entry search row or column = Boolean is rejected."""
        base_mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 1, "value": 5.0}]
        }
        res_row = client.post("/api/entry", json={
            "action": "search",
            "matrix": base_mat,
            "row": True,
            "col": 1
        })
        assert res_row.status_code == 422
        assert res_row.get_json()["error"]["code"] == "ZF_ERR_INDEX"

        res_col = client.post("/api/entry", json={
            "action": "search",
            "matrix": base_mat,
            "row": 1,
            "col": False
        })
        assert res_col.status_code == 422
        assert res_col.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_10_valid_integer_coordinates_accepted(self, client):
        """BOOL-10: Valid integer coordinates are accepted."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 0, "value": 1.0},
                {"row": 2, "col": 2, "value": 9.0}
            ]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 200
        assert res.get_json()["nnz"] == 2

    def test_bool_11_fractional_coordinate_rejected(self, client):
        """BOOL-11: Fractional coordinate such as 1.5 is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1.5, "col": 0, "value": 1.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_12_string_coordinate_rejected(self, client):
        """BOOL-12: String coordinate such as '1' is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": "1", "col": 0, "value": 1.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_13_negative_or_out_of_range_coordinate_rejected(self, client):
        """BOOL-13: Negative or out-of-range coordinate is rejected."""
        # Negative
        mat_neg = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": -1, "col": 0, "value": 1.0}]
        }
        res_neg = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat_neg})
        assert res_neg.status_code == 422
        assert res_neg.get_json()["error"]["code"] == "ZF_ERR_INDEX"

        # Out of bounds
        mat_oob = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 3, "col": 0, "value": 1.0}]
        }
        res_oob = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat_oob})
        assert res_oob.status_code == 422
        assert res_oob.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_14_boolean_in_matrix_b_sparse_coordinates_rejected(self, client):
        """BOOL-14: Boolean in Matrix B sparse coordinates is rejected."""
        mat_a = [[1.0, 0.0], [0.0, 1.0]]
        mat_b = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": True, "col": 0, "value": 4.0}]
        }
        res = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"

    def test_bool_15_invalid_boolean_coordinate_engine_not_invoked(self, client):
        """BOOL-15: Invalid Boolean coordinate request does not invoke C engine."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": True, "col": 0, "value": 1.0}]
        }
        with patch("backend.app.execute_engine") as mock_engine:
            res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"
            mock_engine.assert_not_called()

    def test_bool_16_valid_sparse_operations_after_invalid_requests_succeed(self, client):
        """BOOL-16: Valid sparse operations after invalid requests continue working normally."""
        # 1. Send invalid boolean coordinate requests
        for bad_coord in [True, False]:
            mat_bad = {
                "format": "sparse",
                "rows": 2,
                "cols": 2,
                "entries": [{"row": bad_coord, "col": 0, "value": 1.0}]
            }
            res_bad = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat_bad})
            assert res_bad.status_code == 422

        # 2. Subsequent valid request succeeds
        mat_valid = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 1, "col": 1, "value": 7.5}]
        }
        res_valid = client.post("/api/operation", json={"operation": "to_dense", "matrix": mat_valid})
        assert res_valid.status_code == 200
        assert res_valid.get_json()["matrix"] == [[0.0, 0.0], [0.0, 7.5]]

        # Health endpoint remains healthy
        res_h = client.get("/api/health")
        assert res_h.status_code == 200
        assert res_h.get_json()["status"] == "healthy"

    def test_bool_17_direct_validate_coordinate_unit_suite(self):
        """BOOL-17: Direct unit test for validate_coordinate helper."""
        # Valid cases
        assert validate_coordinate(0, 5, "row") == 0
        assert validate_coordinate(4, 5, "row") == 4

        # Booleans rejected
        for b in [True, False]:
            with pytest.raises(ValidationError) as exc:
                validate_coordinate(b, 5, "row")
            assert exc.value.status_code == 422
            assert exc.value.code == "ZF_ERR_INDEX"
            assert "must be an integer" in exc.value.message

        # Invalid types rejected
        for bad in [1.5, "0", None, [], {}]:
            with pytest.raises(ValidationError) as exc:
                validate_coordinate(bad, 5, "row")
            assert exc.value.status_code == 422
            assert exc.value.code == "ZF_ERR_INDEX"

        # Out of bounds rejected
        for oob in [-1, 5, 10, 10**50]:
            with pytest.raises(ValidationError) as exc:
                validate_coordinate(oob, 5, "row")
            assert exc.value.status_code == 422
            assert exc.value.code == "ZF_ERR_INDEX"
            assert "out of bounds" in exc.value.message

    def test_bool_18_declared_nnz_boolean_rejected(self, client):
        """BOOL-18: Field 'nnz' with boolean value is rejected with 400."""
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "nnz": True,
            "entries": []
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 400

    def test_bool_19_dense_cells_boolean_coordinates_rejected(self, client):
        """BOOL-19: Dense cells format with boolean row or col rejected with 422 ZF_ERR_INDEX."""
        cells = [
            {"row": True, "col": 0, "value": 1.0},
            {"row": 0, "col": 1, "value": 2.0},
            {"row": 1, "col": 0, "value": 3.0},
            {"row": 1, "col": 1, "value": 4.0}
        ]
        mat = {
            "format": "dense",
            "rows": 2,
            "cols": 2,
            "cells": cells
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"


class TestExplicitZeroSparseRegression:
    """Regression test suite for P3: Explicit Zero Entries in Sparse Matrices (ZERO-01 to ZERO-15)."""

    def test_zero_01_sparse_entry_int_zero_rejected(self, client):
        """ZERO-01: Sparse entry value 0 (integer) is rejected with 422 ZF_ERR_PARAM."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 0, "col": 1, "value": 0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        err = res.get_json()["error"]
        assert err["code"] == "ZF_ERR_PARAM"
        assert "explicit zero" in err["message"].lower()

    def test_zero_02_sparse_entry_float_zero_rejected(self, client):
        """ZERO-02: Sparse entry value 0.0 (float) is rejected with 422 ZF_ERR_PARAM."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 2, "value": 0.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        err = res.get_json()["error"]
        assert err["code"] == "ZF_ERR_PARAM"
        assert "explicit zero" in err["message"].lower()

    def test_zero_03_sparse_entry_negative_zero_rejected(self, client):
        """ZERO-03: Sparse entry value -0.0 is rejected with 422 ZF_ERR_PARAM."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 2, "col": 0, "value": -0.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        err = res.get_json()["error"]
        assert err["code"] == "ZF_ERR_PARAM"
        assert "explicit zero" in err["message"].lower()

    def test_zero_04_sparse_entry_positive_nonzero_accepted(self, client):
        """ZERO-04: Sparse entry with a positive nonzero value is accepted."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 0, "col": 1, "value": 4.5}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 200
        data = res.get_json()
        assert data["nnz"] == 1
        assert data["triplets"] == [{"row": 0, "col": 1, "value": 4.5}]

    def test_zero_05_sparse_entry_negative_nonzero_accepted(self, client):
        """ZERO-05: Sparse entry with a negative nonzero value is accepted."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 2, "value": -3.25}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 200
        data = res.get_json()
        assert data["nnz"] == 1
        assert data["triplets"] == [{"row": 1, "col": 2, "value": -3.25}]

    def test_zero_06_sparse_entry_small_nonzero_accepted(self, client):
        """ZERO-06: Sparse entry with a small finite nonzero value is accepted."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 0, "value": 1e-12},
                {"row": 2, "col": 2, "value": -1e-15}
            ]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 200
        data = res.get_json()
        assert data["nnz"] == 2

    def test_zero_07_analyze_explicit_zero_rejected_before_engine(self, client):
        """ZERO-07: Sparse /api/analyze request containing an explicit zero is rejected before engine dispatch."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 0, "value": 1.0},
                {"row": 0, "col": 1, "value": 0.0}
            ]
        }
        with patch("backend.app.execute_engine") as mock_exec:
            res = client.post("/api/analyze", json={"matrix": mat})
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_PARAM"
            mock_exec.assert_not_called()

    def test_zero_08_operation_explicit_zero_rejected_before_engine(self, client):
        """ZERO-08: Sparse /api/operation request containing an explicit zero is rejected before engine dispatch."""
        # Test in matrix_a
        mat_bad_a = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 0, "value": 0}]
        }
        with patch("backend.app.execute_engine") as mock_exec:
            res = client.post("/api/operation", json={"operation": "to_dense", "matrix": mat_bad_a})
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_PARAM"
            mock_exec.assert_not_called()

        # Test in matrix_b for addition
        mat_good_a = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }
        mat_bad_b = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 1, "col": 1, "value": 0.0}]
        }
        with patch("backend.app.execute_engine") as mock_exec:
            res = client.post("/api/operation", json={
                "operation": "add",
                "matrix_a": mat_good_a,
                "matrix_b": mat_bad_b
            })
            assert res.status_code == 422
            assert res.get_json()["error"]["code"] == "ZF_ERR_PARAM"
            mock_exec.assert_not_called()

    def test_zero_09_triplets_representation_explicit_zero_rejected(self, client):
        """ZERO-09: Sparse input using the 'triplets' key representation with an explicit zero is rejected."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "triplets": [{"row": 0, "col": 1, "value": 0.0}]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat})
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_PARAM"

    def test_zero_10_entries_representation_explicit_zero_rejected(self, client):
        """ZERO-10: Sparse input using 'entries' representation (dict or [r, c, v] tuple) rejects zero."""
        # Dict entry
        mat_dict = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 1, "value": 0}]
        }
        res_d = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat_dict})
        assert res_d.status_code == 422
        assert res_d.get_json()["error"]["code"] == "ZF_ERR_PARAM"

        # Tuple entry
        mat_tuple = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [[1, 1, 0.0]]
        }
        res_t = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat_tuple})
        assert res_t.status_code == 422
        assert res_t.get_json()["error"]["code"] == "ZF_ERR_PARAM"

    def test_zero_11_dense_matrix_with_zero_cells_accepted(self, client):
        """ZERO-11: Dense matrix containing zero-valued cells remains completely valid and accepted."""
        mat_dense = {
            "format": "dense",
            "rows": 2,
            "cols": 2,
            "data": [
                [0.0, 3.5],
                [4.0, 0.0]
            ]
        }
        # Operation on dense matrix containing zero cells
        res = client.post("/api/operation", json={"operation": "transpose", "matrix": mat_dense})
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 2
        assert data["cols"] == 2
        assert data["nnz"] == 2
        assert data["triplets"] == [
            {"row": 0, "col": 1, "value": 4.0},
            {"row": 1, "col": 0, "value": 3.5}
        ]

    def test_zero_12_dense_to_sparse_omits_zero_cells(self, client):
        """ZERO-12: Dense-to-sparse conversion with zero cells succeeds; zeros are omitted from sparse output."""
        mat_dense = {
            "format": "dense",
            "rows": 3,
            "cols": 3,
            "data": [
                [0.0, 0.0, 1.5],
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0]
            ]
        }
        res = client.post("/api/operation", json={"operation": "to_sparse", "matrix": mat_dense})
        assert res.status_code == 200
        data = res.get_json()
        # Out of 9 total cells, only 2 are nonzero
        assert data["nnz"] == 2
        assert data["triplets"] == [
            {"row": 0, "col": 2, "value": 1.5},
            {"row": 2, "col": 0, "value": 2.5}
        ]

    def test_zero_13_sparse_matrix_no_explicit_zeros_preserved(self, client):
        """ZERO-13: Sparse matrix with only valid nonzero entries preserves all existing functionality."""
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 0, "value": 2.0},
                {"row": 1, "col": 2, "value": 5.0}
            ]
        }
        # to_dense
        res_d = client.post("/api/operation", json={"operation": "to_dense", "matrix": mat})
        assert res_d.status_code == 200
        assert res_d.get_json()["matrix"] == [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0]
        ]

        # analyze
        res_a = client.post("/api/analyze", json={"matrix": mat})
        assert res_a.status_code == 200
        assert res_a.get_json()["nnz"] == 2
        assert res_a.get_json()["zero_count"] == 7

    def test_zero_14_invalid_explicit_zero_followed_by_valid_request(self, client):
        """ZERO-14: Invalid explicit zero requests followed by a valid request leaves application healthy."""
        # 1. Submit multiple invalid requests with explicit zeros
        for bad_val in [0, 0.0, -0.0]:
            bad_mat = {
                "format": "sparse",
                "rows": 2,
                "cols": 2,
                "entries": [{"row": 0, "col": 0, "value": bad_val}]
            }
            res_bad = client.post("/api/operation", json={"operation": "to_sparse", "matrix": bad_mat})
            assert res_bad.status_code == 422

        # 2. Subsequent valid request succeeds
        valid_mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "entries": [{"row": 0, "col": 1, "value": 8.5}]
        }
        res_ok = client.post("/api/operation", json={"operation": "to_dense", "matrix": valid_mat})
        assert res_ok.status_code == 200
        assert res_ok.get_json()["matrix"] == [[0.0, 8.5], [0.0, 0.0]]

        # Health endpoint remains healthy
        res_h = client.get("/api/health")
        assert res_h.status_code == 200
        assert res_h.get_json()["status"] == "healthy"

    def test_zero_15_direct_c_protocol_explicit_zero_rejected(self):
        """ZERO-15: Direct C/protocol input with an explicit zero is consistently rejected."""
        # A_ENTRY with explicit 0.0
        raw_req_a = (
            "CMD TO_DENSE v1\n"
            "D A_ROWS 2\n"
            "D A_COLS 2\n"
            "D A_FORMAT SPARSE\n"
            "D A_NNZ 1\n"
            "D A_ENTRY 0 1 0.0\n"
            "END\n"
        )
        stdout_a = execute_engine(raw_req_a)
        res_a = parse_response(stdout_a)
        assert res_a["ok"] is False
        assert res_a["code"] == "ZF_ERR_PARAM"
        assert "explicit_zero" in res_a["message"]

        # B_ENTRY with explicit -0.0
        raw_req_b = (
            "CMD ADD v1\n"
            "D A_ROWS 2\n"
            "D A_COLS 2\n"
            "D A_FORMAT SPARSE\n"
            "D A_NNZ 1\n"
            "D A_ENTRY 0 0 1.0\n"
            "D B_ROWS 2\n"
            "D B_COLS 2\n"
            "D B_FORMAT SPARSE\n"
            "D B_NNZ 1\n"
            "D B_ENTRY 1 1 -0.0\n"
            "END\n"
        )
        stdout_b = execute_engine(raw_req_b)
        res_b = parse_response(stdout_b)
        assert res_b["ok"] is False
        assert res_b["code"] == "ZF_ERR_PARAM"
        assert "explicit_zero" in res_b["message"]

    def test_zero_16_direct_validate_matrix_dict_unit(self):
        """Direct unit test for validate_matrix_dict explicit zero check."""
        for zero_val in [0, 0.0, -0.0]:
            mat = {
                "format": "sparse",
                "rows": 3,
                "cols": 3,
                "entries": [{"row": 1, "col": 1, "value": zero_val}]
            }
            with pytest.raises(ValidationError) as exc:
                validate_matrix_dict(mat, "test_mat")
            assert exc.value.status_code == 422
            assert exc.value.code == "ZF_ERR_PARAM"
            assert "explicit zero" in exc.value.message.lower()

        # Valid non-zero works
        mat_ok = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 1, "col": 1, "value": 1.0}]
        }
        res_dict = validate_matrix_dict(mat_ok, "test_mat")
        assert res_dict["nnz"] == 1
        assert res_dict["entries"][0]["value"] == 1.0





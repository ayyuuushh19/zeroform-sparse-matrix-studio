"""
test_frontend_integration.py — Integration and workflow verification for ZEROFORM Stage 6.

Validates that:
1. All frontend markup elements, controls, dialogs, and scripts are served correctly.
2. The complete 20-step operational workflow functions end-to-end through Flask and the real C engine.
3. Coordinate translation, error codes, and result schemas conform exactly to protocol specifications.
"""

import pytest
from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestFrontendMarkupAndAssets:
    """Verify that all required Stage 6 controls, buttons, and dialogs exist in index.html."""

    def test_homepage_serves_all_required_controls(self, client):
        res = client.get("/")
        assert res.status_code == 200
        html = res.get_data(as_text=True)

        # Header badges and triggers
        assert 'id="status-badge"' in html
        assert 'id="btn-open-help"' in html

        # Top preset strip
        assert 'id="sample-preset-select"' in html
        assert 'id="btn-load-preset"' in html
        assert 'id="btn-toggle-matrix-b"' in html

        # Matrix A controls
        assert 'id="dim-a-rows"' in html
        assert 'id="dim-a-cols"' in html
        assert 'id="btn-resize-a"' in html
        assert 'id="btn-paste-a"' in html
        assert 'id="btn-view-a-dense"' in html
        assert 'id="btn-view-a-triplets"' in html
        assert 'id="table-matrix-a"' in html

        # Matrix B controls
        assert 'id="panel-matrix-b"' in html
        assert 'id="dim-b-rows"' in html
        assert 'id="dim-b-cols"' in html
        assert 'id="btn-resize-b"' in html
        assert 'id="btn-paste-b"' in html
        assert 'id="btn-view-b-dense"' in html
        assert 'id="btn-view-b-triplets"' in html
        assert 'id="table-matrix-b"' in html

        # Command Deck buttons
        assert 'id="op-to-sparse"' in html
        assert 'id="op-to-dense"' in html
        assert 'id="op-transpose"' in html
        assert 'id="op-add"' in html
        assert 'id="op-multiply"' in html
        assert 'id="op-analyze"' in html
        assert 'id="op-reset"' in html

        # Result Inspector panel & tabs
        assert 'id="result-badge"' in html
        assert 'id="result-stale-badge"' in html
        assert 'id="btn-export-dense"' in html
        assert 'id="btn-export-triplets"' in html
        assert 'id="btn-copy-triplets"' in html
        assert 'data-tab="tab-dense"' in html
        assert 'data-tab="tab-triplets"' in html
        assert 'data-tab="tab-stats"' in html
        assert 'data-tab="tab-memory"' in html

        # Verify Sparse Element Inspector is removed
        assert 'id="input-row"' not in html
        assert 'id="input-col"' not in html
        assert 'id="input-val"' not in html
        assert 'id="btn-crud-insert"' not in html
        assert 'id="btn-crud-set"' not in html
        assert 'id="btn-crud-delete"' not in html
        assert 'id="btn-crud-search"' not in html
        assert 'id="btn-crud-traverse"' not in html
        assert 'id="btn-crud-sort"' not in html
        assert 'id="crud-feedback"' not in html

        # Telemetry panel
        assert 'id="telemetry-stale-badge"' in html
        assert 'id="metric-dims"' in html
        assert 'id="metric-total-cells"' in html
        assert 'id="metric-nnz"' in html
        assert 'id="metric-zeros"' in html
        assert 'id="metric-density"' in html
        assert 'id="metric-sparsity"' in html
        assert 'id="metric-dense-bytes"' in html
        assert 'id="metric-sparse-bytes"' in html
        assert 'id="metric-savings-badge"' in html
        assert 'id="scale-bar-fill"' in html

        # Dialogs
        assert 'id="help-dialog"' in html
        assert 'id="reset-dialog"' in html
        assert 'id="paste-dialog"' in html

    def test_static_assets_serve_correctly(self, client):
        css_res = client.get("/static/css/style.css")
        assert css_res.status_code == 200
        css_text = css_res.get_data(as_text=True)
        assert ".panel-tag.stale" in css_text
        assert ".btn.in-flight" in css_text
        assert ".dim-input.invalid" in css_text
        assert 'aria-invalid="true"' in css_text

        js_res = client.get("/static/js/app.js")
        assert js_res.status_code == 200
        js_text = js_res.get_data(as_text=True)
        assert "displayToWire" in js_text
        assert "apiClient" in js_text
        assert "parseStrictDimension" in js_text
        assert "validateMatrixDimensions" in js_text
        assert "applyMatrixResize" in js_text


class TestEndToEndOperationFlows:
    """Verify that all 20 operations function accurately through Flask and the C CLI engine."""

    def test_flow_01_load_samples_endpoint(self, client):
        res = client.get("/api/samples")
        assert res.status_code == 200
        data = res.get_json()
        assert "sparse_sample" in data
        assert "addition_pair" in data
        assert "multiplication_pair" in data

    def test_flow_02_dense_to_sparse_conversion(self, client):
        # 3x3 matrix with (0,0)=1.0, (1,2)=2.0
        grid = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0]
        ]
        res = client.post("/api/operation", json={
            "operation": "to_sparse",
            "matrix_a": grid
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 3
        assert data["cols"] == 3
        assert data["nnz"] == 2
        assert data["triplets"] == [
            {"row": 0, "col": 0, "value": 1.0},
            {"row": 1, "col": 2, "value": 2.0}
        ]

    def test_flow_03_sparse_to_dense_conversion(self, client):
        sparse_input = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [
                {"row": 0, "col": 0, "value": 4.5},
                {"row": 2, "col": 1, "value": -1.2}
            ]
        }
        res = client.post("/api/operation", json={
            "operation": "to_dense",
            "matrix_a": sparse_input
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 3
        assert data["cols"] == 3
        assert data["matrix"] == [
            [4.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.2, 0.0]
        ]

    def test_flow_04_addition_with_cancellation(self, client):
        # A: (0,0)=3.0, (0,1)=2.0
        # B: (0,0)=-3.0, (0,1)=5.0
        mat_a = [
            [3.0, 2.0],
            [0.0, 0.0]
        ]
        mat_b = [
            [-3.0, 5.0],
            [0.0, 0.0]
        ]
        res = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 2
        assert data["cols"] == 2
        # (0,0) cancels to 0.0, remaining is (0,1) = 7.0
        assert data["nnz"] == 1
        assert data["triplets"] == [
            {"row": 0, "col": 1, "value": 7.0}
        ]

    def test_flow_05_rectangular_multiplication(self, client):
        # A: 2x3, B: 3x2 -> 2x2
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
        assert data["triplets"] == [
            {"row": 0, "col": 0, "value": 5.0},
            {"row": 0, "col": 1, "value": 12.0},
            {"row": 1, "col": 0, "value": 28.0},
            {"row": 1, "col": 1, "value": 50.0}
        ]

    def test_flow_06_transpose_rectangular(self, client):
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
        assert len(data["triplets"]) == 6

    def test_flow_07_memory_analysis(self, client):
        grid = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
        res = client.post("/api/analyze", json={"matrix": grid})
        assert res.status_code == 200
        data = res.get_json()
        assert data["rows"] == 3
        assert data["cols"] == 4
        assert data["total_cells"] == 12
        assert data["nnz"] == 2
        assert data["zero_count"] == 10
        assert data["dense_payload_bytes"] == 96   # 12 * 8
        assert data["sparse_payload_bytes"] == 32  # 2 * 16
        assert data["sparse_is_smaller"] is True
        assert data["savings_bytes"] == 64

    def test_flow_08_sparse_crud_lifecycle(self, client):
        # 1. Start with 1 entry at wire [0, 0] (display [1, 1])
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]
        }

        # 2. Insert new coordinate [1, 2] (display [2, 3]) with value 4.0
        ins_res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": mat,
            "row": 1,
            "col": 2,
            "value": 4.0
        })
        assert ins_res.status_code == 200
        ins_data = ins_res.get_json()
        assert ins_data["nnz"] == 2

        # Frontend adapter wraps state into sparse format
        mat_after_ins = {
            "format": "sparse",
            "rows": ins_data["rows"],
            "cols": ins_data["cols"],
            "entries": ins_data["triplets"]
        }

        # 3. Duplicate insert fails with 422 ZF_ERR_DUPLICATE
        dup_res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": mat_after_ins,
            "row": 1,
            "col": 2,
            "value": 9.9
        })
        assert dup_res.status_code == 422
        assert dup_res.get_json()["error"]["code"] == "ZF_ERR_DUPLICATE"

        # 4. Search existing coordinate
        search_res = client.post("/api/entry", json={
            "action": "search",
            "matrix": mat_after_ins,
            "row": 1,
            "col": 2
        })
        assert search_res.status_code == 200
        s_data = search_res.get_json()
        assert s_data["found"] is True
        assert s_data["value"] == 4.0

        # 5. Search absent coordinate (implicit zero)
        search_zero = client.post("/api/entry", json={
            "action": "search",
            "matrix": mat_after_ins,
            "row": 2,
            "col": 2
        })
        assert search_zero.status_code == 200
        assert search_zero.get_json()["found"] is False
        assert search_zero.get_json()["value"] == 0.0

        # 6. Set / update existing coordinate
        set_res = client.post("/api/entry", json={
            "action": "set",
            "matrix": mat_after_ins,
            "row": 1,
            "col": 2,
            "value": 8.0
        })
        assert set_res.status_code == 200
        set_data = set_res.get_json()
        assert set_data["nnz"] == 2

        mat_after_set = {
            "format": "sparse",
            "rows": set_data["rows"],
            "cols": set_data["cols"],
            "entries": set_data["triplets"]
        }

        # 7. Set to 0.0 deletes the entry
        set_zero = client.post("/api/entry", json={
            "action": "set",
            "matrix": mat_after_set,
            "row": 1,
            "col": 2,
            "value": 0.0
        })
        assert set_zero.status_code == 200
        zero_data = set_zero.get_json()
        assert zero_data["nnz"] == 1

        mat_after_zero = {
            "format": "sparse",
            "rows": zero_data["rows"],
            "cols": zero_data["cols"],
            "entries": zero_data["triplets"]
        }

        # 8. Delete missing coordinate returns 404
        del_absent = client.post("/api/entry", json={
            "action": "delete",
            "matrix": mat_after_zero,
            "row": 1,
            "col": 2
        })
        assert del_absent.status_code == 404
        assert del_absent.get_json()["error"]["code"] == "ZF_ERR_NOT_FOUND"

        # 9. Delete existing coordinate [0, 0]
        del_res = client.post("/api/entry", json={
            "action": "delete",
            "matrix": mat_after_zero,
            "row": 0,
            "col": 0
        })
        assert del_res.status_code == 200
        assert del_res.get_json()["nnz"] == 0

    def test_flow_09_dimension_mismatch_addition_fails(self, client):
        mat_a = [[1.0, 2.0]]
        mat_b = [[1.0], [2.0]]
        res = client.post("/api/operation", json={
            "operation": "add",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_DIMENSION"

    def test_flow_10_dimension_mismatch_multiplication_fails(self, client):
        mat_a = [[1.0, 2.0]]  # 1x2
        mat_b = [[1.0, 2.0]]  # 1x2 -> cols(2) != rows(1)
        res = client.post("/api/operation", json={
            "operation": "multiply",
            "matrix_a": mat_a,
            "matrix_b": mat_b
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_DIMENSION"

    def test_flow_11_out_of_bounds_coordinate_fails(self, client):
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "entries": []
        }
        res = client.post("/api/entry", json={
            "action": "insert",
            "matrix": mat,
            "row": 5,
            "col": 1,
            "value": 3.0
        })
        assert res.status_code == 422
        assert res.get_json()["error"]["code"] == "ZF_ERR_INDEX"


class TestFrontendDimensionValidationRegression:
    """Automated regression testing for P2: Fractional Matrix Dimensions."""

    def test_frontend_dimension_validation_suite(self):
        """Execute Node.js test runner verifying cases A-01..A-10, B-01..B-03, R-01..R-05."""
        import subprocess
        import os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_script = os.path.join(repo_root, "tests", "test_frontend_validation.js")

        result = subprocess.run(
            ["node", test_script],
            cwd=repo_root,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Frontend validation test suite failed:\n{result.stderr}\n{result.stdout}"
        assert "All 18/18 frontend dimension validation tests passed!" in result.stdout


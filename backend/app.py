"""
app.py — Flask REST API for ZEROFORM Sparse Matrix Studio.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.

Enforces:
1. Pure process coordination: No matrix arithmetic in Python.
2. Complete input validation and strict error response schema.
3. Subprocess boundary with C engine via protocol.py and engine.py.
4. HTTP Status Codes:
   - 200 OK
   - 400 Bad Request
   - 404 Not Found (absent coordinate)
   - 413 Payload Too Large (exceeds cell/dim limits)
   - 422 Unprocessable Entity (dimension mismatch, duplicate coordinate)
   - 500 Internal Server Error (C process crash or unhandled error)
   - 503 Service Unavailable (C engine missing or unresponsive)
   - 504 Gateway Timeout (C engine execution timed out)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify, Response, render_template

from backend.engine import (
    execute_engine,
    ping_engine,
    get_engine_path,
    EngineError,
    EngineNotFoundError,
    EngineTimeoutError,
    EngineCrashError
)
from backend.protocol import (
    serialize_request,
    parse_response,
    ProtocolError,
    ProtocolFramingError,
    ProtocolParseError
)
from backend.validator import (
    ValidationError,
    validate_matrix_dict,
    validate_operation_payload,
    validate_entry_payload,
    validate_cell_limit
)


def c_status_to_http(code: str) -> int:
    """Map real C ZF_Status code names to standard HTTP status codes."""
    mapping = {
        "ZF_ERR_NOT_FOUND": 404,
        "ZF_ERR_DIMENSION": 422,
        "ZF_ERR_DUPLICATE": 422,
        "ZF_ERR_INDEX": 422,
        "ZF_ERR_VALUE": 422,
        "ZF_ERR_OVERFLOW": 413,
        "ZF_ERR_PARAM": 400,
        "ZF_ERR_NOMEM": 500,
        "ZF_ERR_INTERNAL": 500,
    }
    return mapping.get(code, 400)


def create_app() -> Flask:
    """Application factory for ZEROFORM Flask backend."""
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "frontend"
    templates_dir = frontend_dir / "templates"
    static_dir = frontend_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
        static_url_path="/static"
    )

    @app.route("/", methods=["GET"])
    def index() -> str:
        """Render the ZEROFORM application shell."""
        return render_template("index.html")

    # ── Error Handlers ──────────────────────────────────────────────
    @app.errorhandler(ValidationError)
    def handle_validation_error(err: ValidationError) -> Tuple[Response, int]:
        return jsonify(err.to_dict()), err.status_code

    @app.errorhandler(EngineNotFoundError)
    def handle_engine_not_found(err: EngineNotFoundError) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "ENGINE_UNAVAILABLE",
                "message": "C engine executable is unavailable or not found."
            }
        }), 503

    @app.errorhandler(EngineTimeoutError)
    def handle_engine_timeout(err: EngineTimeoutError) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "ENGINE_TIMEOUT",
                "message": "C engine execution timed out."
            }
        }), 504

    @app.errorhandler(EngineCrashError)
    def handle_engine_crash(err: EngineCrashError) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "ENGINE_CRASH",
                "message": f"C engine process exited abnormally (code {err.exit_code})."
            }
        }), 500

    @app.errorhandler(ProtocolError)
    def handle_protocol_error(err: ProtocolError) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "PROTOCOL_ERROR",
                "message": f"Engine communication protocol failure: {str(err)}"
            }
        }), 500

    @app.errorhandler(400)
    def handle_bad_request(err: Any) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "BAD_REQUEST",
                "message": "Malformed JSON or request syntax."
            }
        }), 400

    @app.errorhandler(404)
    def handle_not_found(err: Any) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested API endpoint was not found."
            }
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(err: Any) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": "HTTP method not allowed for this route."
            }
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(err: Any) -> Tuple[Response, int]:
        return jsonify({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred."
            }
        }), 500

    # ── Routes ──────────────────────────────────────────────────────

    @app.route("/api/health", methods=["GET"])
    def get_health() -> Tuple[Response, int]:
        """
        Verify that the C engine executable is present and responds to PING.
        """
        try:
            ping_info = ping_engine(timeout=2.0)
            return jsonify({
                "status": "healthy",
                "engine": "connected",
                "c_engine_version": ping_info.get("version", "1.0.0"),
                "executable": str(get_engine_path().name)
            }), 200
        except EngineNotFoundError:
            return jsonify({
                "status": "unhealthy",
                "engine": "unavailable",
                "error": "C engine executable not found"
            }), 503
        except EngineError as exc:
            return jsonify({
                "status": "unhealthy",
                "engine": "failed",
                "error": str(exc)
            }), 503

    @app.route("/api/samples", methods=["GET"])
    def get_samples() -> Tuple[Response, int]:
        """
        Return sample input matrices for interactive testing.
        """
        samples = {
            "sparse_sample": {
                "rows": 4,
                "cols": 5,
                "format": "sparse",
                "entries": [
                    {"row": 0, "col": 1, "value": 3.5},
                    {"row": 1, "col": 4, "value": 7.0},
                    {"row": 3, "col": 2, "value": -2.1}
                ]
            },
            "addition_pair": {
                "matrix_a": {
                    "rows": 2,
                    "cols": 3,
                    "format": "dense",
                    "data": [
                        [1.0, 0.0, 2.0],
                        [0.0, 3.0, 0.0]
                    ]
                },
                "matrix_b": {
                    "rows": 2,
                    "cols": 3,
                    "format": "dense",
                    "data": [
                        [-1.0, 4.0, 0.0],
                        [5.0, -3.0, 7.0]
                    ]
                }
            },
            "multiplication_pair": {
                "matrix_a": {
                    "rows": 2,
                    "cols": 3,
                    "format": "dense",
                    "data": [
                        [1.0, 2.0, 0.0],
                        [0.0, 3.0, 4.0]
                    ]
                },
                "matrix_b": {
                    "rows": 3,
                    "cols": 2,
                    "format": "dense",
                    "data": [
                        [5.0, 0.0],
                        [0.0, 6.0],
                        [7.0, 8.0]
                    ]
                }
            }
        }
        return jsonify(samples), 200

    @app.route("/api/analyze", methods=["POST"])
    def post_analyze() -> Tuple[Response, int]:
        """
        Analyze dense or sparse matrix memory characteristics using the C engine.
        """
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Request body must be a valid JSON object", 400)

        raw_mat = payload.get("matrix", payload.get("matrix_a", payload))
        mat = validate_matrix_dict(raw_mat, "matrix")

        req_text = serialize_request("ANALYZE", matrix_a=mat)
        stdout = execute_engine(req_text)
        res = parse_response(stdout)

        if not res["ok"]:
            http_code = c_status_to_http(res.get("code", "ZF_ERR_INTERNAL"))
            return jsonify({
                "error": {
                    "code": res.get("code", "ZF_ERR_INTERNAL"),
                    "message": res.get("message", "Analysis failed in C engine")
                }
            }), http_code

        return jsonify({
            "rows": res["rows"],
            "cols": res["cols"],
            "total_cells": res["total_cells"],
            "nnz": res["nnz"],
            "zero_count": res["zero_count"],
            "density_percent": res["density"],
            "sparsity_percent": res["sparsity"],
            "dense_payload_bytes": res["dense_bytes"],
            "sparse_payload_bytes": res["sparse_bytes"],
            "sparse_is_smaller": res["sparse_is_smaller"],
            "savings_bytes": res["savings_bytes"],
            "percent_saved": res["percent_saved"],
            "triplets": res["entries"]
        }), 200

    @app.route("/api/operation", methods=["POST"])
    def post_operation() -> Tuple[Response, int]:
        """
        Execute core matrix operations (to_sparse, to_dense, add, multiply, transpose).
        All computation delegated to the C engine.
        """
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Request body must be a valid JSON object", 400)

        op, mat_a, mat_b = validate_operation_payload(payload)

        req_text = serialize_request(op.upper(), matrix_a=mat_a, matrix_b=mat_b)
        stdout = execute_engine(req_text)
        res = parse_response(stdout)

        if not res["ok"]:
            http_code = c_status_to_http(res.get("code", "ZF_ERR_INTERNAL"))
            return jsonify({
                "error": {
                    "code": res.get("code", "ZF_ERR_INTERNAL"),
                    "message": res.get("message", f"Operation '{op}' failed in C engine")
                }
            }), http_code

        # For to_dense, reconstruct 2D matrix from C cells for client convenience
        if op == "to_dense":
            validate_cell_limit(res["rows"], res["cols"], "dense_materialization")
            grid = [[0.0 for _ in range(res["cols"])] for _ in range(res["rows"])]
            for cell in res["cells"]:
                grid[cell["row"]][cell["col"]] = cell["value"]
            return jsonify({
                "operation": op,
                "rows": res["rows"],
                "cols": res["cols"],
                "cells": res["cells"],
                "matrix": grid
            }), 200

        # For sparse-output operations
        return jsonify({
            "operation": op,
            "rows": res["rows"],
            "cols": res["cols"],
            "nnz": res["nnz"],
            "triplets": res["entries"]
        }), 200

    @app.route("/api/entry", methods=["POST"])
    def post_entry() -> Tuple[Response, int]:
        """
        Execute entry-level actions (insert, set, delete, search, traverse, sort).
        All data structures managed in C.
        """
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Request body must be a valid JSON object", 400)

        action, mat, params = validate_entry_payload(payload)

        req_text = serialize_request(action.upper(), matrix_a=mat, params=params)
        stdout = execute_engine(req_text)
        res = parse_response(stdout)

        if not res["ok"]:
            http_code = c_status_to_http(res.get("code", "ZF_ERR_INTERNAL"))
            return jsonify({
                "error": {
                    "code": res.get("code", "ZF_ERR_INTERNAL"),
                    "message": res.get("message", f"Entry action '{action}' failed in C engine")
                }
            }), http_code

        if action == "search":
            return jsonify({
                "action": "search",
                "found": res["found"],
                "row": res["row"],
                "col": res["col"],
                "value": res["value"],
                "pos": res.get("pos", 0)
            }), 200

        # For mutations and traversal/sorting
        return jsonify({
            "action": action,
            "rows": res["rows"],
            "cols": res["cols"],
            "nnz": res["nnz"],
            "triplets": res["entries"]
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )

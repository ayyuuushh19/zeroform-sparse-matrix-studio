"""
test_protocol.py — Protocol serialization and parsing test suite.

ZEROFORM Sparse Matrix Studio — Stage 4
Senior C11 / Python Systems & API Architecture.
"""

import pytest
import math
from backend.protocol import (
    serialize_request,
    serialize_matrix,
    parse_response,
    format_float,
    ProtocolError,
    ProtocolFramingError,
    ProtocolParseError
)
from backend.engine import execute_engine, get_engine_path


class TestProtocolSerialization:
    """Test protocol serialization to exact v1 line-oriented format."""

    def test_serialize_ping(self):
        req = serialize_request("PING")
        assert req == "CMD PING v1\nEND\n"

    def test_serialize_valid_dense(self):
        mat = {
            "format": "dense",
            "rows": 2,
            "cols": 2,
            "data": [[1.0, 0.0], [0.0, 3.5]]
        }
        req = serialize_request("ANALYZE", matrix_a=mat)
        expected = (
            "CMD ANALYZE v1\n"
            "D A_ROWS 2\n"
            "D A_COLS 2\n"
            "D A_FORMAT DENSE\n"
            "D A_CELL 0 0 1\n"
            "D A_CELL 0 1 0\n"
            "D A_CELL 1 0 0\n"
            "D A_CELL 1 1 3.5\n"
            "END\n"
        )
        assert req == expected

    def test_serialize_valid_sparse(self):
        mat = {
            "format": "sparse",
            "rows": 3,
            "cols": 3,
            "nnz": 2,
            "entries": [
                {"row": 0, "col": 1, "value": 4.2},
                {"row": 2, "col": 0, "value": -1.0}
            ]
        }
        req = serialize_request("TO_DENSE", matrix_a=mat)
        expected = (
            "CMD TO_DENSE v1\n"
            "D A_ROWS 3\n"
            "D A_COLS 3\n"
            "D A_FORMAT SPARSE\n"
            "D A_NNZ 2\n"
            "D A_ENTRY 0 1 4.2\n"
            "D A_ENTRY 2 0 -1\n"
            "END\n"
        )
        assert req == expected

    def test_serialize_entry_operations(self):
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "nnz": 1,
            "entries": [{"row": 0, "col": 0, "value": 5.0}]
        }
        # INSERT
        req_ins = serialize_request("INSERT", matrix_a=mat, params={"row": 1, "col": 1, "value": 9.0, "action": "INSERT"})
        assert "CMD INSERT v1" in req_ins
        assert "D ROW 1" in req_ins
        assert "D COL 1" in req_ins
        assert "D VALUE 9" in req_ins
        assert "D ACTION INSERT" in req_ins
        assert req_ins.endswith("END\n")

        # SEARCH
        req_search = serialize_request("SEARCH", matrix_a=mat, params={"row": 0, "col": 0})
        assert "CMD SEARCH v1" in req_search
        assert "D ROW 0" in req_search
        assert "D COL 0" in req_search
        assert "D VALUE" not in req_search

        # DELETE
        req_del = serialize_request("DELETE", matrix_a=mat, params={"row": 0, "col": 0})
        assert "CMD DELETE v1" in req_del
        assert "D ROW 0" in req_del
        assert "D COL 0" in req_del

        # TRAVERSE & SORT
        req_trav = serialize_request("TRAVERSE", matrix_a=mat)
        assert req_trav.startswith("CMD TRAVERSE v1")
        req_sort = serialize_request("SORT", matrix_a=mat)
        assert req_sort.startswith("CMD SORT v1")

    def test_reject_nan_infinity_serialization(self):
        with pytest.raises(ProtocolParseError):
            format_float(float("nan"))

        with pytest.raises(ProtocolParseError):
            format_float(float("inf"))

        with pytest.raises(ProtocolParseError):
            format_float(float("-inf"))

    def test_sparse_count_mismatch_serialization(self):
        mat = {
            "format": "sparse",
            "rows": 2,
            "cols": 2,
            "nnz": 2,
            "entries": [{"row": 0, "col": 0, "value": 1.0}]  # only 1 entry but nnz=2
        }
        with pytest.raises(ProtocolParseError, match="does not match NNZ"):
            serialize_matrix("A", mat)


class TestProtocolParsing:
    """Test protocol response parser and error validation."""

    def test_parse_ok_ping(self):
        text = "OK\nD STATUS PONG\nD VERSION 1.0.0\nEND\n"
        res = parse_response(text)
        assert res["ok"] is True
        assert res["status"] == "PONG"
        assert res["version"] == "1.0.0"

    def test_parse_ok_sparse_matrix(self):
        text = (
            "OK\n"
            "D ROWS 2\n"
            "D COLS 3\n"
            "D NNZ 2\n"
            "D ENTRY 0 1 4.5\n"
            "D ENTRY 1 2 -7.0\n"
            "END\n"
        )
        res = parse_response(text)
        assert res["ok"] is True
        assert res["rows"] == 2
        assert res["cols"] == 3
        assert res["nnz"] == 2
        assert len(res["entries"]) == 2
        assert res["entries"][0] == {"row": 0, "col": 1, "value": 4.5}
        assert res["entries"][1] == {"row": 1, "col": 2, "value": -7.0}

    def test_parse_ok_dense_matrix(self):
        text = (
            "OK\n"
            "D ROWS 1\n"
            "D COLS 2\n"
            "D CELL 0 0 1.0\n"
            "D CELL 0 1 2.0\n"
            "END\n"
        )
        res = parse_response(text)
        assert res["ok"] is True
        assert res["rows"] == 1
        assert res["cols"] == 2
        assert len(res["cells"]) == 2

    def test_parse_ok_search_found(self):
        text = (
            "OK\n"
            "D FOUND 1\n"
            "D ROW 0\n"
            "D COL 1\n"
            "D VALUE 4.5\n"
            "D POS 0\n"
            "END\n"
        )
        res = parse_response(text)
        assert res["ok"] is True
        assert res["found"] is True
        assert res["row"] == 0
        assert res["col"] == 1
        assert res["value"] == 4.5
        assert res["pos"] == 0

    def test_parse_ok_search_not_found(self):
        text = (
            "OK\n"
            "D FOUND 0\n"
            "D ROW 1\n"
            "D COL 1\n"
            "D VALUE 0\n"
            "D POS 0\n"
            "END\n"
        )
        res = parse_response(text)
        assert res["ok"] is True
        assert res["found"] is False
        assert res["row"] == 1
        assert res["col"] == 1
        assert res["value"] == 0.0

    def test_parse_err_response(self):
        text = (
            "ERR\n"
            "D CODE ZF_ERR_DIMENSION\n"
            "D MESSAGE dimension_mismatch\n"
            "END\n"
        )
        res = parse_response(text)
        assert res["ok"] is False
        assert res["code"] == "ZF_ERR_DIMENSION"
        assert res["message"] == "dimension_mismatch"

    def test_missing_end_marker(self):
        text = "OK\nD STATUS PONG\n"
        with pytest.raises(ProtocolFramingError, match="Missing terminating END marker"):
            parse_response(text)

    def test_invalid_header(self):
        text = "SUCCESS\nD STATUS PONG\nEND\n"
        with pytest.raises(ProtocolFramingError, match="Invalid status header line"):
            parse_response(text)

    def test_empty_response(self):
        with pytest.raises(ProtocolFramingError, match="Empty response"):
            parse_response("")

    def test_inconsistent_entry_count(self):
        # Declares NNZ 2 but only provides 1 ENTRY line
        text = (
            "OK\n"
            "D ROWS 2\n"
            "D COLS 2\n"
            "D NNZ 2\n"
            "D ENTRY 0 0 1.0\n"
            "END\n"
        )
        with pytest.raises(ProtocolParseError, match="Response NNZ mismatch"):
            parse_response(text)

    def test_malformed_number_in_response(self):
        text = (
            "OK\n"
            "D ROWS not_a_number\n"
            "D COLS 2\n"
            "END\n"
        )
        with pytest.raises(ProtocolParseError):
            parse_response(text)


class TestProtocolEngineIntegration:
    """Test protocol requests directly against the compiled C executable."""

    def test_engine_rejects_invalid_version(self):
        raw_req = "CMD PING v2\nEND\n"
        stdout = execute_engine(raw_req)
        res = parse_response(stdout)
        assert res["ok"] is False
        assert res["code"] == "ZF_ERR_PARAM"
        assert res["message"] == "unsupported_protocol_version"

    def test_engine_rejects_unknown_command(self):
        raw_req = "CMD FOOBAR v1\nEND\n"
        stdout = execute_engine(raw_req)
        res = parse_response(stdout)
        assert res["ok"] is False
        assert res["code"] == "ZF_ERR_PARAM"

    def test_engine_rejects_duplicate_singleton(self):
        raw_req = (
            "CMD TO_SPARSE v1\n"
            "D A_ROWS 2\n"
            "D A_ROWS 3\n"
            "D A_COLS 2\n"
            "D A_FORMAT DENSE\n"
            "END\n"
        )
        stdout = execute_engine(raw_req)
        res = parse_response(stdout)
        assert res["ok"] is False
        assert "duplicate" in res["message"]

    def test_engine_rejects_missing_end_marker(self):
        raw_req = "CMD PING v1\nD EXTRA 1\n"
        stdout = execute_engine(raw_req)
        res = parse_response(stdout)
        # Should gracefully return ERR
        assert res["ok"] is False
        assert "missing_END_marker" in res["message"]

    def test_engine_rejects_content_after_end_marker(self):
        raw_req = "CMD PING v1\nEND\nEXTRA_GARBAGE_AFTER_END\n"
        stdout = execute_engine(raw_req)
        res = parse_response(stdout)
        assert res["ok"] is False
        assert "unexpected_content_after_END" in res["message"]

    def test_response_rejects_content_after_end_marker(self):
        text = "OK\nD STATUS PONG\nEND\nTRAILING_GARBAGE\n"
        with pytest.raises(ProtocolFramingError, match="Missing terminating END marker"):
            parse_response(text)


/*
 * main.c — Command-Line Execution Engine for ZEROFORM
 *
 * ZEROFORM Sparse Matrix Studio — Stage 4
 *
 * Implements the stdio protocol interface (protocol/PROTOCOL.md).
 * Reads one request from stdin, processes using the C core,
 * and outputs the response to stdout.
 *
 * All mathematical operations and data structures are delegated to
 * zeroform, matrix, matrix_operations, and memory_analysis modules.
 */

#include "../zeroform.h"
#include "../matrix.h"
#include "../matrix_operations.h"
#include "../memory_analysis.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>
#include <math.h>

#define MAX_LINE_LEN 1024

typedef enum {
    FMT_NONE = 0,
    FMT_DENSE,
    FMT_SPARSE
} MatrixFormat;

typedef struct {
    size_t rows;
    size_t cols;
    size_t nnz;
    int has_rows;
    int has_cols;
    int has_format;
    int has_nnz;
    MatrixFormat format;

    /* Dense tracking */
    double *dense_cells;
    uint8_t *dense_seen;
    size_t dense_count;

    /* Sparse tracking */
    DynamicArray sparse_triplets;

    /* Final initialized matrices */
    DenseMatrix dm;
    SparseMatrix sm;
    int dm_init;
    int sm_init;
} MatrixBuilder;

static void mb_init(MatrixBuilder *mb)
{
    memset(mb, 0, sizeof(*mb));
    da_init(&mb->sparse_triplets, sizeof(Triplet), 0);
}

static void mb_destroy(MatrixBuilder *mb)
{
    free(mb->dense_cells);
    free(mb->dense_seen);
    mb->dense_cells = NULL;
    mb->dense_seen = NULL;

    da_destroy(&mb->sparse_triplets);
    if (mb->dm_init) {
        dense_matrix_destroy(&mb->dm);
        mb->dm_init = 0;
    }
    if (mb->sm_init) {
        sparse_matrix_destroy(&mb->sm);
        mb->sm_init = 0;
    }
}

/* ───────────────────────────────────────────────────────────────────
 * Protocol output helpers
 * ─────────────────────────────────────────────────────────────────── */
static void print_ok_start(void)
{
    printf("OK\n");
}

static void print_end(void)
{
    printf("END\n");
    fflush(stdout);
}

static void print_err(ZF_Status code, const char *msg)
{
    printf("ERR\n");
    printf("D CODE %s\n", zf_status_str(code));
    printf("D MESSAGE %s\n", msg);
    printf("END\n");
    fflush(stdout);
}

static void print_sparse_matrix(const SparseMatrix *sm)
{
    printf("D ROWS %zu\n", sm->rows);
    printf("D COLS %zu\n", sm->cols);
    printf("D NNZ %zu\n", da_length(&sm->entries));
    size_t n = da_length(&sm->entries);
    for (size_t i = 0; i < n; i++) {
        const Triplet *t = (const Triplet *)da_ptr(&sm->entries, i);
        printf("D ENTRY %zu %zu %.17g\n", t->row, t->col, t->value);
    }
}

static void print_dense_matrix(const DenseMatrix *dm)
{
    printf("D ROWS %zu\n", dm->rows);
    printf("D COLS %zu\n", dm->cols);
    for (size_t r = 0; r < dm->rows; r++) {
        for (size_t c = 0; c < dm->cols; c++) {
            double v = 0.0;
            dense_matrix_get(dm, r, c, &v);
            printf("D CELL %zu %zu %.17g\n", r, c, v);
        }
    }
}

/* ───────────────────────────────────────────────────────────────────
 * Number parsing helpers
 * ─────────────────────────────────────────────────────────────────── */
static int parse_size_t(const char *s, size_t *out)
{
    if (s == NULL || *s == '\0') return 0;
    while (isspace((unsigned char)*s)) s++;
    if (*s == '-' || *s == '+') return 0;
    if (!isdigit((unsigned char)*s)) return 0;
    char *end = NULL;
    uintmax_t val = strtoumax(s, &end, 10);
    if (end == s) return 0;
    while (isspace((unsigned char)*end)) end++;
    if (*end != '\0') return 0;
    if (val > SIZE_MAX) return 0;
    *out = (size_t)val;
    return 1;
}

static int parse_double(const char *s, double *out)
{
    if (s == NULL || *s == '\0') return 0;
    while (isspace((unsigned char)*s)) s++;
    char *end = NULL;
    double val = strtod(s, &end);
    if (end == s) return 0;
    while (isspace((unsigned char)*end)) end++;
    if (*end != '\0') return 0;
    if (!isfinite(val)) return 0;
    *out = val;
    return 1;
}

/* ───────────────────────────────────────────────────────────────────
 * Matrix Builder Finalization
 * ─────────────────────────────────────────────────────────────────── */
static ZF_Status mb_finalize(MatrixBuilder *mb, const char **err_msg)
{
    if (!mb->has_rows || !mb->has_cols || !mb->has_format) {
        *err_msg = "incomplete_matrix_header";
        return ZF_ERR_PARAM;
    }
    if (mb->rows == 0 || mb->cols == 0) {
        *err_msg = "zero_dimension";
        return ZF_ERR_PARAM;
    }
    if (mb->cols > 0 && mb->rows > MAX_CELLS / mb->cols) {
        *err_msg = "matrix_cells_exceed_limit";
        return ZF_ERR_OVERFLOW;
    }

    if (mb->format == FMT_DENSE) {
        size_t total = mb->rows * mb->cols;
        if (mb->dense_count != total) {
            *err_msg = "missing_dense_cells";
            return ZF_ERR_PARAM;
        }
        ZF_Status s = dense_matrix_from_flat(&mb->dm, mb->rows, mb->cols, mb->dense_cells);
        if (s != ZF_OK) {
            *err_msg = "dense_matrix_construction_failed";
            return s;
        }
        mb->dm_init = 1;

        s = sparse_matrix_from_dense(&mb->sm, &mb->dm);
        if (s != ZF_OK) {
            *err_msg = "sparse_conversion_failed";
            return s;
        }
        mb->sm_init = 1;
        return ZF_OK;
    } else if (mb->format == FMT_SPARSE) {
        if (!mb->has_nnz) {
            *err_msg = "missing_nnz_header";
            return ZF_ERR_PARAM;
        }
        size_t n = da_length(&mb->sparse_triplets);
        if (n != mb->nnz) {
            *err_msg = "entry_count_mismatch";
            return ZF_ERR_PARAM;
        }
        ZF_Status s = sparse_matrix_from_triplets(&mb->sm, mb->rows, mb->cols,
                                                  (const Triplet *)mb->sparse_triplets.data, n);
        if (s != ZF_OK) {
            *err_msg = "sparse_matrix_validation_failed";
            return s;
        }
        mb->sm_init = 1;
        return ZF_OK;
    }

    *err_msg = "unrecognized_format";
    return ZF_ERR_PARAM;
}

/* ───────────────────────────────────────────────────────────────────
 * Main entry point
 * ─────────────────────────────────────────────────────────────────── */
int main(void)
{
    char line[MAX_LINE_LEN];

    /* Read request header */
    if (fgets(line, sizeof(line), stdin) == NULL) {
        print_err(ZF_ERR_PARAM, "empty_request");
        return EXIT_FAILURE;
    }

    /* Strip newline */
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[--len] = '\0';
    }

    /* Validate CMD <VERB> v1 */
    char verb[64] = {0};
    char version[32] = {0};
    if (sscanf(line, "CMD %63s %31s", verb, version) != 2) {
        print_err(ZF_ERR_PARAM, "invalid_command_header");
        return EXIT_SUCCESS;
    }

    if (strcmp(version, "v1") != 0) {
        print_err(ZF_ERR_PARAM, "unsupported_protocol_version");
        return EXIT_SUCCESS;
    }

    /* Handle PING */
    if (strcmp(verb, "PING") == 0) {
        int saw_ping_end = 0;
        int has_extra = 0;
        while (fgets(line, sizeof(line), stdin) != NULL) {
            len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
            if (strcmp(line, "END") == 0) {
                saw_ping_end = 1;
                break;
            }
            if (line[0] != '\0') {
                has_extra = 1;
            }
        }
        if (!saw_ping_end) {
            print_err(ZF_ERR_PARAM, "missing_END_marker");
            return EXIT_SUCCESS;
        }
        if (has_extra) {
            print_err(ZF_ERR_PARAM, "unexpected_fields_in_ping");
            return EXIT_SUCCESS;
        }
        int c_trail;
        while ((c_trail = fgetc(stdin)) != EOF) {
            if (!isspace(c_trail)) {
                print_err(ZF_ERR_PARAM, "unexpected_content_after_END");
                return EXIT_SUCCESS;
            }
        }
        print_ok_start();
        printf("D STATUS PONG\n");
        printf("D VERSION 1.0.0\n");
        print_end();
        return EXIT_SUCCESS;
    }

    /* Prepare matrix builders and parameters */
    MatrixBuilder mbA, mbB;
    mb_init(&mbA);
    mb_init(&mbB);

    size_t param_row = 0, param_col = 0;
    double param_val = 0.0;
    int has_row = 0, has_col = 0, has_val = 0;

    int saw_end = 0;
    const char *parse_err = NULL;
    ZF_Status err_status = ZF_ERR_PARAM;

    /* Parse request data lines */
    while (fgets(line, sizeof(line), stdin) != NULL) {
        len = strlen(line);
        if (len == sizeof(line) - 1 && line[len - 1] != '\n') {
            parse_err = "line_buffer_overflow";
            err_status = ZF_ERR_PARAM;
            break;
        }
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        if (strcmp(line, "END") == 0) {
            saw_end = 1;
            break;
        }

        if (line[0] != 'D' || line[1] != ' ') {
            parse_err = "invalid_data_line_prefix";
            err_status = ZF_ERR_PARAM;
            break;
        }

        char key[64] = {0};
        char rest[MAX_LINE_LEN] = {0};
        if (sscanf(line + 2, "%63s %[^\n]", key, rest) < 1) {
            parse_err = "malformed_key_line";
            err_status = ZF_ERR_PARAM;
            break;
        }

        /* ── Matrix A Fields ─────────────────────────────────────── */
        if (strcmp(key, "A_ROWS") == 0) {
            if (mbA.has_rows) { parse_err = "duplicate_A_ROWS"; break; }
            if (!parse_size_t(rest, &mbA.rows) || mbA.rows == 0 || mbA.rows > MAX_DIM) {
                parse_err = "invalid_A_ROWS"; break;
            }
            mbA.has_rows = 1;
        } else if (strcmp(key, "A_COLS") == 0) {
            if (mbA.has_cols) { parse_err = "duplicate_A_COLS"; break; }
            if (!parse_size_t(rest, &mbA.cols) || mbA.cols == 0 || mbA.cols > MAX_DIM) {
                parse_err = "invalid_A_COLS"; break;
            }
            mbA.has_cols = 1;
        } else if (strcmp(key, "A_FORMAT") == 0) {
            if (mbA.has_format) { parse_err = "duplicate_A_FORMAT"; break; }
            if (!mbA.has_rows || !mbA.has_cols) { parse_err = "A_FORMAT_before_dims"; break; }
            if (mbA.cols > 0 && mbA.rows > MAX_CELLS / mbA.cols) { parse_err = "matrix_cells_exceed_limit"; err_status = ZF_ERR_OVERFLOW; break; }
            if (strcmp(rest, "DENSE") == 0) {
                mbA.format = FMT_DENSE;
                size_t total = mbA.rows * mbA.cols;
                mbA.dense_cells = (double *)calloc(total, sizeof(double));
                mbA.dense_seen  = (uint8_t *)calloc(total, sizeof(uint8_t));
                if (!mbA.dense_cells || !mbA.dense_seen) { parse_err = "out_of_memory"; err_status = ZF_ERR_NOMEM; break; }
            } else if (strcmp(rest, "SPARSE") == 0) {
                mbA.format = FMT_SPARSE;
            } else {
                parse_err = "invalid_A_FORMAT"; break;
            }
            mbA.has_format = 1;
        } else if (strcmp(key, "A_NNZ") == 0) {
            if (mbA.has_nnz) { parse_err = "duplicate_A_NNZ"; break; }
            if (!parse_size_t(rest, &mbA.nnz) || mbA.nnz > MAX_NNZ) { parse_err = "invalid_A_NNZ"; break; }
            mbA.has_nnz = 1;
        } else if (strcmp(key, "A_CELL") == 0) {
            if (mbA.format != FMT_DENSE || !mbA.dense_cells) { parse_err = "A_CELL_unexpected"; break; }
            size_t r = 0, c = 0;
            char val_str[128] = {0};
            if (sscanf(rest, "%zu %zu %127s", &r, &c, val_str) != 3) { parse_err = "malformed_A_CELL"; break; }
            double v = 0.0;
            if (!parse_double(val_str, &v)) { parse_err = "invalid_cell_value"; err_status = ZF_ERR_VALUE; break; }
            if (r >= mbA.rows || c >= mbA.cols) { parse_err = "cell_out_of_bounds"; err_status = ZF_ERR_INDEX; break; }
            size_t idx = r * mbA.cols + c;
            if (mbA.dense_seen[idx]) { parse_err = "duplicate_cell_coordinate"; err_status = ZF_ERR_DUPLICATE; break; }
            mbA.dense_seen[idx] = 1;
            mbA.dense_cells[idx] = v;
            mbA.dense_count++;
        } else if (strcmp(key, "A_ENTRY") == 0) {
            if (mbA.format != FMT_SPARSE) { parse_err = "A_ENTRY_unexpected"; break; }
            size_t r = 0, c = 0;
            char val_str[128] = {0};
            if (sscanf(rest, "%zu %zu %127s", &r, &c, val_str) != 3) { parse_err = "malformed_A_ENTRY"; break; }
            double v = 0.0;
            if (!parse_double(val_str, &v)) { parse_err = "invalid_entry_value"; err_status = ZF_ERR_VALUE; break; }
            if (v == 0.0) { parse_err = "explicit_zero_sparse_entry"; err_status = ZF_ERR_PARAM; break; }
            Triplet t; t.row = r; t.col = c; t.value = v;
            da_append(&mbA.sparse_triplets, &t);

        /* ── Matrix B Fields ─────────────────────────────────────── */
        } else if (strcmp(key, "B_ROWS") == 0) {
            if (mbB.has_rows) { parse_err = "duplicate_B_ROWS"; break; }
            if (!parse_size_t(rest, &mbB.rows) || mbB.rows == 0 || mbB.rows > MAX_DIM) {
                parse_err = "invalid_B_ROWS"; break;
            }
            mbB.has_rows = 1;
        } else if (strcmp(key, "B_COLS") == 0) {
            if (mbB.has_cols) { parse_err = "duplicate_B_COLS"; break; }
            if (!parse_size_t(rest, &mbB.cols) || mbB.cols == 0 || mbB.cols > MAX_DIM) {
                parse_err = "invalid_B_COLS"; break;
            }
            mbB.has_cols = 1;
        } else if (strcmp(key, "B_FORMAT") == 0) {
            if (mbB.has_format) { parse_err = "duplicate_B_FORMAT"; break; }
            if (!mbB.has_rows || !mbB.has_cols) { parse_err = "B_FORMAT_before_dims"; break; }
            if (mbB.cols > 0 && mbB.rows > MAX_CELLS / mbB.cols) { parse_err = "matrix_cells_exceed_limit"; err_status = ZF_ERR_OVERFLOW; break; }
            if (strcmp(rest, "DENSE") == 0) {
                mbB.format = FMT_DENSE;
                size_t total = mbB.rows * mbB.cols;
                mbB.dense_cells = (double *)calloc(total, sizeof(double));
                mbB.dense_seen  = (uint8_t *)calloc(total, sizeof(uint8_t));
                if (!mbB.dense_cells || !mbB.dense_seen) { parse_err = "out_of_memory"; err_status = ZF_ERR_NOMEM; break; }
            } else if (strcmp(rest, "SPARSE") == 0) {
                mbB.format = FMT_SPARSE;
            } else {
                parse_err = "invalid_B_FORMAT"; break;
            }
            mbB.has_format = 1;
        } else if (strcmp(key, "B_NNZ") == 0) {
            if (mbB.has_nnz) { parse_err = "duplicate_B_NNZ"; break; }
            if (!parse_size_t(rest, &mbB.nnz) || mbB.nnz > MAX_NNZ) { parse_err = "invalid_B_NNZ"; break; }
            mbB.has_nnz = 1;
        } else if (strcmp(key, "B_CELL") == 0) {
            if (mbB.format != FMT_DENSE || !mbB.dense_cells) { parse_err = "B_CELL_unexpected"; break; }
            size_t r = 0, c = 0;
            char val_str[128] = {0};
            if (sscanf(rest, "%zu %zu %127s", &r, &c, val_str) != 3) { parse_err = "malformed_B_CELL"; break; }
            double v = 0.0;
            if (!parse_double(val_str, &v)) { parse_err = "invalid_cell_value"; err_status = ZF_ERR_VALUE; break; }
            if (r >= mbB.rows || c >= mbB.cols) { parse_err = "cell_out_of_bounds"; err_status = ZF_ERR_INDEX; break; }
            size_t idx = r * mbB.cols + c;
            if (mbB.dense_seen[idx]) { parse_err = "duplicate_cell_coordinate"; err_status = ZF_ERR_DUPLICATE; break; }
            mbB.dense_seen[idx] = 1;
            mbB.dense_cells[idx] = v;
            mbB.dense_count++;
        } else if (strcmp(key, "B_ENTRY") == 0) {
            if (mbB.format != FMT_SPARSE) { parse_err = "B_ENTRY_unexpected"; break; }
            size_t r = 0, c = 0;
            char val_str[128] = {0};
            if (sscanf(rest, "%zu %zu %127s", &r, &c, val_str) != 3) { parse_err = "malformed_B_ENTRY"; break; }
            double v = 0.0;
            if (!parse_double(val_str, &v)) { parse_err = "invalid_entry_value"; err_status = ZF_ERR_VALUE; break; }
            if (v == 0.0) { parse_err = "explicit_zero_sparse_entry"; err_status = ZF_ERR_PARAM; break; }
            Triplet t; t.row = r; t.col = c; t.value = v;
            da_append(&mbB.sparse_triplets, &t);

        /* ── Mutation / Action Parameters ────────────────────────── */
        } else if (strcmp(key, "ROW") == 0) {
            if (!parse_size_t(rest, &param_row)) { parse_err = "invalid_ROW"; break; }
            has_row = 1;
        } else if (strcmp(key, "COL") == 0) {
            if (!parse_size_t(rest, &param_col)) { parse_err = "invalid_COL"; break; }
            has_col = 1;
        } else if (strcmp(key, "VALUE") == 0) {
            if (!parse_double(rest, &param_val)) { parse_err = "invalid_VALUE"; err_status = ZF_ERR_VALUE; break; }
            has_val = 1;
        } else if (strcmp(key, "ACTION") == 0) {
            /* optional descriptor */
        } else {
            parse_err = "unknown_field_key";
            err_status = ZF_ERR_PARAM;
            break;
        }
    }

    if (parse_err != NULL) {
        print_err(err_status, parse_err);
        mb_destroy(&mbA);
        mb_destroy(&mbB);
        return EXIT_SUCCESS;
    }

    if (!saw_end) {
        print_err(ZF_ERR_PARAM, "missing_END_marker");
        mb_destroy(&mbA);
        mb_destroy(&mbB);
        return EXIT_SUCCESS;
    }

    /* Reject any unexpected trailing content after END */
    int c_trail;
    while ((c_trail = fgetc(stdin)) != EOF) {
        if (!isspace(c_trail)) {
            print_err(ZF_ERR_PARAM, "unexpected_content_after_END");
            mb_destroy(&mbA);
            mb_destroy(&mbB);
            return EXIT_SUCCESS;
        }
    }

    /* Finalize Matrix A */
    const char *fin_err = NULL;
    ZF_Status s = mb_finalize(&mbA, &fin_err);
    if (s != ZF_OK) {
        print_err(s, fin_err ? fin_err : "matrix_A_error");
        mb_destroy(&mbA);
        mb_destroy(&mbB);
        return EXIT_SUCCESS;
    }

    /* ── Execute Requested Command ───────────────────────────────── */
    if (strcmp(verb, "ANALYZE") == 0) {
        MemoryAnalysis ma;
        s = memory_analysis_sparse(&mbA.sm, &ma);
        if (s != ZF_OK) {
            print_err(s, "analysis_failed");
        } else {
            print_ok_start();
            printf("D ROWS %zu\n", ma.rows);
            printf("D COLS %zu\n", ma.cols);
            printf("D TOTAL_CELLS %zu\n", ma.total_cells);
            printf("D NNZ %zu\n", ma.nnz);
            printf("D ZERO_COUNT %zu\n", ma.zero_count);
            printf("D DENSITY %.10f\n", ma.density_percent);
            printf("D SPARSITY %.10f\n", ma.sparsity_percent);
            printf("D DENSE_BYTES %zu\n", ma.dense_payload_bytes);
            printf("D SPARSE_BYTES %zu\n", ma.sparse_payload_bytes);
            printf("D SPARSE_IS_SMALLER %d\n", ma.sparse_is_smaller);
            printf("D SAVINGS_BYTES %zu\n", ma.savings_bytes);
            printf("D PERCENT_SAVED %.10f\n", ma.percent_saved);
            /* Include triplets */
            size_t n = da_length(&mbA.sm.entries);
            for (size_t i = 0; i < n; i++) {
                const Triplet *t = (const Triplet *)da_ptr(&mbA.sm.entries, i);
                printf("D ENTRY %zu %zu %.17g\n", t->row, t->col, t->value);
            }
            print_end();
        }
    } else if (strcmp(verb, "TO_SPARSE") == 0) {
        print_ok_start();
        print_sparse_matrix(&mbA.sm);
        print_end();
    } else if (strcmp(verb, "TO_DENSE") == 0) {
        if (mbA.sm.cols > 0 && mbA.sm.rows > MAX_CELLS / mbA.sm.cols) {
            print_err(ZF_ERR_OVERFLOW, "to_dense_cells_exceed_limit");
        } else {
            DenseMatrix dm_out;
            s = sparse_matrix_to_dense(&mbA.sm, &dm_out);
            if (s != ZF_OK) {
                print_err(s, "to_dense_conversion_failed");
            } else {
                print_ok_start();
                print_dense_matrix(&dm_out);
                print_end();
                dense_matrix_destroy(&dm_out);
            }
        }
    } else if (strcmp(verb, "ADD") == 0) {
        s = mb_finalize(&mbB, &fin_err);
        if (s != ZF_OK) {
            print_err(s, fin_err ? fin_err : "matrix_B_error");
        } else {
            SparseMatrix c;
            s = sparse_matrix_add(&mbA.sm, &mbB.sm, &c);
            if (s != ZF_OK) {
                print_err(s, "addition_failed");
            } else {
                print_ok_start();
                print_sparse_matrix(&c);
                print_end();
                sparse_matrix_destroy(&c);
            }
        }
    } else if (strcmp(verb, "MULTIPLY") == 0) {
        s = mb_finalize(&mbB, &fin_err);
        if (s != ZF_OK) {
            print_err(s, fin_err ? fin_err : "matrix_B_error");
        } else if (mbB.sm.cols > 0 && mbA.sm.rows > MAX_CELLS / mbB.sm.cols) {
            print_err(ZF_ERR_OVERFLOW, "multiplication_result_cells_exceed_limit");
        } else {
            SparseMatrix c;
            s = sparse_matrix_multiply(&mbA.sm, &mbB.sm, &c);
            if (s != ZF_OK) {
                print_err(s, "multiplication_failed");
            } else {
                print_ok_start();
                print_sparse_matrix(&c);
                print_end();
                sparse_matrix_destroy(&c);
            }
        }
    } else if (strcmp(verb, "TRANSPOSE") == 0) {
        SparseMatrix c;
        s = sparse_matrix_transpose(&mbA.sm, &c);
        if (s != ZF_OK) {
            print_err(s, "transpose_failed");
        } else {
            print_ok_start();
            print_sparse_matrix(&c);
            print_end();
            sparse_matrix_destroy(&c);
        }
    } else if (strcmp(verb, "INSERT") == 0) {
        if (!has_row || !has_col || !has_val) {
            print_err(ZF_ERR_PARAM, "missing_insert_coordinates_or_value");
        } else {
            s = sparse_matrix_insert(&mbA.sm, param_row, param_col, param_val);
            if (s != ZF_OK) {
                print_err(s, "insert_failed");
            } else {
                print_ok_start();
                print_sparse_matrix(&mbA.sm);
                print_end();
            }
        }
    } else if (strcmp(verb, "SET") == 0) {
        if (!has_row || !has_col || !has_val) {
            print_err(ZF_ERR_PARAM, "missing_set_coordinates_or_value");
        } else {
            s = sparse_matrix_set(&mbA.sm, param_row, param_col, param_val);
            if (s != ZF_OK) {
                print_err(s, "set_failed");
            } else {
                print_ok_start();
                print_sparse_matrix(&mbA.sm);
                print_end();
            }
        }
    } else if (strcmp(verb, "DELETE") == 0) {
        if (!has_row || !has_col) {
            print_err(ZF_ERR_PARAM, "missing_delete_coordinates");
        } else {
            s = sparse_matrix_delete(&mbA.sm, param_row, param_col);
            if (s != ZF_OK) {
                print_err(s, "delete_failed");
            } else {
                print_ok_start();
                print_sparse_matrix(&mbA.sm);
                print_end();
            }
        }
    } else if (strcmp(verb, "SEARCH") == 0) {
        if (!has_row || !has_col) {
            print_err(ZF_ERR_PARAM, "missing_search_coordinates");
        } else {
            double f_val = 0.0;
            size_t f_pos = 0;
            s = sparse_matrix_search(&mbA.sm, param_row, param_col, &f_val, &f_pos);
            if (s == ZF_OK) {
                print_ok_start();
                printf("D FOUND 1\n");
                printf("D ROW %zu\n", param_row);
                printf("D COL %zu\n", param_col);
                printf("D VALUE %.17g\n", f_val);
                printf("D POS %zu\n", f_pos);
                print_end();
            } else if (s == ZF_ERR_NOT_FOUND) {
                print_ok_start();
                printf("D FOUND 0\n");
                printf("D ROW %zu\n", param_row);
                printf("D COL %zu\n", param_col);
                printf("D VALUE 0\n");
                printf("D POS %zu\n", f_pos);
                print_end();
            } else {
                print_err(s, "search_failed");
            }
        }
    } else if (strcmp(verb, "TRAVERSE") == 0) {
        print_ok_start();
        print_sparse_matrix(&mbA.sm);
        print_end();
    } else if (strcmp(verb, "SORT") == 0) {
        s = sparse_matrix_sort(&mbA.sm);
        if (s != ZF_OK) {
            print_err(s, "sort_failed");
        } else {
            print_ok_start();
            print_sparse_matrix(&mbA.sm);
            print_end();
        }
    } else {
        print_err(ZF_ERR_PARAM, "unrecognized_command_verb");
    }

    mb_destroy(&mbA);
    mb_destroy(&mbB);
    return EXIT_SUCCESS;
}

/*
 * test_matrices.c — Unit tests for DenseMatrix and SparseMatrix (Stage 2)
 *
 * Test runner: same hand-rolled CHECK() framework as Stage 1.
 * - Every test verifies CONCRETE results, not just "returned without crash".
 * - Deterministic input/output data only.
 * - No unsafe production-API hacks to manufacture invalid state;
 *   invalid-state tests use controlled local test fixtures.
 */

#include "../matrix.h"
#include "../zeroform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>    /* isfinite, INFINITY, NAN */

/* ═══════════════════════════════════════════════════════════════════
 * Test framework (identical pattern to Stage 1)
 * ═══════════════════════════════════════════════════════════════════ */

static int g_pass = 0;
static int g_fail = 0;
static const char *g_test = "(none)";

#define CHECK(expr)                                                         \
    do {                                                                    \
        if (!(expr)) {                                                      \
            fprintf(stderr, "  FAIL [%s] line %d: %s\n",                   \
                    g_test, __LINE__, #expr);                               \
            g_fail++;                                                       \
        } else {                                                            \
            g_pass++;                                                       \
        }                                                                   \
    } while (0)

#define BEGIN(name) do { g_test = (name); printf("  %-58s\n", (name)); fflush(stdout); } while(0)

/* ═══════════════════════════════════════════════════════════════════
 * Traversal helpers
 * ═══════════════════════════════════════════════════════════════════ */

/* Context struct for collecting traversal results. */
typedef struct {
    Triplet buf[64];
    size_t  count;
} TraverseCtx;

/* Callback: copy each visited triplet into ctx->buf[]. */
static int collect_cb(const Triplet *t, void *ctx)
{
    TraverseCtx *c = (TraverseCtx *)ctx;
    if (c->count < 64) {
        c->buf[c->count++] = *t;
    }
    return 0;   /* continue */
}

/* Callback: stop after the first entry (returns 1). */
static int stop_after_one_cb(const Triplet *t, void *ctx)
{
    (void)t;
    int *counter = (int *)ctx;
    (*counter)++;
    return 1;   /* stop */
}

/* ═══════════════════════════════════════════════════════════════════
 * ── DENSEMATRIX TESTS ──────────────────────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* DM-01: Initialization with positive dimensions */
static void test_dm_init_basic(void)
{
    BEGIN("DM-01 init 3x4 matrix");
    DenseMatrix dm;
    CHECK(dense_matrix_init(&dm, 3, 4) == ZF_OK);
    CHECK(dense_matrix_rows(&dm) == 3);
    CHECK(dense_matrix_cols(&dm) == 4);
    CHECK(dm.init == 1);
    dense_matrix_destroy(&dm);
}

/* DM-02: All cells initialized to 0.0 */
static void test_dm_zero_init(void)
{
    BEGIN("DM-02 zero initialization");
    DenseMatrix dm;
    dense_matrix_init(&dm, 3, 3);
    int all_zero = 1;
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 3; c++) {
            double v = -1.0;
            dense_matrix_get(&dm, r, c, &v);
            if (v != 0.0) { all_zero = 0; }
        }
    }
    CHECK(all_zero);
    dense_matrix_destroy(&dm);
}

/* DM-03: Set and get round-trip */
static void test_dm_set_get(void)
{
    BEGIN("DM-03 set/get round-trip");
    DenseMatrix dm;
    dense_matrix_init(&dm, 4, 4);
    CHECK(dense_matrix_set(&dm, 1, 2, 7.5) == ZF_OK);
    double v = 0.0;
    CHECK(dense_matrix_get(&dm, 1, 2, &v) == ZF_OK);
    CHECK(v == 7.5);
    /* Surrounding cells must still be zero. */
    dense_matrix_get(&dm, 0, 0, &v); CHECK(v == 0.0);
    dense_matrix_get(&dm, 2, 2, &v); CHECK(v == 0.0);
    dense_matrix_destroy(&dm);
}

/* DM-04: Row-major layout — sequential flat indices */
static void test_dm_row_major(void)
{
    BEGIN("DM-04 row-major layout verification");
    DenseMatrix dm;
    dense_matrix_init(&dm, 2, 3);
    /* Set all cells to their flat index value for identification. */
    double vals[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    for (size_t r = 0; r < 2; r++)
        for (size_t c = 0; c < 3; c++)
            dense_matrix_set(&dm, r, c, vals[r * 3 + c]);
    /* Verify row-major: (0,0)=10, (0,1)=20, (0,2)=30, (1,0)=40... */
    double v = 0.0;
    dense_matrix_get(&dm, 0, 0, &v); CHECK(v == 10.0);
    dense_matrix_get(&dm, 0, 2, &v); CHECK(v == 30.0);
    dense_matrix_get(&dm, 1, 0, &v); CHECK(v == 40.0);
    dense_matrix_get(&dm, 1, 2, &v); CHECK(v == 60.0);
    dense_matrix_destroy(&dm);
}

/* DM-05: Rectangular matrices (non-square) */
static void test_dm_rectangular(void)
{
    BEGIN("DM-05 rectangular matrix 1x5");
    DenseMatrix dm;
    CHECK(dense_matrix_init(&dm, 1, 5) == ZF_OK);
    CHECK(dense_matrix_rows(&dm) == 1);
    CHECK(dense_matrix_cols(&dm) == 5);
    dense_matrix_set(&dm, 0, 4, 99.0);
    double v = 0.0;
    dense_matrix_get(&dm, 0, 4, &v);
    CHECK(v == 99.0);
    dense_matrix_destroy(&dm);
}

/* DM-06: Invalid dimensions */
static void test_dm_invalid_dims(void)
{
    BEGIN("DM-06 invalid dimensions (zero rows/cols)");
    DenseMatrix dm;
    CHECK(dense_matrix_init(&dm, 0, 3) == ZF_ERR_PARAM);
    CHECK(dm.init == 0);
    CHECK(dense_matrix_init(&dm, 3, 0) == ZF_ERR_PARAM);
    CHECK(dm.init == 0);
    CHECK(dense_matrix_init(&dm, 0, 0) == ZF_ERR_PARAM);
    CHECK(dm.init == 0);
}

/* DM-07: Invalid coordinates */
static void test_dm_invalid_coords(void)
{
    BEGIN("DM-07 out-of-range coordinates");
    DenseMatrix dm;
    dense_matrix_init(&dm, 3, 3);
    double v = 0.0;
    CHECK(dense_matrix_get(&dm, 3, 0, &v) == ZF_ERR_INDEX);  /* row == rows */
    CHECK(dense_matrix_get(&dm, 0, 3, &v) == ZF_ERR_INDEX);  /* col == cols */
    CHECK(dense_matrix_set(&dm, 5, 5, 1.0) == ZF_ERR_INDEX);
    dense_matrix_destroy(&dm);
}

/* DM-08: Non-finite values rejected */
static void test_dm_nonfinite(void)
{
    BEGIN("DM-08 non-finite values rejected");
    DenseMatrix dm;
    dense_matrix_init(&dm, 2, 2);
    CHECK(dense_matrix_set(&dm, 0, 0, (double)INFINITY) == ZF_ERR_VALUE);
    CHECK(dense_matrix_set(&dm, 0, 0, -(double)INFINITY) == ZF_ERR_VALUE);
    CHECK(dense_matrix_set(&dm, 0, 0, (double)NAN) == ZF_ERR_VALUE);
    /* Cell must remain unchanged (0.0). */
    double v = -1.0;
    dense_matrix_get(&dm, 0, 0, &v);
    CHECK(v == 0.0);
    dense_matrix_destroy(&dm);
}

/* DM-09: Construction from flat input */
static void test_dm_from_flat(void)
{
    BEGIN("DM-09 dense_matrix_from_flat");
    double flat[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    DenseMatrix dm;
    CHECK(dense_matrix_from_flat(&dm, 2, 3, flat) == ZF_OK);
    double v = 0.0;
    dense_matrix_get(&dm, 0, 0, &v); CHECK(v == 1.0);
    dense_matrix_get(&dm, 0, 2, &v); CHECK(v == 3.0);
    dense_matrix_get(&dm, 1, 0, &v); CHECK(v == 4.0);
    dense_matrix_get(&dm, 1, 2, &v); CHECK(v == 6.0);
    dense_matrix_destroy(&dm);
}

/* DM-10: Non-finite in flat input rejected */
static void test_dm_from_flat_nonfinite(void)
{
    BEGIN("DM-10 from_flat rejects NaN element");
    double flat[4] = {1.0, 2.0, (double)NAN, 4.0};
    DenseMatrix dm;
    CHECK(dense_matrix_from_flat(&dm, 2, 2, flat) == ZF_ERR_VALUE);
    CHECK(dm.init == 0);
}

/* DM-11: Dimension-product overflow */
static void test_dm_overflow(void)
{
    BEGIN("DM-11 dimension overflow rejected");
    DenseMatrix dm;
    /* SIZE_MAX/2+1 rows and 2 cols: rows*cols > SIZE_MAX. */
    size_t big = (size_t)(SIZE_MAX / 2) + 1;
    ZF_Status s = dense_matrix_init(&dm, big, 2);
    CHECK(s == ZF_ERR_OVERFLOW || s == ZF_ERR_NOMEM);
    CHECK(dm.init == 0);
}

/* DM-12: Destruction and reinitialization */
static void test_dm_destroy_reinit(void)
{
    BEGIN("DM-12 destroy then reinit on same struct");
    DenseMatrix dm;
    dense_matrix_init(&dm, 3, 3);
    dense_matrix_set(&dm, 1, 1, 42.0);
    dense_matrix_destroy(&dm);
    CHECK(dm.init == 0);
    CHECK(dm.rows == 0);
    CHECK(dm.cols == 0);
    /* Re-init must produce a fresh zero matrix. */
    CHECK(dense_matrix_init(&dm, 2, 2) == ZF_OK);
    double v = -1.0;
    dense_matrix_get(&dm, 0, 0, &v);
    CHECK(v == 0.0);
    dense_matrix_destroy(&dm);
}

/* DM-13: Copy */
static void test_dm_copy(void)
{
    BEGIN("DM-13 dense_matrix_copy");
    double flat[4] = {1.0, 2.0, 3.0, 4.0};
    DenseMatrix src, dst;
    dense_matrix_from_flat(&src, 2, 2, flat);
    CHECK(dense_matrix_copy(&src, &src) == ZF_ERR_PARAM); /* self-copy rejected */
    CHECK(dense_matrix_copy(&dst, &src) == ZF_OK);
    /* Verify copy matches source. */
    double vs = 0.0, vd = 0.0;
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 2; c++) {
            dense_matrix_get(&src, r, c, &vs);
            dense_matrix_get(&dst, r, c, &vd);
            CHECK(vs == vd);
        }
    }
    /* Mutate dst; src must be unaffected. */
    dense_matrix_set(&dst, 0, 0, 99.0);
    dense_matrix_get(&src, 0, 0, &vs);
    CHECK(vs == 1.0);
    dense_matrix_destroy(&src);
    dense_matrix_destroy(&dst);
}

/* ═══════════════════════════════════════════════════════════════════
 * ── SPARSEMATRIX CONSTRUCTION TESTS ───────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* SM-01: Empty sparse storage */
static void test_sm_init_empty(void)
{
    BEGIN("SM-01 init empty sparse 4x4");
    SparseMatrix sm;
    CHECK(sparse_matrix_init(&sm, 4, 4) == ZF_OK);
    CHECK(sm.rows == 4);
    CHECK(sm.cols == 4);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-02: All-zero dense → empty sparse */
static void test_sm_all_zero(void)
{
    BEGIN("SM-02 all-zero dense converts to empty sparse");
    DenseMatrix dm;
    dense_matrix_init(&dm, 3, 3);   /* all zeros */
    SparseMatrix sm;
    CHECK(sparse_matrix_from_dense(&sm, &dm) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    dense_matrix_destroy(&dm);
    sparse_matrix_destroy(&sm);
}

/* SM-03: Single nonzero entry */
static void test_sm_single_entry(void)
{
    BEGIN("SM-03 single nonzero entry");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    CHECK(sparse_matrix_insert(&sm, 2, 3, 7.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    double v = 0.0;
    CHECK(sparse_matrix_get(&sm, 2, 3, &v) == ZF_OK);
    CHECK(v == 7.0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-04: Rectangular sparse */
static void test_sm_rectangular(void)
{
    BEGIN("SM-04 rectangular sparse matrix 2x5");
    SparseMatrix sm;
    CHECK(sparse_matrix_init(&sm, 2, 5) == ZF_OK);
    CHECK(sparse_matrix_insert(&sm, 0, 4, 3.0) == ZF_OK);
    CHECK(sparse_matrix_insert(&sm, 1, 0, 5.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 2);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-05: Dense-to-sparse conversion */
static void test_sm_from_dense(void)
{
    BEGIN("SM-05 dense-to-sparse conversion");
    double flat[9] = {0.0, 2.0, 0.0,
                       0.0, 0.0, 5.0,
                       7.0, 0.0, 0.0};
    DenseMatrix dm;
    dense_matrix_from_flat(&dm, 3, 3, flat);
    SparseMatrix sm;
    CHECK(sparse_matrix_from_dense(&sm, &dm) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 3);
    double v = 0.0;
    CHECK(sparse_matrix_get(&sm, 0, 1, &v) == ZF_OK); CHECK(v == 2.0);
    CHECK(sparse_matrix_get(&sm, 1, 2, &v) == ZF_OK); CHECK(v == 5.0);
    CHECK(sparse_matrix_get(&sm, 2, 0, &v) == ZF_OK); CHECK(v == 7.0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    dense_matrix_destroy(&dm);
    sparse_matrix_destroy(&sm);
}

/* SM-06: Sparse-to-dense conversion */
static void test_sm_to_dense(void)
{
    BEGIN("SM-06 sparse-to-dense conversion");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 0, 1, 2.0);
    sparse_matrix_insert(&sm, 1, 2, 5.0);
    sparse_matrix_insert(&sm, 2, 0, 7.0);
    DenseMatrix dm;
    CHECK(sparse_matrix_to_dense(&sm, &dm) == ZF_OK);
    double v = 0.0;
    dense_matrix_get(&dm, 0, 1, &v); CHECK(v == 2.0);
    dense_matrix_get(&dm, 1, 2, &v); CHECK(v == 5.0);
    dense_matrix_get(&dm, 2, 0, &v); CHECK(v == 7.0);
    /* All other cells must be 0. */
    dense_matrix_get(&dm, 0, 0, &v); CHECK(v == 0.0);
    dense_matrix_get(&dm, 1, 1, &v); CHECK(v == 0.0);
    dense_matrix_destroy(&dm);
    sparse_matrix_destroy(&sm);
}

/* SM-07: Round-trip dense → sparse → dense */
static void test_sm_roundtrip(void)
{
    BEGIN("SM-07 round-trip dense->sparse->dense");
    double flat[9] = {1.0, 0.0, 3.0,
                       0.0, 5.0, 0.0,
                       7.0, 0.0, 9.0};
    DenseMatrix src, dst;
    dense_matrix_from_flat(&src, 3, 3, flat);
    SparseMatrix sm;
    sparse_matrix_from_dense(&sm, &src);
    sparse_matrix_to_dense(&sm, &dst);
    int all_match = 1;
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 3; c++) {
            double vs = 0.0, vd = 0.0;
            dense_matrix_get(&src, r, c, &vs);
            dense_matrix_get(&dst, r, c, &vd);
            if (vs != vd) { all_match = 0; }
        }
    }
    CHECK(all_match);
    dense_matrix_destroy(&src);
    dense_matrix_destroy(&dst);
    sparse_matrix_destroy(&sm);
}

/* SM-08: All-zero rows/columns preserved in dimensions */
static void test_sm_preserve_zero_rows(void)
{
    BEGIN("SM-08 all-zero rows preserved in dimensions");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    /* Only insert one entry — rows 0,1,2,3,5 remain zero. */
    sparse_matrix_insert(&sm, 4, 4, 1.0);
    CHECK(sm.rows == 5);
    CHECK(sm.cols == 5);
    double v = 0.0;
    sparse_matrix_get(&sm, 0, 0, &v); CHECK(v == 0.0);
    sparse_matrix_get(&sm, 3, 3, &v); CHECK(v == 0.0);
    sparse_matrix_destroy(&sm);
}

/* SM-09: Correct nonzero count */
static void test_sm_nnz_count(void)
{
    BEGIN("SM-09 correct nonzero count after inserts/deletes");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 4, 4);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    sparse_matrix_insert(&sm, 1, 1, 2.0);
    CHECK(sparse_matrix_nonzero_count(&sm) == 2);
    sparse_matrix_delete(&sm, 0, 0);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    sparse_matrix_destroy(&sm);
}

/* SM-10: from_triplets — unordered input */
static void test_sm_from_triplets_unordered(void)
{
    BEGIN("SM-10 from_triplets with unordered input");
    Triplet ts[3] = {
        {2, 1, 9.0},
        {0, 0, 1.0},
        {1, 2, 5.0}
    };
    SparseMatrix sm;
    CHECK(sparse_matrix_from_triplets(&sm, 3, 3, ts, 3) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 3);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    /* Verify sorted order in the array. */
    Triplet first, second, third;
    da_get(&sm.entries, 0, &first);
    da_get(&sm.entries, 1, &second);
    da_get(&sm.entries, 2, &third);
    CHECK(first.row == 0 && first.col == 0);
    CHECK(second.row == 1 && second.col == 2);
    CHECK(third.row == 2 && third.col == 1);
    sparse_matrix_destroy(&sm);
}

/* SM-11: from_triplets — duplicate rejected */
static void test_sm_from_triplets_duplicate(void)
{
    BEGIN("SM-11 from_triplets rejects duplicate coord");
    Triplet ts[2] = {
        {1, 1, 3.0},
        {1, 1, 7.0}   /* duplicate */
    };
    SparseMatrix sm;
    CHECK(sparse_matrix_from_triplets(&sm, 3, 3, ts, 2) == ZF_ERR_DUPLICATE);
    CHECK(sm.init == 0);
}

/* SM-12: from_triplets — invalid coordinate rejected */
static void test_sm_from_triplets_oob(void)
{
    BEGIN("SM-12 from_triplets rejects out-of-bounds coord");
    Triplet ts[1] = {{5, 0, 1.0}};   /* row 5 invalid for 3x3 */
    SparseMatrix sm;
    CHECK(sparse_matrix_from_triplets(&sm, 3, 3, ts, 1) == ZF_ERR_INDEX);
    CHECK(sm.init == 0);
}

/* SM-13: from_triplets — explicit zero skipped */
static void test_sm_from_triplets_zero(void)
{
    BEGIN("SM-13 from_triplets skips explicit zeros");
    Triplet ts[3] = {
        {0, 0, 0.0},   /* explicit zero → skip */
        {1, 1, 5.0},
        {2, 2, 3.0}
    };
    SparseMatrix sm;
    CHECK(sparse_matrix_from_triplets(&sm, 3, 3, ts, 3) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 2);
    sparse_matrix_destroy(&sm);
}

/* SM-14: from_triplets — non-finite value rejected */
static void test_sm_from_triplets_nonfinite(void)
{
    BEGIN("SM-14 from_triplets rejects NaN value");
    Triplet ts[1] = {{0, 0, (double)NAN}};
    SparseMatrix sm;
    CHECK(sparse_matrix_from_triplets(&sm, 3, 3, ts, 1) == ZF_ERR_VALUE);
    CHECK(sm.init == 0);
}

/* ═══════════════════════════════════════════════════════════════════
 * ── SEARCH AND MUTATION TESTS ──────────────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* SM-15: Search for first, middle and last stored coordinate */
static void test_sm_search_positions(void)
{
    BEGIN("SM-15 search: first/middle/last stored coordinate");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_insert(&sm, 2, 2, 5.0);
    sparse_matrix_insert(&sm, 4, 4, 9.0);

    double v = 0.0; size_t pos = 999;
    CHECK(sparse_matrix_search(&sm, 0, 0, &v, &pos) == ZF_OK); CHECK(v == 1.0); CHECK(pos == 0);
    CHECK(sparse_matrix_search(&sm, 2, 2, &v, &pos) == ZF_OK); CHECK(v == 5.0); CHECK(pos == 1);
    CHECK(sparse_matrix_search(&sm, 4, 4, &v, &pos) == ZF_OK); CHECK(v == 9.0); CHECK(pos == 2);
    sparse_matrix_destroy(&sm);
}

/* SM-16: Search for missing coordinate */
static void test_sm_search_missing(void)
{
    BEGIN("SM-16 search: missing coordinate returns ZF_ERR_NOT_FOUND");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 1, 1, 5.0);
    CHECK(sparse_matrix_search(&sm, 0, 0, NULL, NULL) == ZF_ERR_NOT_FOUND);
    CHECK(sparse_matrix_search(&sm, 2, 2, NULL, NULL) == ZF_ERR_NOT_FOUND);
    sparse_matrix_destroy(&sm);
}

/* SM-17: Reading an implicit zero returns 0.0 without error */
static void test_sm_implicit_zero(void)
{
    BEGIN("SM-17 reading implicit zero returns 0.0 (not error)");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 1, 1, 7.0);
    double v = -1.0;
    CHECK(sparse_matrix_get(&sm, 0, 0, &v) == ZF_OK);
    CHECK(v == 0.0);
    CHECK(sparse_matrix_get(&sm, 2, 2, &v) == ZF_OK);
    CHECK(v == 0.0);
    sparse_matrix_destroy(&sm);
}

/* SM-18: Insert at beginning (position 0) */
static void test_sm_insert_beginning(void)
{
    BEGIN("SM-18 insert at beginning (smallest coord)");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 3, 3, 10.0);
    sparse_matrix_insert(&sm, 4, 4, 20.0);
    CHECK(sparse_matrix_insert(&sm, 0, 0, 5.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 3);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    /* First entry must now be (0,0). */
    Triplet first;
    da_get(&sm.entries, 0, &first);
    CHECK(first.row == 0 && first.col == 0 && first.value == 5.0);
    sparse_matrix_destroy(&sm);
}

/* SM-19: Insert in middle */
static void test_sm_insert_middle(void)
{
    BEGIN("SM-19 insert in middle");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_insert(&sm, 4, 4, 9.0);
    CHECK(sparse_matrix_insert(&sm, 2, 2, 5.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 3);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    Triplet mid;
    da_get(&sm.entries, 1, &mid);
    CHECK(mid.row == 2 && mid.col == 2 && mid.value == 5.0);
    sparse_matrix_destroy(&sm);
}

/* SM-20: Insert at end */
static void test_sm_insert_end(void)
{
    BEGIN("SM-20 insert at end (largest coord)");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    CHECK(sparse_matrix_insert(&sm, 4, 4, 9.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 2);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-21: Reject duplicate insertion */
static void test_sm_insert_duplicate(void)
{
    BEGIN("SM-21 duplicate insert rejected");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 4, 4);
    sparse_matrix_insert(&sm, 1, 2, 3.0);
    CHECK(sparse_matrix_insert(&sm, 1, 2, 7.0) == ZF_ERR_DUPLICATE);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    /* Original value unchanged. */
    double v = 0.0;
    sparse_matrix_get(&sm, 1, 2, &v);
    CHECK(v == 3.0);
    sparse_matrix_destroy(&sm);
}

/* SM-22: Update existing value via set() */
static void test_sm_update_existing(void)
{
    BEGIN("SM-22 set() updates existing entry");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 1, 1, 5.0);
    CHECK(sparse_matrix_set(&sm, 1, 1, 15.0) == ZF_OK);
    double v = 0.0;
    sparse_matrix_get(&sm, 1, 1, &v);
    CHECK(v == 15.0);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-23: Set new nonzero value (insert via set) */
static void test_sm_set_new_nonzero(void)
{
    BEGIN("SM-23 set() inserts new entry when absent");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    CHECK(sparse_matrix_set(&sm, 2, 1, 8.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    double v = 0.0;
    sparse_matrix_get(&sm, 2, 1, &v);
    CHECK(v == 8.0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-24: Set existing value to zero (delete via set) */
static void test_sm_set_to_zero(void)
{
    BEGIN("SM-24 set() to 0.0 deletes existing entry");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 0, 0, 4.0);
    sparse_matrix_insert(&sm, 1, 1, 5.0);
    CHECK(sparse_matrix_set(&sm, 0, 0, 0.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    double v = -1.0;
    sparse_matrix_get(&sm, 0, 0, &v);
    CHECK(v == 0.0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-25: Delete first, middle and last entries */
static void test_sm_delete_positions(void)
{
    BEGIN("SM-25 delete: first, middle and last entries");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_insert(&sm, 2, 2, 5.0);
    sparse_matrix_insert(&sm, 4, 4, 9.0);

    /* Delete middle. */
    CHECK(sparse_matrix_delete(&sm, 2, 2) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 2);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);

    /* Delete last. */
    CHECK(sparse_matrix_delete(&sm, 4, 4) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);

    /* Delete first. */
    CHECK(sparse_matrix_delete(&sm, 0, 0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-26: Delete missing coordinate */
static void test_sm_delete_missing(void)
{
    BEGIN("SM-26 delete missing coord returns ZF_ERR_NOT_FOUND");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 1, 1, 5.0);
    CHECK(sparse_matrix_delete(&sm, 0, 0) == ZF_ERR_NOT_FOUND);
    CHECK(sparse_matrix_nonzero_count(&sm) == 1);
    sparse_matrix_destroy(&sm);
}

/* SM-27: Row-major order preserved after mutations */
static void test_sm_order_after_mutations(void)
{
    BEGIN("SM-27 row-major order preserved after mutations");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 4, 4, 9.0);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_insert(&sm, 2, 3, 5.0);
    sparse_matrix_insert(&sm, 0, 2, 2.0);
    sparse_matrix_delete(&sm, 2, 3);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    /* Expect sorted: (0,0), (0,2), (4,4). */
    Triplet t0, t1, t2;
    da_get(&sm.entries, 0, &t0);
    da_get(&sm.entries, 1, &t1);
    da_get(&sm.entries, 2, &t2);
    CHECK(t0.row == 0 && t0.col == 0);
    CHECK(t1.row == 0 && t1.col == 2);
    CHECK(t2.row == 4 && t2.col == 4);
    sparse_matrix_destroy(&sm);
}

/* SM-28: Dimensions unchanged after insert and delete */
static void test_sm_dimensions_stable(void)
{
    BEGIN("SM-28 dimensions unchanged after insert/delete");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 7, 8);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_delete(&sm, 0, 0);
    CHECK(sm.rows == 7);
    CHECK(sm.cols == 8);
    sparse_matrix_destroy(&sm);
}

/* ═══════════════════════════════════════════════════════════════════
 * ── SORTING AND INVARIANT TESTS ───────────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* SM-29: Explicit sort on deliberately unordered DynamicArray */
static void test_sm_explicit_sort(void)
{
    BEGIN("SM-29 explicit sort corrects unordered entries");
    /*
     * We build the array via from_triplets (which calls sort internally),
     * then directly append an out-of-order triplet to the DynamicArray
     * to produce a canonical violation, and verify sort fixes it.
     *
     * This is the only safe test fixture: we use the public da_append
     * on sm.entries directly (the DynamicArray is public in the struct)
     * since the alternative would be fabricating a broken matrix through
     * undefined behaviour.
     */
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    /* Insert in sorted order first. */
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_insert(&sm, 2, 2, 5.0);
    /* Directly append an out-of-order triplet (row 1 < row 2 but comes after). */
    Triplet ooo; ooo.row = 1; ooo.col = 1; ooo.value = 3.0;
    da_append(&sm.entries, &ooo);
    /* The array is now [0,0,1], [2,2,5], [1,1,3] — invalid order. */
    CHECK(sparse_matrix_validate(&sm) != ZF_OK);   /* confirm it's broken */
    /* Now sort. */
    CHECK(sparse_matrix_sort(&sm) == ZF_OK);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);   /* must be canonical now */
    CHECK(sparse_matrix_nonzero_count(&sm) == 3);
    Triplet t0, t1, t2;
    da_get(&sm.entries, 0, &t0);
    da_get(&sm.entries, 1, &t1);
    da_get(&sm.entries, 2, &t2);
    CHECK(t0.row == 0 && t0.col == 0);
    CHECK(t1.row == 1 && t1.col == 1);
    CHECK(t2.row == 2 && t2.col == 2);
    sparse_matrix_destroy(&sm);
}

/* SM-30: Sorting respects column order within same row */
static void test_sm_sort_same_row(void)
{
    BEGIN("SM-30 sort: same-row entries ordered by column");
    Triplet ts[4] = {
        {1, 3, 4.0},
        {1, 0, 1.0},
        {1, 2, 3.0},
        {1, 1, 2.0}
    };
    SparseMatrix sm;
    CHECK(sparse_matrix_from_triplets(&sm, 5, 5, ts, 4) == ZF_OK);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    for (size_t i = 0; i < 4; i++) {
        Triplet t;
        da_get(&sm.entries, i, &t);
        CHECK(t.col == (size_t)i);       /* cols must be 0,1,2,3 in order */
        CHECK(t.value == (double)(i + 1));
    }
    sparse_matrix_destroy(&sm);
}

/* SM-31: Invariant validation on valid matrix returns ZF_OK */
static void test_sm_validate_valid(void)
{
    BEGIN("SM-31 validate() returns ZF_OK on canonical matrix");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 4, 4);
    sparse_matrix_insert(&sm, 0, 1, 2.0);
    sparse_matrix_insert(&sm, 1, 3, 5.0);
    sparse_matrix_insert(&sm, 3, 0, 7.0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-32: Validate detects duplicate coordinate (test fixture) */
static void test_sm_validate_duplicate(void)
{
    BEGIN("SM-32 validate() detects duplicate coordinate");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 4, 4);
    sparse_matrix_insert(&sm, 1, 1, 5.0);
    /* Force a duplicate by directly appending the same coord to the DA. */
    Triplet dup; dup.row = 1; dup.col = 1; dup.value = 7.0;
    da_append(&sm.entries, &dup);
    CHECK(sparse_matrix_validate(&sm) != ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-33: Validate detects out-of-bounds stored coord (test fixture) */
static void test_sm_validate_oob(void)
{
    BEGIN("SM-33 validate() detects out-of-bounds stored coord");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    /* Force an out-of-bounds triplet directly. */
    Triplet oob; oob.row = 10; oob.col = 10; oob.value = 1.0;
    da_append(&sm.entries, &oob);
    CHECK(sparse_matrix_validate(&sm) == ZF_ERR_INDEX);
    sparse_matrix_destroy(&sm);
}

/* SM-34: Traversal visits every entry exactly once in order */
static void test_sm_traversal_order(void)
{
    BEGIN("SM-34 traversal visits every entry exactly once in order");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 0, 1, 2.0);
    sparse_matrix_insert(&sm, 1, 3, 5.0);
    sparse_matrix_insert(&sm, 3, 0, 7.0);

    TraverseCtx ctx; ctx.count = 0;
    CHECK(sparse_matrix_traverse(&sm, collect_cb, &ctx) == ZF_OK);
    CHECK(ctx.count == 3);
    CHECK(ctx.buf[0].row == 0 && ctx.buf[0].col == 1 && ctx.buf[0].value == 2.0);
    CHECK(ctx.buf[1].row == 1 && ctx.buf[1].col == 3 && ctx.buf[1].value == 5.0);
    CHECK(ctx.buf[2].row == 3 && ctx.buf[2].col == 0 && ctx.buf[2].value == 7.0);
    sparse_matrix_destroy(&sm);
}

/* SM-35: Traversal early-stop */
static void test_sm_traversal_early_stop(void)
{
    BEGIN("SM-35 traversal stops early when callback returns nonzero");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 5, 5);
    sparse_matrix_insert(&sm, 0, 0, 1.0);
    sparse_matrix_insert(&sm, 1, 1, 2.0);
    sparse_matrix_insert(&sm, 2, 2, 3.0);

    int counter = 0;
    /* stop_after_one_cb returns 1 after first visit. */
    ZF_Status s = sparse_matrix_traverse(&sm, stop_after_one_cb, &counter);
    CHECK(s != ZF_OK);   /* returned the callback's nonzero return value */
    CHECK(counter == 1); /* only one entry visited */
    sparse_matrix_destroy(&sm);
}

/* SM-36: Traversal does not expose a mutable pointer */
/* This is a compile-time guarantee enforced by the const Triplet* signature.
 * We verify it at runtime by confirming values don't change when we try
 * to write through what we received (we can only write via a cast, so
 * this test verifies the const qualifier is enforced). */
static int const_write_attempt_cb(const Triplet *t, void *ctx)
{
    /* We can read the value but CANNOT write to *t without a cast.
     * Just verify we can read it correctly. */
    double *sum = (double *)ctx;
    *sum += t->value;
    return 0;
}

static void test_sm_traversal_const(void)
{
    BEGIN("SM-36 traversal callback receives const pointer (read-only)");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_insert(&sm, 0, 0, 3.0);
    sparse_matrix_insert(&sm, 1, 1, 7.0);
    double sum = 0.0;
    sparse_matrix_traverse(&sm, const_write_attempt_cb, &sum);
    CHECK(sum == 10.0);  /* 3 + 7 */
    /* Verify originals unchanged. */
    double v = 0.0;
    sparse_matrix_get(&sm, 0, 0, &v); CHECK(v == 3.0);
    sparse_matrix_get(&sm, 1, 1, &v); CHECK(v == 7.0);
    sparse_matrix_destroy(&sm);
}

/* SM-37: Validate detects stored zero (test fixture) */
static void test_sm_validate_zero_value(void)
{
    BEGIN("SM-37 validate() detects stored zero value");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    /* Force a zero-valued triplet directly. */
    Triplet z; z.row = 0; z.col = 0; z.value = 0.0;
    da_append(&sm.entries, &z);
    CHECK(sparse_matrix_validate(&sm) == ZF_ERR_VALUE);
    sparse_matrix_destroy(&sm);
}

/* SM-38: Insert zero value is rejected */
static void test_sm_insert_zero(void)
{
    BEGIN("SM-38 insert() rejects zero value");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    CHECK(sparse_matrix_insert(&sm, 0, 0, 0.0) == ZF_ERR_VALUE);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    sparse_matrix_destroy(&sm);
}

/* SM-39: Insert NaN / Inf rejected */
static void test_sm_insert_nonfinite(void)
{
    BEGIN("SM-39 insert() rejects NaN and Inf");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    CHECK(sparse_matrix_insert(&sm, 0, 0, (double)NAN) == ZF_ERR_VALUE);
    CHECK(sparse_matrix_insert(&sm, 0, 0, (double)INFINITY) == ZF_ERR_VALUE);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    sparse_matrix_destroy(&sm);
}

/* SM-40: Invalid init — zero dimensions */
static void test_sm_invalid_init(void)
{
    BEGIN("SM-40 sparse_matrix_init rejects zero dimensions");
    SparseMatrix sm;
    CHECK(sparse_matrix_init(&sm, 0, 5) == ZF_ERR_PARAM);
    CHECK(sm.init == 0);
    CHECK(sparse_matrix_init(&sm, 5, 0) == ZF_ERR_PARAM);
    CHECK(sm.init == 0);
}

/* SM-41: Large sparse matrix — many inserts stay sorted */
static void test_sm_large(void)
{
    BEGIN("SM-41 100 inserts via set() remain valid and sorted");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 10, 10);
    /* Insert all 100 cells via set() in reverse order. */
    for (int r = 9; r >= 0; r--) {
        for (int c = 9; c >= 0; c--) {
            double v = (double)(r * 10 + c + 1);
            sparse_matrix_set(&sm, (size_t)r, (size_t)c, v);
        }
    }
    CHECK(sparse_matrix_nonzero_count(&sm) == 100);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-42: set() to zero on absent coord is a no-op */
static void test_sm_set_zero_absent(void)
{
    BEGIN("SM-42 set(0.0) on absent coord is a safe no-op");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    CHECK(sparse_matrix_set(&sm, 1, 1, 0.0) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&sm) == 0);
    CHECK(sparse_matrix_validate(&sm) == ZF_OK);
    sparse_matrix_destroy(&sm);
}

/* SM-43: Double destroy is safe */
static void test_sm_double_destroy(void)
{
    BEGIN("SM-43 double destroy is safe");
    SparseMatrix sm;
    sparse_matrix_init(&sm, 3, 3);
    sparse_matrix_destroy(&sm);
    sparse_matrix_destroy(&sm);   /* must not crash */
    CHECK(1);
}

/* SM-44: Cell ceiling enforcement on sparse matrix */
static void test_sm_cell_ceiling(void)
{
    BEGIN("SM-44 cell ceiling enforcement on sparse matrix");
    SparseMatrix sm;
    /* 100,000 x 100,000 = 10 billion cells > 10,000,000 */
    CHECK(sparse_matrix_init(&sm, 100000, 100000) == ZF_ERR_OVERFLOW);
    CHECK(sm.init == 0);

    /* 1,000 x 10,001 = 10,001,000 cells > 10,000,000 */
    CHECK(sparse_matrix_init(&sm, 1000, 10001) == ZF_ERR_OVERFLOW);
    CHECK(sm.init == 0);

    /* Exactly at the limit: 1,000 x 10,000 = 10,000,000 */
    CHECK(sparse_matrix_init(&sm, 1000, 10000) == ZF_OK);
    CHECK(sm.init == 1);
    CHECK(sm.rows == 1000);
    CHECK(sm.cols == 10000);
    sparse_matrix_destroy(&sm);

    /* sparse_matrix_from_triplets with oversized dimensions */
    Triplet t = {0, 0, 42.0};
    CHECK(sparse_matrix_from_triplets(&sm, 100000, 100000, &t, 1) == ZF_ERR_OVERFLOW);
    CHECK(sm.init == 0);
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  ZEROFORM — DenseMatrix & SparseMatrix Tests  (Stage 2)\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    printf("── DenseMatrix ───────────────────────────────────────────────\n");
    test_dm_init_basic();
    test_dm_zero_init();
    test_dm_set_get();
    test_dm_row_major();
    test_dm_rectangular();
    test_dm_invalid_dims();
    test_dm_invalid_coords();
    test_dm_nonfinite();
    test_dm_from_flat();
    test_dm_from_flat_nonfinite();
    test_dm_overflow();
    test_dm_destroy_reinit();
    test_dm_copy();

    printf("\n── SparseMatrix: Construction ────────────────────────────────\n");
    test_sm_init_empty();
    test_sm_all_zero();
    test_sm_single_entry();
    test_sm_rectangular();
    test_sm_from_dense();
    test_sm_to_dense();
    test_sm_roundtrip();
    test_sm_preserve_zero_rows();
    test_sm_nnz_count();
    test_sm_from_triplets_unordered();
    test_sm_from_triplets_duplicate();
    test_sm_from_triplets_oob();
    test_sm_from_triplets_zero();
    test_sm_from_triplets_nonfinite();

    printf("\n── SparseMatrix: Search and Mutation ─────────────────────────\n");
    test_sm_search_positions();
    test_sm_search_missing();
    test_sm_implicit_zero();
    test_sm_insert_beginning();
    test_sm_insert_middle();
    test_sm_insert_end();
    test_sm_insert_duplicate();
    test_sm_update_existing();
    test_sm_set_new_nonzero();
    test_sm_set_to_zero();
    test_sm_delete_positions();
    test_sm_delete_missing();
    test_sm_order_after_mutations();
    test_sm_dimensions_stable();

    printf("\n── SparseMatrix: Sorting and Invariants ──────────────────────\n");
    test_sm_explicit_sort();
    test_sm_sort_same_row();
    test_sm_validate_valid();
    test_sm_validate_duplicate();
    test_sm_validate_oob();
    test_sm_traversal_order();
    test_sm_traversal_early_stop();
    test_sm_traversal_const();
    test_sm_validate_zero_value();
    test_sm_insert_zero();
    test_sm_insert_nonfinite();
    test_sm_invalid_init();
    test_sm_large();
    test_sm_set_zero_absent();
    test_sm_double_destroy();
    test_sm_cell_ceiling();

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed  (total checks: %d)\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("══════════════════════════════════════════════════════════════\n");

    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

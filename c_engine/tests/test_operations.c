/*
 * test_operations.c — Unit tests for SparseMatrix mathematical operations
 *
 * ZEROFORM Sparse Matrix Studio — Stage 3
 */

#include "../matrix_operations.h"
#include "../matrix.h"
#include "../zeroform.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

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

/* Helper: verify value at coordinate (returns true if matches expected) */
static int check_val(const SparseMatrix *sm, size_t r, size_t c, double expected)
{
    double out = -99999.0;
    if (sparse_matrix_get(sm, r, c, &out) != ZF_OK) return 0;
    return (fabs(out - expected) < 1e-12);
}

/* ═══════════════════════════════════════════════════════════════════
 * ── ADDITION TESTS ─────────────────────────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* ADD-01: Disjoint coordinates */
static void test_add_disjoint(void)
{
    BEGIN("ADD-01 same dimensions, disjoint coordinates");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 3, 3);
    sparse_matrix_init(&b, 3, 3);

    sparse_matrix_set(&a, 0, 0, 1.5);
    sparse_matrix_set(&a, 1, 1, 2.5);

    sparse_matrix_set(&b, 0, 1, 3.5);
    sparse_matrix_set(&b, 2, 2, 4.5);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&c) == 4);
    CHECK(check_val(&c, 0, 0, 1.5));
    CHECK(check_val(&c, 0, 1, 3.5));
    CHECK(check_val(&c, 1, 1, 2.5));
    CHECK(check_val(&c, 2, 2, 4.5));
    CHECK(check_val(&c, 1, 0, 0.0)); /* implicit zero */

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ADD-02: Overlapping coordinates */
static void test_add_overlapping(void)
{
    BEGIN("ADD-02 overlapping coordinates summed");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 2);
    sparse_matrix_init(&b, 2, 2);

    sparse_matrix_set(&a, 0, 1, 4.0);
    sparse_matrix_set(&a, 1, 0, 2.0);

    sparse_matrix_set(&b, 0, 1, 3.5);
    sparse_matrix_set(&b, 1, 0, -1.0);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&c) == 2);
    CHECK(check_val(&c, 0, 1, 7.5));
    CHECK(check_val(&c, 1, 0, 1.0));

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ADD-03: Zero cancellation */
static void test_add_canceling_to_zero(void)
{
    BEGIN("ADD-03 overlapping entries cancel out to zero");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 2);
    sparse_matrix_init(&b, 2, 2);

    sparse_matrix_set(&a, 0, 0, 5.0);
    sparse_matrix_set(&a, 1, 1, 3.0);

    sparse_matrix_set(&b, 0, 0, -5.0);
    sparse_matrix_set(&b, 1, 1, 2.0);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&c) == 1);
    CHECK(check_val(&c, 0, 0, 0.0)); /* cancelled */
    CHECK(check_val(&c, 1, 1, 5.0));

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ADD-04: Addition with all-zero matrix */
static void test_add_all_zero(void)
{
    BEGIN("ADD-04 addition with all-zero matrix");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 3, 4);
    sparse_matrix_init(&b, 3, 4);

    sparse_matrix_set(&a, 0, 2, 10.0);
    sparse_matrix_set(&a, 2, 1, 20.0);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&c) == 2);
    CHECK(check_val(&c, 0, 2, 10.0));
    CHECK(check_val(&c, 2, 1, 20.0));

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ADD-05: Rectangular matrices */
static void test_add_rectangular(void)
{
    BEGIN("ADD-05 addition of rectangular matrices");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 5);
    sparse_matrix_init(&b, 2, 5);

    sparse_matrix_set(&a, 0, 4, 1.0);
    sparse_matrix_set(&b, 1, 0, 2.0);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 2 && c.cols == 5);
    CHECK(sparse_matrix_nonzero_count(&c) == 2);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ADD-06: Dimension mismatch rejected */
static void test_add_dimension_mismatch(void)
{
    BEGIN("ADD-06 dimension mismatch rejected with ZF_ERR_DIMENSION");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 3);
    sparse_matrix_init(&b, 2, 4);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_ERR_DIMENSION);

    SparseMatrix d;
    sparse_matrix_init(&d, 3, 3);
    CHECK(sparse_matrix_add(&a, &d, &c) == ZF_ERR_DIMENSION);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&d);
}

/* ADD-07: Invalid parameters and aliasing rejection */
static void test_add_invalid_params_and_alias(void)
{
    BEGIN("ADD-07 invalid inputs and aliasing rejection");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 2);
    sparse_matrix_init(&b, 2, 2);

    CHECK(sparse_matrix_add(NULL, &b, &c) == ZF_ERR_PARAM);
    CHECK(sparse_matrix_add(&a, NULL, &c) == ZF_ERR_PARAM);
    CHECK(sparse_matrix_add(&a, &b, NULL) == ZF_ERR_PARAM);

    /* Aliasing rejection */
    CHECK(sparse_matrix_add(&a, &b, &a) == ZF_ERR_PARAM);
    CHECK(sparse_matrix_add(&a, &b, &b) == ZF_ERR_PARAM);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
}

/* ADD-08: Exact prompt specification example */
static void test_add_prompt_example(void)
{
    BEGIN("ADD-08 exact prompt addition example");
    /*
     * A = [ 1  0  2
     *       0  3  0 ]
     * B = [-1  4  0
     *       5 -3  7 ]
     * Expected C = [ 0 4 2
     *                5 0 7 ], nnz = 4
     */
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 3);
    sparse_matrix_init(&b, 2, 3);

    sparse_matrix_set(&a, 0, 0, 1.0);
    sparse_matrix_set(&a, 0, 2, 2.0);
    sparse_matrix_set(&a, 1, 1, 3.0);

    sparse_matrix_set(&b, 0, 0, -1.0);
    sparse_matrix_set(&b, 0, 1, 4.0);
    sparse_matrix_set(&b, 1, 0, 5.0);
    sparse_matrix_set(&b, 1, 1, -3.0);
    sparse_matrix_set(&b, 1, 2, 7.0);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 2 && c.cols == 3);
    CHECK(sparse_matrix_nonzero_count(&c) == 4);

    CHECK(check_val(&c, 0, 0, 0.0));
    CHECK(check_val(&c, 0, 1, 4.0));
    CHECK(check_val(&c, 0, 2, 2.0));
    CHECK(check_val(&c, 1, 0, 5.0));
    CHECK(check_val(&c, 1, 1, 0.0));
    CHECK(check_val(&c, 1, 2, 7.0));

    /* Invariant check */
    CHECK(sparse_matrix_validate(&c) == ZF_OK);

    /* Ensure inputs are unchanged */
    CHECK(sparse_matrix_nonzero_count(&a) == 3);
    CHECK(sparse_matrix_nonzero_count(&b) == 5);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ADD-09: Non-finite intermediate sum rejected */
static void test_add_nonfinite(void)
{
    BEGIN("ADD-09 non-finite intermediate sum rejected");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 2);
    sparse_matrix_init(&b, 2, 2);

    sparse_matrix_set(&a, 0, 0, DBL_MAX);
    sparse_matrix_set(&b, 0, 0, DBL_MAX);

    CHECK(sparse_matrix_add(&a, &b, &c) == ZF_ERR_VALUE);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
}

/* ═══════════════════════════════════════════════════════════════════
 * ── MULTIPLICATION TESTS ───────────────────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* MUL-01: Prompt rectangular multiplication example */
static void test_mul_prompt_example(void)
{
    BEGIN("MUL-01 exact prompt rectangular multiplication");
    /*
     * A = [1 0 2
     *      0 3 0] (2x3)
     * B = [4 5
     *      0 6
     *      7 0] (3x2)
     * Expected C = [18  5
     *                0 18] (2x2)
     */
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 3);
    sparse_matrix_init(&b, 3, 2);

    sparse_matrix_set(&a, 0, 0, 1.0);
    sparse_matrix_set(&a, 0, 2, 2.0);
    sparse_matrix_set(&a, 1, 1, 3.0);

    sparse_matrix_set(&b, 0, 0, 4.0);
    sparse_matrix_set(&b, 0, 1, 5.0);
    sparse_matrix_set(&b, 1, 1, 6.0);
    sparse_matrix_set(&b, 2, 0, 7.0);

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 2 && c.cols == 2);
    CHECK(sparse_matrix_nonzero_count(&c) == 3);

    CHECK(check_val(&c, 0, 0, 18.0));
    CHECK(check_val(&c, 0, 1, 5.0));
    CHECK(check_val(&c, 1, 0, 0.0));
    CHECK(check_val(&c, 1, 1, 18.0));

    CHECK(sparse_matrix_validate(&c) == ZF_OK);

    /* Inputs must remain unchanged */
    CHECK(sparse_matrix_nonzero_count(&a) == 3);
    CHECK(sparse_matrix_nonzero_count(&b) == 4);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* MUL-02: Square identity multiplication */
static void test_mul_identity(void)
{
    BEGIN("MUL-02 square identity multiplication A * I = A");
    SparseMatrix a, i, c;
    sparse_matrix_init(&a, 3, 3);
    sparse_matrix_init(&i, 3, 3);

    sparse_matrix_set(&a, 0, 0, 2.0);
    sparse_matrix_set(&a, 0, 2, -1.0);
    sparse_matrix_set(&a, 1, 1, 5.0);
    sparse_matrix_set(&a, 2, 0, 4.0);

    sparse_matrix_set(&i, 0, 0, 1.0);
    sparse_matrix_set(&i, 1, 1, 1.0);
    sparse_matrix_set(&i, 2, 2, 1.0);

    CHECK(sparse_matrix_multiply(&a, &i, &c) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&c) == 4);
    CHECK(check_val(&c, 0, 0, 2.0));
    CHECK(check_val(&c, 0, 2, -1.0));
    CHECK(check_val(&c, 1, 1, 5.0));
    CHECK(check_val(&c, 2, 0, 4.0));

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&i);
    sparse_matrix_destroy(&c);
}

/* MUL-03: Multiplication with all-zero matrix */
static void test_mul_all_zero(void)
{
    BEGIN("MUL-03 multiplication with all-zero matrices");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 3, 4);
    sparse_matrix_init(&b, 4, 2);

    sparse_matrix_set(&a, 0, 1, 5.0);
    /* B is completely empty */

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 3 && c.cols == 2);
    CHECK(sparse_matrix_nonzero_count(&c) == 0);
    sparse_matrix_destroy(&c);

    /* A empty, B nonzero */
    SparseMatrix a_empty;
    sparse_matrix_init(&a_empty, 3, 4);
    sparse_matrix_set(&b, 1, 0, 3.0);
    CHECK(sparse_matrix_multiply(&a_empty, &b, &c) == ZF_OK);
    CHECK(sparse_matrix_nonzero_count(&c) == 0);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&a_empty);
    sparse_matrix_destroy(&c);
}

/* MUL-04: Single-entry matrices */
static void test_mul_single_entry(void)
{
    BEGIN("MUL-04 single-entry multiplication");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 3);
    sparse_matrix_init(&b, 3, 4);

    sparse_matrix_set(&a, 1, 2, 3.0);
    sparse_matrix_set(&b, 2, 3, 4.0);

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 2 && c.cols == 4);
    CHECK(sparse_matrix_nonzero_count(&c) == 1);
    CHECK(check_val(&c, 1, 3, 12.0));

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* MUL-05: Products canceling out to zero */
static void test_mul_products_canceling(void)
{
    BEGIN("MUL-05 accumulated products cancel to zero");
    /*
     * A = [2  3] (1x2)
     * B = [ 3
     *      -2] (2x1)
     * A * B = [2*3 + 3*(-2)] = [6 - 6] = [0]
     */
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 1, 2);
    sparse_matrix_init(&b, 2, 1);

    sparse_matrix_set(&a, 0, 0, 2.0);
    sparse_matrix_set(&a, 0, 1, 3.0);

    sparse_matrix_set(&b, 0, 0, 3.0);
    sparse_matrix_set(&b, 1, 0, -2.0);

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 1 && c.cols == 1);
    CHECK(sparse_matrix_nonzero_count(&c) == 0);
    CHECK(check_val(&c, 0, 0, 0.0));

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* MUL-06: Incompatible dimensions */
static void test_mul_incompatible_dimensions(void)
{
    BEGIN("MUL-06 incompatible dimensions rejected");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 2, 3);
    sparse_matrix_init(&b, 4, 2);

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_ERR_DIMENSION);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
}

/* MUL-07: Aliasing rejection */
static void test_mul_aliasing_rejection(void)
{
    BEGIN("MUL-07 multiplication aliasing rejected");
    SparseMatrix a, b;
    sparse_matrix_init(&a, 2, 2);
    sparse_matrix_init(&b, 2, 2);

    CHECK(sparse_matrix_multiply(&a, &b, &a) == ZF_ERR_PARAM);
    CHECK(sparse_matrix_multiply(&a, &b, &b) == ZF_ERR_PARAM);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
}

/* MUL-08: Non-finite product rejected */
static void test_mul_nonfinite(void)
{
    BEGIN("MUL-08 non-finite product rejected with ZF_ERR_VALUE");
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 1, 1);
    sparse_matrix_init(&b, 1, 1);

    sparse_matrix_set(&a, 0, 0, DBL_MAX);
    sparse_matrix_set(&b, 0, 0, 2.0);

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_ERR_VALUE);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
}

/* MUL-09: Accumulator overflow with finite operands */
static void test_mul_accumulator_overflow(void)
{
    BEGIN("MUL-09 accumulator overflow with finite operands rejected");
    /*
     * A = [0.75 * DBL_MAX, 0.75 * DBL_MAX] (1x2)
     * B = [1.0
     *      1.0] (2x1)
     *
     * Individual products:
     *   (0.75 * DBL_MAX) * 1.0 = 0.75 * DBL_MAX  <-- finite!
     *   (0.75 * DBL_MAX) * 1.0 = 0.75 * DBL_MAX  <-- finite!
     *
     * Accumulator sum:
     *   (0.75 * DBL_MAX) + (0.75 * DBL_MAX) = 1.5 * DBL_MAX = +INFINITY
     *
     * Must return ZF_ERR_VALUE rather than reporting success with non-finite result.
     */
    SparseMatrix a, b, c;
    sparse_matrix_init(&a, 1, 2);
    sparse_matrix_init(&b, 2, 1);

    sparse_matrix_set(&a, 0, 0, 0.75 * DBL_MAX);
    sparse_matrix_set(&a, 0, 1, 0.75 * DBL_MAX);

    sparse_matrix_set(&b, 0, 0, 1.0);
    sparse_matrix_set(&b, 1, 0, 1.0);

    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_ERR_VALUE);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
}

/* MUL-10: Output dimensions exceeding MAX_CELLS rejected */
static void test_mul_output_cells_overflow(void)
{
    BEGIN("MUL-10 output dimensions exceeding MAX_CELLS rejected");
    SparseMatrix a, b, c;
    /* A is 10,000 x 1 (10,000 cells <= MAX_CELLS) */
    CHECK(sparse_matrix_init(&a, 10000, 1) == ZF_OK);
    /* B is 1 x 10,000 (10,000 cells <= MAX_CELLS) */
    CHECK(sparse_matrix_init(&b, 1, 10000) == ZF_OK);

    sparse_matrix_set(&a, 0, 0, 1.0);
    sparse_matrix_set(&b, 0, 0, 2.0);

    /* Output is 10,000 x 10,000 = 100,000,000 cells > MAX_CELLS */
    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_ERR_OVERFLOW);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);

    /* Output exactly at limit: 1,000 x 10,000 = 10,000,000 */
    CHECK(sparse_matrix_init(&a, 1000, 1) == ZF_OK);
    CHECK(sparse_matrix_init(&b, 1, 10000) == ZF_OK);
    sparse_matrix_set(&a, 0, 0, 1.0);
    sparse_matrix_set(&b, 0, 0, 2.0);
    CHECK(sparse_matrix_multiply(&a, &b, &c) == ZF_OK);
    CHECK(c.rows == 1000);
    CHECK(c.cols == 10000);
    double val = 0.0;
    CHECK(sparse_matrix_get(&c, 0, 0, &val) == ZF_OK);
    CHECK(val == 2.0);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&b);
    sparse_matrix_destroy(&c);
}

/* ═══════════════════════════════════════════════════════════════════
 * ── TRANSPOSE TESTS ────────────────────────────────────────────────
 * ═══════════════════════════════════════════════════════════════════ */

/* TR-01: Square matrix transpose */
static void test_transpose_square(void)
{
    BEGIN("TR-01 square matrix transpose");
    SparseMatrix a, at;
    sparse_matrix_init(&a, 3, 3);

    sparse_matrix_set(&a, 0, 1, 2.0);
    sparse_matrix_set(&a, 1, 2, 4.0);
    sparse_matrix_set(&a, 2, 0, 6.0);

    CHECK(sparse_matrix_transpose(&a, &at) == ZF_OK);
    CHECK(at.rows == 3 && at.cols == 3);
    CHECK(sparse_matrix_nonzero_count(&at) == 3);

    CHECK(check_val(&at, 1, 0, 2.0));
    CHECK(check_val(&at, 2, 1, 4.0));
    CHECK(check_val(&at, 0, 2, 6.0));
    CHECK(sparse_matrix_validate(&at) == ZF_OK);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&at);
}

/* TR-02: Rectangular matrix transpose */
static void test_transpose_rectangular(void)
{
    BEGIN("TR-02 rectangular matrix transpose 2x4 -> 4x2");
    SparseMatrix a, at;
    sparse_matrix_init(&a, 2, 4);

    sparse_matrix_set(&a, 0, 3, 7.5);
    sparse_matrix_set(&a, 1, 0, 8.5);
    sparse_matrix_set(&a, 1, 2, 9.5);

    CHECK(sparse_matrix_transpose(&a, &at) == ZF_OK);
    CHECK(at.rows == 4 && at.cols == 2);
    CHECK(sparse_matrix_nonzero_count(&at) == 3);

    CHECK(check_val(&at, 3, 0, 7.5));
    CHECK(check_val(&at, 0, 1, 8.5));
    CHECK(check_val(&at, 2, 1, 9.5));
    CHECK(sparse_matrix_validate(&at) == ZF_OK);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&at);
}

/* TR-03: Double transpose restores original matrix */
static void test_transpose_double(void)
{
    BEGIN("TR-03 double transpose restores original matrix (A^TT == A)");
    SparseMatrix a, at, att;
    sparse_matrix_init(&a, 3, 5);

    sparse_matrix_set(&a, 0, 1, 1.1);
    sparse_matrix_set(&a, 1, 4, 2.2);
    sparse_matrix_set(&a, 2, 2, 3.3);

    CHECK(sparse_matrix_transpose(&a, &at) == ZF_OK);
    CHECK(sparse_matrix_transpose(&at, &att) == ZF_OK);

    CHECK(att.rows == a.rows && att.cols == a.cols);
    CHECK(sparse_matrix_nonzero_count(&att) == sparse_matrix_nonzero_count(&a));

    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 5; c++) {
            double va = 0.0, vatt = 0.0;
            sparse_matrix_get(&a, r, c, &va);
            sparse_matrix_get(&att, r, c, &vatt);
            CHECK(fabs(va - vatt) < 1e-12);
        }
    }

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&at);
    sparse_matrix_destroy(&att);
}

/* TR-04: Transpose all-zero matrix */
static void test_transpose_all_zero(void)
{
    BEGIN("TR-04 transpose of all-zero matrix");
    SparseMatrix a, at;
    sparse_matrix_init(&a, 4, 3);

    CHECK(sparse_matrix_transpose(&a, &at) == ZF_OK);
    CHECK(at.rows == 3 && at.cols == 4);
    CHECK(sparse_matrix_nonzero_count(&at) == 0);

    sparse_matrix_destroy(&a);
    sparse_matrix_destroy(&at);
}

/* TR-05: Transpose aliasing rejection */
static void test_transpose_aliasing(void)
{
    BEGIN("TR-05 transpose aliasing rejected");
    SparseMatrix a;
    sparse_matrix_init(&a, 2, 2);

    CHECK(sparse_matrix_transpose(&a, &a) == ZF_ERR_PARAM);

    sparse_matrix_destroy(&a);
}

/* ═══════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  ZEROFORM — Matrix Operations Tests  (Stage 3)\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    printf("── Matrix Addition ───────────────────────────────────────────\n");
    test_add_disjoint();
    test_add_overlapping();
    test_add_canceling_to_zero();
    test_add_all_zero();
    test_add_rectangular();
    test_add_dimension_mismatch();
    test_add_invalid_params_and_alias();
    test_add_prompt_example();
    test_add_nonfinite();

    printf("\n── Matrix Multiplication ─────────────────────────────────────\n");
    test_mul_prompt_example();
    test_mul_identity();
    test_mul_all_zero();
    test_mul_single_entry();
    test_mul_products_canceling();
    test_mul_incompatible_dimensions();
    test_mul_aliasing_rejection();
    test_mul_nonfinite();
    test_mul_accumulator_overflow();
    test_mul_output_cells_overflow();

    printf("\n── Matrix Transpose ──────────────────────────────────────────\n");
    test_transpose_square();
    test_transpose_rectangular();
    test_transpose_double();
    test_transpose_all_zero();
    test_transpose_aliasing();

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed  (total checks: %d)\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("══════════════════════════════════════════════════════════════\n");

    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

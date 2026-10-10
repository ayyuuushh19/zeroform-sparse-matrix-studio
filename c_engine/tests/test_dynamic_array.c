/*
 * test_dynamic_array.c — Unit tests for DynamicArray (Stage 1)
 *
 * Test runner design
 * ──────────────────
 * • No third-party test framework — hand-rolled CHECK() macro.
 * • Each test is a void function that calls CHECK().
 * • A global counter tracks passes and failures.
 * • main() returns EXIT_SUCCESS only if failures == 0.
 * • Output is human-readable; every failure prints file/line/expression.
 *
 * Test philosophy
 * ───────────────
 * We test BEHAVIOR, not internals.  Every check verifies a documented
 * postcondition of the public API (zeroform.h).
 */

#include "../zeroform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>   /* SIZE_MAX */

/* ═══════════════════════════════════════════════════════════════════
 * Minimal test framework
 * ═══════════════════════════════════════════════════════════════════ */

static int g_pass = 0;
static int g_fail = 0;
static const char *g_current_test = "(none)";

/* CHECK(expr) — verify that `expr` is non-zero (truthy).
 * On failure: prints a diagnostic and increments g_fail.
 * On success: increments g_pass. */
#define CHECK(expr)                                                         \
    do {                                                                    \
        if (!(expr)) {                                                      \
            fprintf(stderr, "  FAIL [%s] line %d: %s\n",                   \
                    g_current_test, __LINE__, #expr);                       \
            g_fail++;                                                       \
        } else {                                                            \
            g_pass++;                                                       \
        }                                                                   \
    } while (0)

/* BEGIN_TEST / END_TEST — wrap each test function for console output. */
#define BEGIN_TEST(name)                                                    \
    do {                                                                    \
        g_current_test = (name);                                            \
        printf("  %-55s", (name));                                          \
        fflush(stdout);                                                     \
    } while (0)

#define END_TEST()                                                          \
    do {                                                                    \
        printf("\n");                                                       \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════
 * Helper: a small struct used to test non-primitive element types
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   id;
    float value;
    char  tag[4];   /* 4-byte fixed label, e.g. "ABC\0" */
} SmallRecord;

static SmallRecord make_record(int id, float val, const char *tag)
{
    SmallRecord r;
    r.id    = id;
    r.value = val;
    /* strncpy to avoid -Wstringop-truncation; tag is always <= 3 chars + NUL */
    strncpy(r.tag, tag, 3);
    r.tag[3] = '\0';
    return r;
}

static int record_eq(const SmallRecord *a, const SmallRecord *b)
{
    return (a->id == b->id)
        && (a->value == b->value)
        && (strncmp(a->tag, b->tag, 4) == 0);
}

/* ═══════════════════════════════════════════════════════════════════
 * Test implementations
 * ═══════════════════════════════════════════════════════════════════ */

/* T01 — Empty array initialization */
static void test_init_empty(void)
{
    BEGIN_TEST("T01 init empty (no pre-alloc)");
    DynamicArray da;
    ZF_Status s = da_init(&da, sizeof(int), 0);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 0);
    CHECK(da_capacity(&da) == 0);
    da_destroy(&da);
    END_TEST();
}

/* T02 — Init with pre-allocation */
static void test_init_preallocated(void)
{
    BEGIN_TEST("T02 init with pre-allocation (cap=8)");
    DynamicArray da;
    ZF_Status s = da_init(&da, sizeof(double), 8);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 0);
    CHECK(da_capacity(&da) == 8);
    da_destroy(&da);
    END_TEST();
}

/* T03 — Invalid element size (zero) */
static void test_invalid_elem_size(void)
{
    BEGIN_TEST("T03 invalid elem_size == 0 -> ZF_ERR_PARAM");
    DynamicArray da;
    ZF_Status s = da_init(&da, 0, 0);
    CHECK(s == ZF_ERR_PARAM);
    /* The struct must be in a safe inert state */
    CHECK(da.data      == NULL);
    CHECK(da.length    == 0);
    CHECK(da.capacity  == 0);
    CHECK(da.elem_size == 0);
    /* da_destroy on an inert struct must be safe */
    da_destroy(&da);
    END_TEST();
}

/* T04 — Append one element */
static void test_append_one(void)
{
    BEGIN_TEST("T04 append one int element");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    int val = 42;
    ZF_Status s = da_append(&da, &val);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 1);

    int got = 0;
    da_get(&da, 0, &got);
    CHECK(got == 42);

    da_destroy(&da);
    END_TEST();
}

/* T05 — Append many elements and verify all values are preserved */
static void test_append_many(void)
{
    BEGIN_TEST("T05 append 100 ints, verify all");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    for (int i = 0; i < 100; i++) {
        ZF_Status s = da_append(&da, &i);
        CHECK(s == ZF_OK);
    }
    CHECK(da_length(&da) == 100);

    int all_ok = 1;
    for (int i = 0; i < 100; i++) {
        int got = -1;
        da_get(&da, (size_t)i, &got);
        if (got != i) { all_ok = 0; }
    }
    CHECK(all_ok);

    da_destroy(&da);
    END_TEST();
}

/* T06 — Capacity grows geometrically; values survive realloc */
static void test_capacity_growth(void)
{
    BEGIN_TEST("T06 capacity >= length after 50 appends");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    for (int i = 0; i < 50; i++) {
        da_append(&da, &i);
        /* invariant: capacity must always be >= length */
        CHECK(da_capacity(&da) >= da_length(&da));
    }
    CHECK(da_length(&da) == 50);

    /* Verify every element kept its value through reallocations. */
    int all_ok = 1;
    for (int i = 0; i < 50; i++) {
        int got = -1;
        da_get(&da, (size_t)i, &got);
        if (got != i) { all_ok = 0; }
    }
    CHECK(all_ok);

    da_destroy(&da);
    END_TEST();
}

/* T07 — Element preservation through exactly 6 doublings (64 elements) */
static void test_six_doublings(void)
{
    BEGIN_TEST("T07 element preservation through 6 capacity doublings");
    DynamicArray da;
    da_init(&da, sizeof(int), 1);  /* start at cap=1 to force growth early */

    for (int i = 0; i < 64; i++) {
        da_append(&da, &i);
    }
    int all_ok = 1;
    for (int i = 0; i < 64; i++) {
        int got = -1;
        da_get(&da, (size_t)i, &got);
        if (got != i) { all_ok = 0; }
    }
    CHECK(all_ok);
    da_destroy(&da);
    END_TEST();
}

/* T08 — Insert at beginning */
static void test_insert_beginning(void)
{
    BEGIN_TEST("T08 insert at index 0 (beginning)");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int a = 10, b = 20, c = 30;
    da_append(&da, &a);
    da_append(&da, &b);

    /* Insert 30 at position 0 → [30, 10, 20] */
    ZF_Status s = da_insert(&da, 0, &c);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 3);

    int v0 = 0, v1 = 0, v2 = 0;
    da_get(&da, 0, &v0);
    da_get(&da, 1, &v1);
    da_get(&da, 2, &v2);
    CHECK(v0 == 30);
    CHECK(v1 == 10);
    CHECK(v2 == 20);

    da_destroy(&da);
    END_TEST();
}

/* T09 — Insert in middle */
static void test_insert_middle(void)
{
    BEGIN_TEST("T09 insert in middle");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int vals[] = {1, 2, 4, 5};
    for (int i = 0; i < 4; i++) da_append(&da, &vals[i]);

    /* Insert 3 at index 2 → [1, 2, 3, 4, 5] */
    int three = 3;
    ZF_Status s = da_insert(&da, 2, &three);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 5);

    int all_ok = 1;
    for (int i = 0; i < 5; i++) {
        int got = -1;
        da_get(&da, (size_t)i, &got);
        if (got != i + 1) { all_ok = 0; }
    }
    CHECK(all_ok);

    da_destroy(&da);
    END_TEST();
}

/* T10 — Insert at end (equivalent to append) */
static void test_insert_end(void)
{
    BEGIN_TEST("T10 insert at index == length (end)");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    int a = 7, b = 9;
    da_append(&da, &a);

    ZF_Status s = da_insert(&da, 1, &b);  /* index == length */
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 2);

    int got = 0;
    da_get(&da, 1, &got);
    CHECK(got == 9);

    da_destroy(&da);
    END_TEST();
}

/* T11 — da_get and da_set */
static void test_get_set(void)
{
    BEGIN_TEST("T11 da_get and da_set round-trip");
    DynamicArray da;
    da_init(&da, sizeof(double), 4);

    double d1 = 3.14, d2 = 2.71;
    da_append(&da, &d1);
    da_append(&da, &d2);

    double got = 0.0;
    CHECK(da_get(&da, 0, &got) == ZF_OK);
    CHECK(got == 3.14);

    double new_val = 1.41;
    CHECK(da_set(&da, 0, &new_val) == ZF_OK);

    da_get(&da, 0, &got);
    CHECK(got == 1.41);

    /* Index 1 still untouched */
    da_get(&da, 1, &got);
    CHECK(got == 2.71);

    da_destroy(&da);
    END_TEST();
}

/* T12 — Remove from beginning */
static void test_remove_beginning(void)
{
    BEGIN_TEST("T12 remove at index 0 (beginning)");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int vals[] = {10, 20, 30};
    for (int i = 0; i < 3; i++) da_append(&da, &vals[i]);

    ZF_Status s = da_remove(&da, 0);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 2);

    int v0 = 0, v1 = 0;
    da_get(&da, 0, &v0);
    da_get(&da, 1, &v1);
    CHECK(v0 == 20);
    CHECK(v1 == 30);

    da_destroy(&da);
    END_TEST();
}

/* T13 — Remove from middle */
static void test_remove_middle(void)
{
    BEGIN_TEST("T13 remove from middle");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int vals[] = {1, 99, 3, 4};
    for (int i = 0; i < 4; i++) da_append(&da, &vals[i]);

    /* Remove index 1 (value 99) → [1, 3, 4] */
    ZF_Status s = da_remove(&da, 1);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 3);

    int v0 = 0, v1 = 0, v2 = 0;
    da_get(&da, 0, &v0);
    da_get(&da, 1, &v1);
    da_get(&da, 2, &v2);
    CHECK(v0 == 1);
    CHECK(v1 == 3);
    CHECK(v2 == 4);

    da_destroy(&da);
    END_TEST();
}

/* T14 — Remove from end */
static void test_remove_end(void)
{
    BEGIN_TEST("T14 remove last element");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int a = 5, b = 10;
    da_append(&da, &a);
    da_append(&da, &b);

    ZF_Status s = da_remove(&da, 1);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 1);

    int got = 0;
    da_get(&da, 0, &got);
    CHECK(got == 5);

    da_destroy(&da);
    END_TEST();
}

/* T15 — Pop from nonempty array */
static void test_pop_nonempty(void)
{
    BEGIN_TEST("T15 pop from nonempty array");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int a = 100, b = 200;
    da_append(&da, &a);
    da_append(&da, &b);

    int popped = 0;
    ZF_Status s = da_pop(&da, &popped);
    CHECK(s == ZF_OK);
    CHECK(popped == 200);
    CHECK(da_length(&da) == 1);

    /* Remaining element still correct */
    int got = 0;
    da_get(&da, 0, &got);
    CHECK(got == 100);

    da_destroy(&da);
    END_TEST();
}

/* T16 — Pop from empty array */
static void test_pop_empty(void)
{
    BEGIN_TEST("T16 pop from empty array -> ZF_ERR_EMPTY");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    ZF_Status s = da_pop(&da, NULL);
    CHECK(s == ZF_ERR_EMPTY);
    CHECK(da_length(&da) == 0);

    da_destroy(&da);
    END_TEST();
}

/* T17 — Invalid access index */
static void test_invalid_access_index(void)
{
    BEGIN_TEST("T17 get/set out-of-range index -> ZF_ERR_INDEX");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int v = 1;
    da_append(&da, &v);

    int out = 0;
    CHECK(da_get(&da, 1, &out) == ZF_ERR_INDEX);  /* only [0] is valid */
    CHECK(da_set(&da, 5, &v)   == ZF_ERR_INDEX);
    CHECK(da_ptr(&da, 2)       == NULL);

    da_destroy(&da);
    END_TEST();
}

/* T18 — Invalid insertion index */
static void test_invalid_insert_index(void)
{
    BEGIN_TEST("T18 insert beyond length -> ZF_ERR_INDEX");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int v = 1;
    da_append(&da, &v);

    /* length is 1; valid insert range is [0, 1].  Index 5 is illegal. */
    CHECK(da_insert(&da, 5, &v) == ZF_ERR_INDEX);
    CHECK(da_length(&da) == 1);  /* must be unchanged */

    da_destroy(&da);
    END_TEST();
}

/* T19 — Invalid removal index */
static void test_invalid_remove_index(void)
{
    BEGIN_TEST("T19 remove out-of-range index -> ZF_ERR_INDEX");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int v = 1;
    da_append(&da, &v);

    CHECK(da_remove(&da, 1) == ZF_ERR_INDEX);  /* only index 0 is valid */
    CHECK(da_remove(&da, 99) == ZF_ERR_INDEX);
    CHECK(da_length(&da) == 1);

    da_destroy(&da);
    END_TEST();
}

/* T20 — Remove from empty array */
static void test_remove_empty(void)
{
    BEGIN_TEST("T20 remove from empty array -> ZF_ERR_INDEX");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    CHECK(da_remove(&da, 0) == ZF_ERR_INDEX);

    da_destroy(&da);
    END_TEST();
}

/* T21 — Clear and reuse */
static void test_clear_and_reuse(void)
{
    BEGIN_TEST("T21 clear preserves allocation; reuse works");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    for (int i = 0; i < 4; i++) da_append(&da, &i);
    size_t cap_before = da_capacity(&da);

    da_clear(&da);
    CHECK(da_length(&da) == 0);
    CHECK(da_capacity(&da) == cap_before);  /* allocation kept */
    CHECK(da.data != NULL);

    /* Reuse without reallocation. */
    int v = 77;
    CHECK(da_append(&da, &v) == ZF_OK);
    CHECK(da_length(&da) == 1);

    int got = 0;
    da_get(&da, 0, &got);
    CHECK(got == 77);

    da_destroy(&da);
    END_TEST();
}

/* T22 — Multiple element types: double */
static void test_elem_type_double(void)
{
    BEGIN_TEST("T22 element type: double");
    DynamicArray da;
    da_init(&da, sizeof(double), 0);

    double vals[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    for (int i = 0; i < 5; i++) da_append(&da, &vals[i]);

    int all_ok = 1;
    for (int i = 0; i < 5; i++) {
        double got = 0.0;
        da_get(&da, (size_t)i, &got);
        if (got != vals[i]) { all_ok = 0; }
    }
    CHECK(all_ok);

    da_destroy(&da);
    END_TEST();
}

/* T23 — Multiple element types: SmallRecord struct */
static void test_elem_type_struct(void)
{
    BEGIN_TEST("T23 element type: SmallRecord struct");
    DynamicArray da;
    da_init(&da, sizeof(SmallRecord), 0);

    SmallRecord r1 = make_record(1, 1.5f, "AAA");
    SmallRecord r2 = make_record(2, 2.5f, "BBB");
    SmallRecord r3 = make_record(3, 3.5f, "CCC");

    da_append(&da, &r1);
    da_append(&da, &r2);
    da_append(&da, &r3);

    CHECK(da_length(&da) == 3);

    SmallRecord got;
    da_get(&da, 0, &got); CHECK(record_eq(&got, &r1));
    da_get(&da, 1, &got); CHECK(record_eq(&got, &r2));
    da_get(&da, 2, &got); CHECK(record_eq(&got, &r3));

    da_destroy(&da);
    END_TEST();
}

/* T24 — Destroy and re-initialize */
static void test_destroy_and_reinit(void)
{
    BEGIN_TEST("T24 destroy then re-init on same struct");
    DynamicArray da;
    da_init(&da, sizeof(int), 8);

    int v = 5;
    da_append(&da, &v);
    da_destroy(&da);

    /* All fields must be zero after destroy. */
    CHECK(da.data      == NULL);
    CHECK(da.length    == 0);
    CHECK(da.capacity  == 0);
    CHECK(da.elem_size == 0);

    /* Re-init with a different elem_size must work. */
    ZF_Status s = da_init(&da, sizeof(double), 4);
    CHECK(s == ZF_OK);
    CHECK(da.elem_size == sizeof(double));

    double d = 9.9;
    da_append(&da, &d);
    CHECK(da_length(&da) == 1);

    double got = 0.0;
    da_get(&da, 0, &got);
    CHECK(got == 9.9);

    da_destroy(&da);
    END_TEST();
}

/* T25 — Double destroy is safe */
static void test_double_destroy(void)
{
    BEGIN_TEST("T25 double destroy is safe");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);
    da_destroy(&da);
    da_destroy(&da);  /* must not crash; free(NULL) is safe */
    CHECK(1);         /* reaching here means no crash */
    END_TEST();
}

/* T26 — da_ptr returns correct pointer */
static void test_ptr_accessor(void)
{
    BEGIN_TEST("T26 da_ptr returns valid pointer to element");
    DynamicArray da;
    da_init(&da, sizeof(int), 4);

    int a = 42, b = 84;
    da_append(&da, &a);
    da_append(&da, &b);

    int *p0 = (int *)da_ptr(&da, 0);
    int *p1 = (int *)da_ptr(&da, 1);
    CHECK(p0 != NULL && *p0 == 42);
    CHECK(p1 != NULL && *p1 == 84);
    CHECK(da_ptr(&da, 2) == NULL);  /* out of range */

    da_destroy(&da);
    END_TEST();
}

/* T27 — Self-referential append */
static void test_self_referential_append(void)
{
    BEGIN_TEST("T27 self-referential append (elem inside array)");
    DynamicArray da;
    da_init(&da, sizeof(int), 1);  /* cap=1 so first self-append may realloc */

    int v = 99;
    da_append(&da, &v);  /* [99], now at cap boundary */

    /* Append a pointer to element [0] — self-referential. */
    void *p = da_ptr(&da, 0);
    ZF_Status s = da_append(&da, p);
    CHECK(s == ZF_OK);
    CHECK(da_length(&da) == 2);

    int got = 0;
    da_get(&da, 1, &got);
    CHECK(got == 99);  /* value must be the original 99, not garbage */

    da_destroy(&da);
    END_TEST();
}

/* T28 — Overflow guard for very large elem_size * capacity */
static void test_overflow_guard(void)
{
    BEGIN_TEST("T28 overflow guard: SIZE_MAX elem_size");
    DynamicArray da;
    /* elem_size = SIZE_MAX means any capacity > 1 would overflow SIZE_MAX.
     * Pre-allocating 2 slots should be caught. */
    ZF_Status s = da_init(&da, SIZE_MAX, 2);
    CHECK(s == ZF_ERR_OVERFLOW || s == ZF_ERR_NOMEM);
    /* The struct must still be in a safe state (don't leak). */
    da_destroy(&da);
    END_TEST();
}

/* T29 — Overflow guard triggered during append growth */
static void test_overflow_on_grow(void)
{
    BEGIN_TEST("T29 overflow guard: large elem_size during append growth");
    /*
     * We can't easily simulate realloc failure portably without mocking.
     * Instead we verify that an array initialised with an intentionally
     * large elem_size but zero initial capacity handles the first append
     * either with success (small system with lots of memory) or one of
     * the documented error codes — never undefined behaviour.
     *
     * We use SIZE_MAX / 2 + 1 as elem_size; allocating 4 slots
     * (the default first grow target) would need (SIZE_MAX/2+1)*4 bytes,
     * which overflows on any 64-bit platform.
     */
    size_t big = SIZE_MAX / 2 + 1;
    DynamicArray da;
    ZF_Status s = da_init(&da, big, 0);
    if (s == ZF_OK) {
        /* init succeeded (zero pre-alloc); now try to append */
        unsigned char dummy[1] = {0};
        /* The dummy element is smaller than big, but the array thinks
         * its elem_size is `big`, so da_grow must detect overflow. */
        ZF_Status s2 = da_append(&da, dummy);
        CHECK(s2 == ZF_ERR_OVERFLOW || s2 == ZF_ERR_NOMEM);
    } else {
        /* init itself hit overflow — also acceptable */
        CHECK(s == ZF_ERR_OVERFLOW || s == ZF_ERR_NOMEM);
    }
    da_destroy(&da);
    END_TEST();
}

/* T30 — Accessor functions on NULL pointer */
static void test_null_accessors(void)
{
    BEGIN_TEST("T30 accessor functions tolerate NULL pointer");
    CHECK(da_length(NULL)   == 0);
    CHECK(da_capacity(NULL) == 0);
    CHECK(da_ptr(NULL, 0)   == NULL);
    END_TEST();
}

/* T31 — Mixed insert and remove sequence */
static void test_mixed_insert_remove(void)
{
    BEGIN_TEST("T31 interleaved insert/remove correctness");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    /* Build [10, 20, 30, 40, 50] */
    int vals[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) da_append(&da, &vals[i]);

    /* Insert 15 at index 1 → [10, 15, 20, 30, 40, 50] */
    int fifteen = 15;
    da_insert(&da, 1, &fifteen);
    CHECK(da_length(&da) == 6);

    /* Remove index 3 (value 30) → [10, 15, 20, 40, 50] */
    da_remove(&da, 3);
    CHECK(da_length(&da) == 5);

    int expected[] = {10, 15, 20, 40, 50};
    int all_ok = 1;
    for (int i = 0; i < 5; i++) {
        int got = 0;
        da_get(&da, (size_t)i, &got);
        if (got != expected[i]) { all_ok = 0; }
    }
    CHECK(all_ok);

    da_destroy(&da);
    END_TEST();
}

/* T32 — Pop returns correct value and mutates length */
static void test_pop_returns_value(void)
{
    BEGIN_TEST("T32 pop with NULL out doesn't crash; length shrinks");
    DynamicArray da;
    da_init(&da, sizeof(int), 0);

    int a = 1, b = 2, c = 3;
    da_append(&da, &a);
    da_append(&da, &b);
    da_append(&da, &c);

    /* Pop without capturing the value */
    CHECK(da_pop(&da, NULL) == ZF_OK);
    CHECK(da_length(&da) == 2);

    /* Verify remaining elements */
    int v0 = 0, v1 = 0;
    da_get(&da, 0, &v0);
    da_get(&da, 1, &v1);
    CHECK(v0 == 1);
    CHECK(v1 == 2);

    da_destroy(&da);
    END_TEST();
}

/* T33 — zf_status_str never returns NULL */
static void test_status_str(void)
{
    BEGIN_TEST("T33 zf_status_str returns non-NULL for all codes");
    CHECK(zf_status_str(ZF_OK)           != NULL);
    CHECK(zf_status_str(ZF_ERR_NOMEM)    != NULL);
    CHECK(zf_status_str(ZF_ERR_INDEX)    != NULL);
    CHECK(zf_status_str(ZF_ERR_EMPTY)    != NULL);
    CHECK(zf_status_str(ZF_ERR_PARAM)    != NULL);
    CHECK(zf_status_str(ZF_ERR_OVERFLOW) != NULL);
    /* Unknown code */
    CHECK(zf_status_str((ZF_Status)999)  != NULL);
    END_TEST();
}

/* ═══════════════════════════════════════════════════════════════════
 * main — run all tests and print summary
 * ═══════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  ZEROFORM — DynamicArray Unit Tests  (Stage 1)\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    test_init_empty();
    test_init_preallocated();
    test_invalid_elem_size();
    test_append_one();
    test_append_many();
    test_capacity_growth();
    test_six_doublings();
    test_insert_beginning();
    test_insert_middle();
    test_insert_end();
    test_get_set();
    test_remove_beginning();
    test_remove_middle();
    test_remove_end();
    test_pop_nonempty();
    test_pop_empty();
    test_invalid_access_index();
    test_invalid_insert_index();
    test_invalid_remove_index();
    test_remove_empty();
    test_clear_and_reuse();
    test_elem_type_double();
    test_elem_type_struct();
    test_destroy_and_reinit();
    test_double_destroy();
    test_ptr_accessor();
    test_self_referential_append();
    test_overflow_guard();
    test_overflow_on_grow();
    test_null_accessors();
    test_mixed_insert_remove();
    test_pop_returns_value();
    test_status_str();

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed  (total checks: %d)\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("══════════════════════════════════════════════════════════════\n");

    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

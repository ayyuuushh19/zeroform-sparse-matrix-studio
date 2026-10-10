/*
 * zeroform.h — Public interface for ZEROFORM Sparse Matrix Studio
 *
 * This header defines:
 *   - ZF_Status  : error/success codes used by every public function
 *   - DynamicArray: a generic, resizable array for any fixed-size element
 *
 * Stage 1 scope: DynamicArray only.
 * Stages 2+: Triplet, SparseMatrix, DenseMatrix will be added here.
 *
 * Memory ownership rule (read before using this API)
 * ─────────────────────────────────────────────────
 *   A DynamicArray OWNS its backing allocation.  You initialise it with
 *   da_init(), use it, and release it with da_destroy().  After
 *   da_destroy() the struct is in a safe zeroed state and must NOT be used
 *   without another call to da_init().
 *
 *   Elements are copied INTO the array as raw bytes.  If an element itself
 *   contains a pointer (e.g. a char*), the array does NOT manage that
 *   pointed-to memory; that is the caller's responsibility.
 *
 * Index convention
 * ────────────────
 *   All indexes are zero-based.  Valid element indexes are [0, length-1].
 *   Valid insertion indexes are [0, length] (inserting at `length`
 *   appends).  Any index outside these ranges returns ZF_ERR_INDEX.
 *
 * Thread safety
 * ─────────────
 *   None.  External locking is required for concurrent access.
 */

#ifndef ZEROFORM_H
#define ZEROFORM_H

#include <stddef.h>   /* size_t  */
#include <stdint.h>   /* SIZE_MAX */

/* ═══════════════════════════════════════════════════════════════════
 * Status codes
 * ═══════════════════════════════════════════════════════════════════
 * Every public function that can fail returns a ZF_Status.
 * ZF_OK is always zero so callers can write:  if (da_append(...)) { error }
 */
typedef enum {
    ZF_OK            =  0,  /* operation succeeded                          */
    ZF_ERR_NOMEM     = -1,  /* malloc / realloc returned NULL               */
    ZF_ERR_INDEX     = -2,  /* index out of valid range                     */
    ZF_ERR_EMPTY     = -3,  /* operation requires a non-empty array         */
    ZF_ERR_PARAM     = -4,  /* invalid parameter (e.g. elem_size == 0)      */
    ZF_ERR_OVERFLOW  = -5   /* size calculation would overflow SIZE_MAX      */
} ZF_Status;

/* Human-readable string for a status code.  Never returns NULL. */
const char *zf_status_str(ZF_Status s);

/* ═══════════════════════════════════════════════════════════════════
 * DynamicArray
 * ═══════════════════════════════════════════════════════════════════
 *
 * A generic, heap-backed resizable array that stores elements as
 * contiguous raw bytes.  The element type is fixed at init time via
 * `elem_size`.
 *
 * Fields (treat as read-only from outside; use the accessor functions):
 *
 *   data      — pointer to the heap allocation (may be NULL when cap == 0)
 *   length    — number of elements currently stored
 *   capacity  — number of elements the current allocation can hold
 *   elem_size — size in bytes of a single element (immutable after init)
 *
 * Growth strategy
 * ───────────────
 *   When an append or insert would exceed `capacity`, the array doubles
 *   its capacity (capped to prevent overflow).  This gives O(1) amortised
 *   cost per append over any sequence of n appends.
 *
 * Shrinking
 * ─────────
 *   This implementation does NOT shrink automatically.  Use da_clear()
 *   to reset length to zero while keeping the allocation, or da_destroy()
 *   to free everything.  This avoids allocation failures on shrink paths.
 */
typedef struct {
    void   *data;       /* heap allocation; NULL iff capacity == 0          */
    size_t  length;     /* current number of elements                       */
    size_t  capacity;   /* allocated slots (in elements, not bytes)         */
    size_t  elem_size;  /* bytes per element; set at init, never changed    */
} DynamicArray;

/* ───────────────────────────────────────────────────────────────────
 * Lifecycle
 * ─────────────────────────────────────────────────────────────────── */

/*
 * da_init — initialise a DynamicArray.
 *
 * Parameters
 *   da           : pointer to an uninitialised or zeroed DynamicArray struct.
 *   elem_size    : size in bytes of each element; must be >= 1.
 *   init_capacity: number of elements to pre-allocate.  Pass 0 to defer
 *                  the first allocation until the first append/insert.
 *
 * Returns
 *   ZF_OK         — success; `da` is ready to use.
 *   ZF_ERR_PARAM  — elem_size == 0.
 *   ZF_ERR_NOMEM  — pre-allocation failed.
 *   ZF_ERR_OVERFLOW — init_capacity * elem_size would overflow SIZE_MAX.
 *
 * On any error the struct is left in a safe, inert state (length == 0,
 * capacity == 0, data == NULL).
 */
ZF_Status da_init(DynamicArray *da, size_t elem_size, size_t init_capacity);

/*
 * da_destroy — release the heap allocation and reset all fields to zero.
 *
 * After this call the struct is in the same state as a zero-initialised
 * struct.  It is safe to call da_init() on it again.
 *
 * Calling da_destroy() on an already-destroyed (zeroed) array is safe
 * (free(NULL) is a no-op in standard C).
 *
 * Parameters
 *   da : must not be NULL.
 */
void da_destroy(DynamicArray *da);

/* ───────────────────────────────────────────────────────────────────
 * Mutation
 * ─────────────────────────────────────────────────────────────────── */

/*
 * da_append — copy `elem` as the new last element.
 *
 * If `elem` points inside the array's own storage (self-referential
 * append), the implementation copies the element value BEFORE any
 * reallocation so the source bytes remain valid.
 *
 * Amortised O(1) due to the doubling growth strategy.
 *
 * Returns ZF_OK, ZF_ERR_NOMEM, or ZF_ERR_OVERFLOW.
 */
ZF_Status da_append(DynamicArray *da, const void *elem);

/*
 * da_insert — insert a copy of `elem` at position `index`.
 *
 * Elements at [index, length) are shifted one slot toward the end.
 * `index` == length is equivalent to da_append().
 *
 * Returns ZF_OK, ZF_ERR_NOMEM, ZF_ERR_OVERFLOW, or ZF_ERR_INDEX
 * (if index > length).
 */
ZF_Status da_insert(DynamicArray *da, size_t index, const void *elem);

/*
 * da_remove — remove the element at `index`, shifting later elements left.
 *
 * The array's length decreases by one.  The backing allocation is NOT
 * shrunk.
 *
 * Returns ZF_OK or ZF_ERR_INDEX (index >= length, or array is empty).
 */
ZF_Status da_remove(DynamicArray *da, size_t index);

/*
 * da_pop — remove the last element, optionally writing it to `out`.
 *
 * Parameters
 *   da  : the array; must not be NULL.
 *   out : if not NULL, `elem_size` bytes are copied here before removal.
 *
 * Returns ZF_OK or ZF_ERR_EMPTY.
 */
ZF_Status da_pop(DynamicArray *da, void *out);

/*
 * da_clear — set length to zero without freeing the allocation.
 *
 * The capacity and backing buffer are preserved so the array can be
 * refilled without a new allocation.  Existing bytes in the buffer are
 * NOT zeroed (treat them as uninitialised after a clear).
 *
 * Parameters
 *   da : must not be NULL.
 */
void da_clear(DynamicArray *da);

/* ───────────────────────────────────────────────────────────────────
 * Access
 * ─────────────────────────────────────────────────────────────────── */

/*
 * da_get — copy the element at `index` into the buffer pointed to by `out`.
 *
 * `out` must point to at least `elem_size` writable bytes.
 *
 * Returns ZF_OK or ZF_ERR_INDEX.
 */
ZF_Status da_get(const DynamicArray *da, size_t index, void *out);

/*
 * da_set — overwrite the element at `index` with a copy of `elem`.
 *
 * Returns ZF_OK or ZF_ERR_INDEX.
 */
ZF_Status da_set(DynamicArray *da, size_t index, const void *elem);

/*
 * da_ptr — return a raw pointer to the element at `index`.
 *
 * The pointer is valid only until the next mutation (append/insert/remove).
 * Returns NULL if index >= length.
 *
 * Use this for read-only access when avoiding a copy is important.
 */
void *da_ptr(const DynamicArray *da, size_t index);

/* ───────────────────────────────────────────────────────────────────
 * Accessors
 * ─────────────────────────────────────────────────────────────────── */

/* Return the number of elements currently stored. */
size_t da_length(const DynamicArray *da);

/* Return the number of element slots in the current allocation. */
size_t da_capacity(const DynamicArray *da);

#endif /* ZEROFORM_H */

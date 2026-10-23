/*
 * zeroform.c — Core data-structure implementation for ZEROFORM
 *
 * Stage 1: DynamicArray — a generic, heap-backed, resizable array.
 *
 * Design notes
 * ────────────
 * • Standard C11 only.  No third-party libraries, no C++ containers.
 * • Memory operations: malloc, realloc, free, memcpy, memmove, memset.
 * • Every public function validates its inputs and returns a ZF_Status.
 * • No global mutable state.
 * • Variable-length arrays (VLAs) are NOT used for storage.
 *
 * Inline commentary is intentionally detailed so a student can follow
 * every decision during a viva.
 */

#include "zeroform.h"

#include <stdlib.h>   /* malloc, realloc, free          */
#include <string.h>   /* memcpy, memmove, memset        */
#include <stdint.h>   /* SIZE_MAX                       */
#include <stdio.h>    /* fprintf (error diagnostics)    */

/* ═══════════════════════════════════════════════════════════════════
 * Internal helpers — not exposed in the header
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * elem_ptr — return a byte pointer to the start of element `index`.
 *
 * This is the core addressing calculation: base + index * elem_size.
 * We cast to unsigned char* because pointer arithmetic on void* is
 * a GCC extension; char* arithmetic is defined in the C standard.
 *
 * Pre-condition: da->data != NULL  (caller must check)
 */
static inline unsigned char *elem_ptr(const DynamicArray *da, size_t index)
{
    return (unsigned char *)da->data + index * da->elem_size;
}

/*
 * checked_mul — multiply two size_t values and detect overflow.
 *
 * Sets *result = a * b and returns 0 on success, -1 on overflow.
 * Uses the standard safe-multiply technique: if a != 0 and
 * result/a != b then overflow occurred.
 */
static int checked_mul(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) {
        *result = 0;
        return 0;
    }
    if (a > SIZE_MAX / b) {
        return -1;   /* overflow */
    }
    *result = a * b;
    return 0;
}

/*
 * da_grow — ensure the array can hold at least `min_cap` elements.
 *
 * Growth strategy: double the current capacity until it meets min_cap,
 * starting from 4 if the array has never been allocated.
 *
 * If realloc fails we leave da->data pointing at the OLD allocation
 * and return ZF_ERR_NOMEM — the array stays in a valid state.
 *
 * If the computed byte count would overflow SIZE_MAX we return
 * ZF_ERR_OVERFLOW before attempting any allocation.
 *
 * Returns ZF_OK, ZF_ERR_NOMEM, or ZF_ERR_OVERFLOW.
 */
static ZF_Status da_grow(DynamicArray *da, size_t min_cap)
{
    /* Start from 4 so we never iterate more than ~60 times for any
     * realistic capacity on a 64-bit machine. */
    size_t new_cap = (da->capacity == 0) ? 4 : da->capacity;

    /* Double until we reach min_cap. */
    while (new_cap < min_cap) {
        /* Check that doubling won't overflow size_t. */
        if (new_cap > SIZE_MAX / 2) {
            /* We can't double further; if SIZE_MAX itself is enough, use it,
             * otherwise overflow. */
            if (SIZE_MAX >= min_cap) {
                new_cap = SIZE_MAX;
                break;
            }
            return ZF_ERR_OVERFLOW;
        }
        new_cap *= 2;
    }

    /* Compute new byte count and check overflow. */
    size_t new_bytes;
    if (checked_mul(new_cap, da->elem_size, &new_bytes) != 0) {
        return ZF_ERR_OVERFLOW;
    }

    /*
     * realloc may return NULL without freeing the original block.
     * We store the result in a temporary pointer so that if it fails,
     * da->data still points at the original valid allocation.
     */
    void *new_data = realloc(da->data, new_bytes);
    if (new_data == NULL) {
        /* Original da->data is unchanged and still valid. */
        return ZF_ERR_NOMEM;
    }

    da->data     = new_data;
    da->capacity = new_cap;
    return ZF_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 * ZF_Status string helper
 * ═══════════════════════════════════════════════════════════════════ */

const char *zf_status_str(ZF_Status s)
{
    switch (s) {
        case ZF_OK:            return "ZF_OK";
        case ZF_ERR_NOMEM:     return "ZF_ERR_NOMEM";
        case ZF_ERR_INDEX:     return "ZF_ERR_INDEX";
        case ZF_ERR_EMPTY:     return "ZF_ERR_EMPTY";
        case ZF_ERR_PARAM:     return "ZF_ERR_PARAM";
        case ZF_ERR_OVERFLOW:  return "ZF_ERR_OVERFLOW";
        case ZF_ERR_DUPLICATE: return "ZF_ERR_DUPLICATE";
        case ZF_ERR_NOT_FOUND: return "ZF_ERR_NOT_FOUND";
        case ZF_ERR_VALUE:     return "ZF_ERR_VALUE";
        case ZF_ERR_DIMENSION: return "ZF_ERR_DIMENSION";
        default:               return "ZF_ERR_UNKNOWN";
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status da_init(DynamicArray *da, size_t elem_size, size_t init_capacity)
{
    if (da == NULL) {
        return ZF_ERR_PARAM;
    }
    /* An element size of zero makes pointer arithmetic undefined and is
     * a usage error — the caller doesn't know what they're storing. */
    if (elem_size == 0) {
        /* Put the struct in a safe inert state before returning. */
        da->data      = NULL;
        da->length    = 0;
        da->capacity  = 0;
        da->elem_size = 0;
        return ZF_ERR_PARAM;
    }

    da->data      = NULL;
    da->length    = 0;
    da->capacity  = 0;
    da->elem_size = elem_size;

    /* Optional pre-allocation: avoids the first few reallocations when
     * the caller already knows the expected size. */
    if (init_capacity > 0) {
        size_t bytes;
        if (checked_mul(init_capacity, elem_size, &bytes) != 0) {
            return ZF_ERR_OVERFLOW;
        }
        da->data = malloc(bytes);
        if (da->data == NULL) {
            return ZF_ERR_NOMEM;
        }
        da->capacity = init_capacity;
    }

    return ZF_OK;
}

void da_destroy(DynamicArray *da)
{
    if (da == NULL) {
        return;
    }
    /* free(NULL) is defined to be a no-op by the C standard (§7.22.3.3),
     * so this is safe even if capacity was 0 and data was never allocated. */
    free(da->data);

    /* Zero every field so the struct can't be accidentally reused
     * without a new da_init() call. */
    da->data      = NULL;
    da->length    = 0;
    da->capacity  = 0;
    da->elem_size = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Mutation
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status da_append(DynamicArray *da, const void *elem)
{
    if (da == NULL || elem == NULL) {
        return ZF_ERR_PARAM;
    }

    /*
     * Self-referential append guard
     * ─────────────────────────────
     * If `elem` points into da->data (e.g. the caller passes a pointer
     * they obtained via da_ptr()), we must copy its value BEFORE
     * potentially moving the backing buffer via realloc.
     *
     * We detect this by checking whether the source pointer falls
     * within [da->data, da->data + length * elem_size).
     *
     * We copy onto the C stack.  elem_size is bounded by the maximum
     * object size (~SIZE_MAX/2) but for practical matrix elements
     * (double, Triplet) this is at most ~24 bytes.  For safety we
     * use a 256-byte stack buffer; if elem_size exceeds that we heap-
     * allocate a temporary, avoiding a VLA.
     */
    const unsigned char *src       = (const unsigned char *)elem;
    const unsigned char *buf_start = (const unsigned char *)da->data;
    const unsigned char *buf_end   = buf_start + da->length * da->elem_size;

    /* is_self: 1 if elem lives inside the current backing buffer */
    int is_self = (da->data != NULL)
               && (src >= buf_start)
               && (src <  buf_end);

    /* Temporary storage for the self-referential case. */
    unsigned char  stack_buf[256];
    unsigned char *tmp_buf = NULL;

    if (is_self) {
        if (da->elem_size <= sizeof(stack_buf)) {
            /* Fast path: copy onto the stack. */
            memcpy(stack_buf, elem, da->elem_size);
            elem = stack_buf;
        } else {
            /* Slow path: heap-allocate a temporary copy. */
            tmp_buf = (unsigned char *)malloc(da->elem_size);
            if (tmp_buf == NULL) {
                return ZF_ERR_NOMEM;
            }
            memcpy(tmp_buf, elem, da->elem_size);
            elem = tmp_buf;
        }
    }

    /* Grow if needed.  da->data may change here, but elem now points
     * at our local copy so it remains valid. */
    if (da->length == da->capacity) {
        /* Need at least length + 1 slots. */
        if (da->length == SIZE_MAX) {
            free(tmp_buf);
            return ZF_ERR_OVERFLOW;
        }
        ZF_Status st = da_grow(da, da->length + 1);
        if (st != ZF_OK) {
            free(tmp_buf);
            return st;
        }
    }

    /* Copy the element bytes into the next slot. */
    memcpy(elem_ptr(da, da->length), elem, da->elem_size);
    da->length++;

    free(tmp_buf);   /* NULL-safe */
    return ZF_OK;
}

ZF_Status da_insert(DynamicArray *da, size_t index, const void *elem)
{
    if (da == NULL || elem == NULL) {
        return ZF_ERR_PARAM;
    }
    /* Valid insertion range is [0, length].  index == length means append. */
    if (index > da->length) {
        return ZF_ERR_INDEX;
    }

    /* Inserting at the end is just an append — that function already
     * handles self-referential inputs and growth. */
    if (index == da->length) {
        return da_append(da, elem);
    }

    /* For interior inserts we also guard against the self-referential case
     * before growing (which may move da->data). */
    const unsigned char *src       = (const unsigned char *)elem;
    const unsigned char *buf_start = (const unsigned char *)da->data;
    const unsigned char *buf_end   = buf_start + da->length * da->elem_size;
    int is_self = (da->data != NULL)
               && (src >= buf_start)
               && (src <  buf_end);

    unsigned char  stack_buf[256];
    unsigned char *tmp_buf = NULL;

    if (is_self) {
        if (da->elem_size <= sizeof(stack_buf)) {
            memcpy(stack_buf, elem, da->elem_size);
            elem = stack_buf;
        } else {
            tmp_buf = (unsigned char *)malloc(da->elem_size);
            if (tmp_buf == NULL) return ZF_ERR_NOMEM;
            memcpy(tmp_buf, elem, da->elem_size);
            elem = tmp_buf;
        }
    }

    /* Grow before shifting; da->data may change. */
    if (da->length == da->capacity) {
        if (da->length == SIZE_MAX) {
            free(tmp_buf);
            return ZF_ERR_OVERFLOW;
        }
        ZF_Status st = da_grow(da, da->length + 1);
        if (st != ZF_OK) {
            free(tmp_buf);
            return st;
        }
    }

    /*
     * Shift elements [index, length) one slot to the right to open a gap.
     * memmove handles overlapping regions correctly.
     *
     * Byte count: (length - index) * elem_size
     * Source:     elem_ptr(da, index)
     * Destination: elem_ptr(da, index + 1)
     */
    size_t n_shift = da->length - index;
    memmove(elem_ptr(da, index + 1),
            elem_ptr(da, index),
            n_shift * da->elem_size);

    /* Write the new element into the now-empty slot at `index`. */
    memcpy(elem_ptr(da, index), elem, da->elem_size);
    da->length++;

    free(tmp_buf);
    return ZF_OK;
}

ZF_Status da_remove(DynamicArray *da, size_t index)
{
    if (da == NULL) {
        return ZF_ERR_PARAM;
    }
    /* An empty array has no valid element indexes. */
    if (da->length == 0 || index >= da->length) {
        return ZF_ERR_INDEX;
    }

    /*
     * Shift elements [index+1, length) one slot to the left.
     * The last slot becomes "unused" (length decreases by one).
     * We do NOT zero the vacated slot; its bytes are undefined.
     */
    size_t n_shift = da->length - index - 1;
    if (n_shift > 0) {
        memmove(elem_ptr(da, index),
                elem_ptr(da, index + 1),
                n_shift * da->elem_size);
    }

    da->length--;
    return ZF_OK;
}

ZF_Status da_pop(DynamicArray *da, void *out)
{
    if (da == NULL) {
        return ZF_ERR_PARAM;
    }
    if (da->length == 0) {
        return ZF_ERR_EMPTY;
    }

    da->length--;

    /* Copy the value out BEFORE decrementing is already done above;
     * elem_ptr(da, da->length) is now the vacated slot. */
    if (out != NULL) {
        memcpy(out, elem_ptr(da, da->length), da->elem_size);
    }

    return ZF_OK;
}

void da_clear(DynamicArray *da)
{
    if (da == NULL) {
        return;
    }
    /* Reset the logical length to zero; the physical allocation survives. */
    da->length = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Access
 * ═══════════════════════════════════════════════════════════════════ */

ZF_Status da_get(const DynamicArray *da, size_t index, void *out)
{
    if (da == NULL || out == NULL) {
        return ZF_ERR_PARAM;
    }
    if (index >= da->length) {
        return ZF_ERR_INDEX;
    }
    memcpy(out, elem_ptr(da, index), da->elem_size);
    return ZF_OK;
}

ZF_Status da_set(DynamicArray *da, size_t index, const void *elem)
{
    if (da == NULL || elem == NULL) {
        return ZF_ERR_PARAM;
    }
    if (index >= da->length) {
        return ZF_ERR_INDEX;
    }
    memcpy(elem_ptr(da, index), elem, da->elem_size);
    return ZF_OK;
}

void *da_ptr(const DynamicArray *da, size_t index)
{
    if (da == NULL || index >= da->length) {
        return NULL;
    }
    return elem_ptr(da, index);
}

/* ═══════════════════════════════════════════════════════════════════
 * Accessors
 * ═══════════════════════════════════════════════════════════════════ */

size_t da_length(const DynamicArray *da)
{
    return (da != NULL) ? da->length : 0;
}

size_t da_capacity(const DynamicArray *da)
{
    return (da != NULL) ? da->capacity : 0;
}

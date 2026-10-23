# =============================================================================
# Makefile — ZEROFORM Sparse Matrix Studio  (Stage 3: Matrix Operations & Analysis)
#
# Tool chain: GNU Make 4.4.1  +  GCC 15.2.0 (x86_64-w64-mingw32)
# Platform:   Windows 11 x64 via MSYS2 / mingw64
#
# Invocation (from the c_engine/ directory):
#   mingw32-make build      — compile all library object files
#   mingw32-make test       — build & run all test suites
#   mingw32-make sanitize   — build & run all tests with -ftrapv
#   mingw32-make clean      — remove all build artifacts
#
# NOTE on sanitizers
# ──────────────────
# AddressSanitizer and UndefinedBehaviorSanitizer require libasan.a /
# libubsan.a. These libraries are NOT shipped in this MSYS2 mingw64
# toolchain (confirmed: ld reports "cannot find -lasan" and
# "cannot find -lubsan"). The `sanitize` target compiles with
# -ftrapv (signed integer overflow trap), which instruments signed arithmetic
# to trap at runtime without requiring any extra runtime library.
# =============================================================================

# --- Compiler settings -------------------------------------------------------
CC       := gcc
CFLAGS   := -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
OPTFLAGS := -O2

# Library source and object files
LIB_SRCS := zeroform.c dense_matrix.c sparse_matrix.c matrix_operations.c memory_analysis.c
LIB_OBJS := zeroform.o dense_matrix.o sparse_matrix.o matrix_operations.o memory_analysis.o
HEADERS  := zeroform.h matrix.h matrix_operations.h memory_analysis.h

# Test sources and executables
TEST_DA_SRC   := tests/test_dynamic_array.c
TEST_DA_EXE   := tests/test_dynamic_array.exe
TEST_DA_TRAPV := tests/test_da_trapv.exe

TEST_MAT_SRC   := tests/test_matrices.c
TEST_MAT_EXE   := tests/test_matrices.exe
TEST_MAT_TRAPV := tests/test_matrices_trapv.exe

TEST_OPS_SRC   := tests/test_operations.c
TEST_OPS_EXE   := tests/test_operations.exe
TEST_OPS_TRAPV := tests/test_operations_trapv.exe

TEST_MA_SRC   := tests/test_memory_analysis.c
TEST_MA_EXE   := tests/test_memory_analysis.exe
TEST_MA_TRAPV := tests/test_memory_analysis_trapv.exe

ALL_TEST_EXES   := $(TEST_DA_EXE) $(TEST_MAT_EXE) $(TEST_OPS_EXE) $(TEST_MA_EXE)
ALL_TRAPV_EXES  := $(TEST_DA_TRAPV) $(TEST_MAT_TRAPV) $(TEST_OPS_TRAPV) $(TEST_MA_TRAPV)

# Executable
EXE := zeroform.exe

# --- Phony targets -----------------------------------------------------------
.PHONY: all build test sanitize clean

# --- Default target ----------------------------------------------------------
all: build

# --- build -------------------------------------------------------------------
# Compile library objects and CLI engine executable with strict C11 flags and -O2.
build: $(LIB_OBJS) $(EXE)
	@echo "  [OK] All library objects and $(EXE) compiled successfully."

$(EXE): src/main.c $(LIB_OBJS) $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) src/main.c $(LIB_OBJS) -o $(EXE)
	@echo "  [OK] Binary built: $(EXE)"

zeroform.o: zeroform.c zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) -c zeroform.c -o zeroform.o
	@echo "  [OK] zeroform.o compiled."

dense_matrix.o: dense_matrix.c matrix.h zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) -c dense_matrix.c -o dense_matrix.o
	@echo "  [OK] dense_matrix.o compiled."

sparse_matrix.o: sparse_matrix.c matrix.h zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) -c sparse_matrix.c -o sparse_matrix.o
	@echo "  [OK] sparse_matrix.o compiled."

matrix_operations.o: matrix_operations.c matrix_operations.h matrix.h zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) -c matrix_operations.c -o matrix_operations.o
	@echo "  [OK] matrix_operations.o compiled."

memory_analysis.o: memory_analysis.c memory_analysis.h matrix.h zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) -c memory_analysis.c -o memory_analysis.o
	@echo "  [OK] memory_analysis.o compiled."

# --- test --------------------------------------------------------------------
# Build and run all test suites (Stage 1, Stage 2, and Stage 3).
# Exits with nonzero status if any test fails.
test: $(ALL_TEST_EXES)
	@echo ""
	@echo "Running DynamicArray tests (Stage 1) ..."
	@echo ""
	$(TEST_DA_EXE)
	@echo ""
	@echo "Running DenseMatrix & SparseMatrix tests (Stage 2) ..."
	@echo ""
	$(TEST_MAT_EXE)
	@echo ""
	@echo "Running Matrix Operations tests (Stage 3) ..."
	@echo ""
	$(TEST_OPS_EXE)
	@echo ""
	@echo "Running Memory Analysis tests (Stage 3) ..."
	@echo ""
	$(TEST_MA_EXE)

$(TEST_DA_EXE): $(TEST_DA_SRC) zeroform.o
	$(CC) $(CFLAGS) $(OPTFLAGS) $(TEST_DA_SRC) zeroform.o -o $(TEST_DA_EXE)
	@echo "  [OK] Test binary built: $(TEST_DA_EXE)"

$(TEST_MAT_EXE): $(TEST_MAT_SRC) zeroform.o dense_matrix.o sparse_matrix.o
	$(CC) $(CFLAGS) $(OPTFLAGS) $(TEST_MAT_SRC) zeroform.o dense_matrix.o sparse_matrix.o -o $(TEST_MAT_EXE)
	@echo "  [OK] Test binary built: $(TEST_MAT_EXE)"

$(TEST_OPS_EXE): $(TEST_OPS_SRC) $(LIB_OBJS)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(TEST_OPS_SRC) $(LIB_OBJS) -o $(TEST_OPS_EXE)
	@echo "  [OK] Test binary built: $(TEST_OPS_EXE)"

$(TEST_MA_EXE): $(TEST_MA_SRC) $(LIB_OBJS)
	$(CC) $(CFLAGS) $(OPTFLAGS) $(TEST_MA_SRC) $(LIB_OBJS) -o $(TEST_MA_EXE)
	@echo "  [OK] Test binary built: $(TEST_MA_EXE)"

# --- sanitize ----------------------------------------------------------------
# libasan and libubsan are absent on MSYS2 mingw64.
# Use -ftrapv to catch signed overflow UB at runtime.
sanitize: $(ALL_TRAPV_EXES)
	@echo ""
	@echo "SANITIZE NOTE: libasan and libubsan are absent on this MSYS2 mingw64"
	@echo "toolchain. Using -ftrapv (signed-overflow trap) as a supported"
	@echo "alternative. Catches signed integer overflow at runtime."
	@echo ""
	@echo "Running DynamicArray tests with -ftrapv ..."
	@echo ""
	$(TEST_DA_TRAPV)
	@echo ""
	@echo "Running DenseMatrix & SparseMatrix tests with -ftrapv ..."
	@echo ""
	$(TEST_MAT_TRAPV)
	@echo ""
	@echo "Running Matrix Operations tests with -ftrapv ..."
	@echo ""
	$(TEST_OPS_TRAPV)
	@echo ""
	@echo "Running Memory Analysis tests with -ftrapv ..."
	@echo ""
	$(TEST_MA_TRAPV)

$(TEST_DA_TRAPV): $(TEST_DA_SRC) zeroform.c zeroform.h
	$(CC) $(CFLAGS) -O1 -ftrapv zeroform.c $(TEST_DA_SRC) -o $(TEST_DA_TRAPV)
	@echo "  [OK] -ftrapv binary built: $(TEST_DA_TRAPV)"

$(TEST_MAT_TRAPV): $(TEST_MAT_SRC) zeroform.c dense_matrix.c sparse_matrix.c matrix.h zeroform.h
	$(CC) $(CFLAGS) -O1 -ftrapv zeroform.c dense_matrix.c sparse_matrix.c $(TEST_MAT_SRC) -o $(TEST_MAT_TRAPV)
	@echo "  [OK] -ftrapv binary built: $(TEST_MAT_TRAPV)"

$(TEST_OPS_TRAPV): $(TEST_OPS_SRC) $(LIB_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -O1 -ftrapv $(LIB_SRCS) $(TEST_OPS_SRC) -o $(TEST_OPS_TRAPV)
	@echo "  [OK] -ftrapv binary built: $(TEST_OPS_TRAPV)"

$(TEST_MA_TRAPV): $(TEST_MA_SRC) $(LIB_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -O1 -ftrapv $(LIB_SRCS) $(TEST_MA_SRC) -o $(TEST_MA_TRAPV)
	@echo "  [OK] -ftrapv binary built: $(TEST_MA_TRAPV)"

# --- clean -------------------------------------------------------------------
# Remove all generated files. Source files are never deleted.
clean:
	-del /Q $(LIB_OBJS) 2>NUL
	-del /Q $(EXE) 2>NUL
	-del /Q tests\test_dynamic_array.exe 2>NUL
	-del /Q tests\test_matrices.exe 2>NUL
	-del /Q tests\test_operations.exe 2>NUL
	-del /Q tests\test_memory_analysis.exe 2>NUL
	-del /Q tests\test_da_trapv.exe 2>NUL
	-del /Q tests\test_matrices_trapv.exe 2>NUL
	-del /Q tests\test_operations_trapv.exe 2>NUL
	-del /Q tests\test_memory_analysis_trapv.exe 2>NUL
	@echo "  [OK] Build artifacts removed."

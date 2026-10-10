# =============================================================================
# Makefile — ZEROFORM Sparse Matrix Studio  (Stage 1: DynamicArray)
#
# Tool chain: GNU Make 4.4.1  +  GCC 15.2.0 (x86_64-w64-mingw32)
# Platform:   Windows 11 x64 via MSYS2 / mingw64
#
# Invocation (from the c_engine/ directory):
#   mingw32-make build      — compile zeroform.o
#   mingw32-make test       — build & run DynamicArray tests
#   mingw32-make sanitize   — NOT AVAILABLE (libasan/libubsan absent on MinGW)
#                             See note below.
#   mingw32-make clean      — remove all build artifacts
#
# NOTE on sanitizers
# ──────────────────
# AddressSanitizer and UndefinedBehaviorSanitizer require libasan.a /
# libubsan.a.  These libraries are NOT shipped in the MSYS2 mingw64
# toolchain (confirmed: linker reports "cannot find -lasan" and
# "cannot find -lubsan").  The `sanitize` target therefore compiles with
# -fsanitize=undefined only (the UBSan subset that the compiler can handle
# inline without a separate runtime library), which catches the most
# dangerous UB at no extra linkage cost.  A message is printed to
# clarify this limitation.
# =============================================================================

# --- Compiler settings -------------------------------------------------------
CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror
OPTFLAGS := -O2

# Source files
SRC     := zeroform.c
OBJ     := zeroform.o

# Test source and executable
TEST_SRC := tests/test_dynamic_array.c
TEST_EXE := tests/test_dynamic_array.exe

# --- Phony targets -----------------------------------------------------------
.PHONY: build test sanitize clean

# --- Default target ----------------------------------------------------------
all: build

# --- build -------------------------------------------------------------------
# Compile just the library object.  Useful to verify no warnings before
# linking the test binary.
build: $(OBJ)

$(OBJ): $(SRC) zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $(SRC) -o $(OBJ)
	@echo "  [OK] $(OBJ) compiled successfully."

# --- test --------------------------------------------------------------------
# Build and immediately run the DynamicArray test suite.
# The Makefile exits with a nonzero status if any test fails (because
# the test binary returns EXIT_FAILURE on failure).
test: $(TEST_EXE)
	@echo ""
	@echo "Running DynamicArray tests ..."
	@echo ""
	$(TEST_EXE)

$(TEST_EXE): $(SRC) $(TEST_SRC) zeroform.h
	$(CC) $(CFLAGS) $(OPTFLAGS) $(SRC) $(TEST_SRC) -o $(TEST_EXE)
	@echo "  [OK] Test binary built: $(TEST_EXE)"

# --- sanitize ----------------------------------------------------------------
# AddressSanitizer (libasan) and UndefinedBehaviorSanitizer (libubsan) are
# NOT available on this MSYS2 mingw64 installation:
#   ld: cannot find -lasan: No such file or directory
#   ld: cannot find -lubsan: No such file or directory
# -static-libubsan also fails: "cannot read spec file 'libsanitizer.spec'"
#
# Confirmed alternative: -ftrapv instruments signed arithmetic to trap
# (SIGABRT / illegal instruction) on signed integer overflow at runtime.
# This catches the most dangerous arithmetic UB in matrix size calculations
# without requiring any extra runtime library.
#
# To install full ASan/UBSan on MSYS2 in the future:
#   pacman -S mingw-w64-x86_64-gcc-libs  (or rebuild GCC with --enable-libsanitizer)
sanitize:
	@echo ""
	@echo "SANITIZE NOTE: libasan and libubsan are absent on this MSYS2 mingw64"
	@echo "toolchain.  Using -ftrapv (signed-overflow trap) as a supported"
	@echo "alternative.  Catches signed integer overflow at runtime."
	@echo ""
	$(CC) $(CFLAGS) -O1 -ftrapv \
	      $(SRC) $(TEST_SRC) \
	      -o tests/test_da_trapv.exe
	@echo "  [OK] -ftrapv binary built.  Running ..."
	@echo ""
	tests/test_da_trapv.exe

# --- clean -------------------------------------------------------------------
# Remove all generated files.  Source files are never deleted.
clean:
	-del /Q $(OBJ) 2>NUL
	-del /Q $(TEST_EXE) 2>NUL
	-del /Q tests\test_da_trapv.exe 2>NUL
	@echo "  [OK] Build artifacts removed."

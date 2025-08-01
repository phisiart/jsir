# JSIR - JavaScript Intermediate Representation

JSIR is an MLIR-based JavaScript intermediate representation.

For more information on MLIR, see the MLIR homepage.

## Getting started

### Install build tools

We have only tested `clang` on Linux:

```shell
# Install clang:

sudo apt update
sudo apt install clang
```

We use the `Bazel` build system.
It is recommended to use `Bazelisk` to manage `Bazel` versions:

```shell
# Install Bazelisk through npm:

sudo apt install npm
sudo npm install -g @bazel/bazelisk
```

### Build

Note: The build takes a lot of storage space.
If you run out of space, Bazel will return a cryptic error.

LLVM takes a long time to fetch and build.
We can test if LLVM is properly included by building a part of it:

```shell
# This will fetch LLVM and build its support library:

bazelisk build @llvm-project//llvm:Support
```

To build JSIR:

```shell
# Build everything:
bazelisk build //...

# Or, build a single target:
bazelisk build //maldoca/js/ir:jsir_gen

# Or, build all targets in a directory:
bazelisk build //maldoca/js/ir/...
```

### Test

To run test cases:

```shell
# Run all tests:
bazelisk test //...

# Or, run a specific test:
bazelisk test //maldoca/js/quickjs:quickjs_test

# Or, run all tests under a directory:
bazelisk test //maldoca/js/ir/conversion/...
```

### Run the `jsir_gen` tool

Convert a JavaScript source file to JSHIR:

```shell
bazelisk run //maldoca/js/ir:jsir_gen --\
    --input_file=$(pwd)/maldoca/js/ir/conversion/tests/if_statement/input.js \
    --passes=source2ast,ast2hir
```

## Overview

NOTE: Documentation is under construction.

JSIR comes with two dialects:

### JSHIR

This is a high-level IR that uses MLIR regions to accurately model control flow
structures.

### JSLIR

This is a low-level IR that uses CFGs to represent branching behaviors. JSLIR
adds extra operations to annotate the kind of original control flow structures.
This allows JSLIR to be fully converted back to JSHIR.

## DISCLAIMER
This is not an official Google product.

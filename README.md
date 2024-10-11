# JSIR - JavaScript Intermediate Representation

JSIR is an MLIR-based JavaScript intermediate representation.

For more information on MLIR, see the MLIR homepage.

JSIR comes with two dialects:

## JSHIR

This is a high-level IR that uses MLIR regions to accurately model control flow
structures.

## JSLIR

This is a low-level IR that uses CFGs to represent branching behaviors. JSLIR
adds extra operations to annotate the kind of original control flow structures.
This allows JSLIR to be fully converted back to JSHIR.

## Build

```
bazel build //...
```

## DISCLAIMER
This is not an official Google product.

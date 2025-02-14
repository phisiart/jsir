// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MALDOCA_JS_IR_CONVERSION_UTILS_H_
#define MALDOCA_JS_IR_CONVERSION_UTILS_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "absl/status/statusor.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// context must contain the dialects
//  - JsirDialect
//  - JshirDialect
//  - JslirDialect
//  - mlir::func::FuncDialect
absl::StatusOr<mlir::OwningOpRef<JsirFileOp>> AstToJshirFile(
    const JsFile &ast, mlir::MLIRContext &context);

mlir::OwningOpRef<JsirFileOp> JshirFileToJslir(JsirFileOp hir_file);

mlir::OwningOpRef<JsirFileOp> JslirFileToJshir(JsirFileOp lir_file);

absl::StatusOr<std::unique_ptr<JsFile>> JshirFileToAst(JsirFileOp hir_file);

void LoadNecessaryDialects(mlir::MLIRContext &context);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_UTILS_H_

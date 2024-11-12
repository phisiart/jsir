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

#include "maldoca/js/ir/conversion/utils.h"

#include <memory>

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "absl/status/statusor.h"
#include "maldoca/base/ret_check.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/conversion/ast_to_jsir.h"
#include "maldoca/js/ir/conversion/jshir_to_jslir.h"
#include "maldoca/js/ir/conversion/jsir_to_ast.h"
#include "maldoca/js/ir/conversion/jslir_to_jshir.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

absl::StatusOr<mlir::OwningOpRef<JsirFileOp>> AstToJshirFile(
    const JsFile &ast, mlir::MLIRContext &context) {
  // Check for all the dialects
  MALDOCA_RET_CHECK_NE(context.getLoadedDialect<JsirDialect>(), nullptr);
  MALDOCA_RET_CHECK_NE(context.getLoadedDialect<JshirDialect>(), nullptr);
  MALDOCA_RET_CHECK_NE(context.getLoadedDialect<JslirDialect>(), nullptr);
  MALDOCA_RET_CHECK_NE(context.getLoadedDialect<mlir::func::FuncDialect>(),
                       nullptr);

  mlir::OpBuilder builder(&context);
  AstToJsir ast_to_jsir(builder);
  mlir::OwningOpRef<JsirFileOp> hir_file = ast_to_jsir.VisitFile(&ast);

  MALDOCA_RET_CHECK(mlir::verify(*hir_file).succeeded());

  return hir_file;
}

mlir::OwningOpRef<JsirFileOp> JshirFileToJslir(JsirFileOp hir_file) {
  mlir::OpBuilder builder(hir_file.getContext());
  JshirToJslir jshir_to_jslir(builder);

  mlir::OwningOpRef<JsirFileOp> lir_file =
      builder.cloneWithoutRegions(hir_file);
  jshir_to_jslir.VisitRegion(&hir_file.getProgram(), &lir_file->getProgram());

  return lir_file;
}

mlir::OwningOpRef<JsirFileOp> JslirFileToJshir(JsirFileOp lir_file) {
  mlir::OpBuilder builder(lir_file->getContext());
  JslirToJshir jslir_to_jshir(&builder);

  mlir::OwningOpRef<JsirFileOp> hir_file =
      builder.cloneWithoutRegions(lir_file);

  if (lir_file.getProgram().empty()) {
    return hir_file;
  }
  mlir::Block &lir_program_block = lir_file.getProgram().front();
  mlir::Block &hir_program_block = hir_file->getProgram().emplaceBlock();

  if (lir_program_block.empty()) {
    return hir_file;
  }
  mlir::Operation *lir_first_op = &lir_program_block.front();

  builder.setInsertionPointToStart(&hir_program_block);
  for (mlir::Operation *lir_op = lir_first_op; lir_op != nullptr;) {
    lir_op = jslir_to_jshir.VisitOperation(lir_op);
  }

  return hir_file;
}

absl::StatusOr<std::unique_ptr<JsFile>> JshirFileToAst(JsirFileOp hir_file) {
  JsirToAst jsir_to_ast;
  return jsir_to_ast.VisitFile(hir_file);
}

void LoadNecessaryDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<maldoca::JsirDialect>();
  context.getOrLoadDialect<maldoca::JshirDialect>();
  context.getOrLoadDialect<maldoca::JslirDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
}

}  // namespace maldoca

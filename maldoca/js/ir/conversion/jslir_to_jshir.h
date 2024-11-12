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

#ifndef MALDOCA_JS_IR_CONVERSION_JSLIR_TO_JSHIR_H_
#define MALDOCA_JS_IR_CONVERSION_JSLIR_TO_JSHIR_H_

#include <optional>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/jslir_visitor.h"

namespace maldoca {

class JslirToJshir : public JslirVisitor<mlir::Operation *> {
 public:
  explicit JslirToJshir(mlir::OpBuilder *builder) : builder_(builder) {}

  // Converts the op and inserts the new op at the current builder insertion
  // point.
  mlir::Operation *VisitOperation(mlir::Operation *lir_op);

  void VisitCondBranch(mlir::cf::CondBranchOp lir_op);

  void VisitBranch(mlir::cf::BranchOp lir_op);

  mlir::Operation *VisitOperationDefault(mlir::Operation *lir_op);

  mlir::Operation *VisitJslirOpDefault(mlir::Operation *lir_op) override {
    return VisitOperationDefault(lir_op);
  }

  mlir::Operation *VisitControlFlowStarter(
      JslirControlFlowStarterOp lir_op) override;

  mlir::Operation *VisitControlFlowMarker(
      JslirControlFlowMarkerOp lir_op) override {
    return nullptr;
  }

  // Given the starting operation of a block-statement in JSLIR, converts all
  // JSLIR operations related to this block-statement into a
  // JshirBlockStatementOp.
  //
  // Returns the next JSLIR operation after the block-statement.
  //
  // %token = jslir.control_flow_starter {BlockStatement}  <----------- `lir_op`
  // jslir.control_flow_marker {BlockStatementDirectives} (%token)
  // ...  // directives
  // jslir.control_flow_marker {BlockStatementBody} (%token)
  // ...  // body
  // jslir.control_flow_marker {BlockStatementEnd} (%token)
  // ...  <-------------------------------------------------------- return value
  mlir::Operation *VisitBlockStatementStart(JslirControlFlowStarterOp lir_op);

  // Given the starting operation of a with-statement in JSLIR, converts all
  // JSLIR operations related to this with-statement into a
  // JshirWithStatementOp.
  //
  // Returns the next JSLIR operation after the with-statement.
  //
  //   %object = ...
  //   %token = jslir.with_statement_start (%object) <----------- `lir_op`
  //   cf.br [^body_bb]
  //
  // ^body_bb:
  //   jslir.control_flow_marker {WithStatementBody} (%token)
  //   ...  // body
  //   cf.br [^end_bb]
  //
  // ^end_bb:
  //   jslir.control_flow_marker {WithStatementEnd} (%token)
  // ...  <-------------------------------------------------------- return value
  mlir::Operation *VisitWithStatementStart(
      JslirWithStatementStartOp lir_op) override;

  // Given the starting operation of a labeled-statement in JSLIR, converts all
  // JSLIR operations related to this labeled-statement into a
  // JshirLabeledStatementOp.
  //
  // Returns the next JSLIR operation after the labeled-statement.
  //
  // %token = jslir.labeled_statement {label} <------------------------ `lir_op`
  // ...  // body
  // jslir.control_flow_marker {LabeledStatementEnd} (%token)
  // ...  <-------------------------------------------------------- return value
  mlir::Operation *VisitLabeledStatementStart(
      JslirLabeledStatementStartOp lir_op) override;

  // Given the starting operation of a if-statement in JSLIR, converts all
  // JSLIR operations related to this if-statement into a JshirIfStatementOp.
  //
  // Returns the next JSLIR operation after the if-statement.
  //
  //   %token = jslir.control_flow_starter {IfStatement} <------------- `lir_op`
  //   cf.cond_br (%test) [^consequent_bb, ^alternate_bb]
  // ^consequent_bb:
  //   jslir.control_flow_marker {IfStatementConsequent} (%token)
  //   ...  // consequent
  //   cf.br [^end_bb]
  //
  // ^alternate_bb:
  //   jslir.control_flow_marker {IfStatementAlternate} (%token)
  //   ...  // alternate
  //   cf.br [^end_bb]
  //
  // ^end_bb:
  //   jslir.control_flow_marker {IfStatementEnd} (%token)
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitIfStatementStart(JslirControlFlowStarterOp lir_op);

  // Given the starting operation of a switch-statement in JSLIR, converts all
  // JSLIR operations related to this switch-statement into a
  // JshirSwitchStatementOp.
  //
  // Returns the next JSLIR operation after the switch-statement.
  //
  //   %switch_token = jslir.switch_statement_start (%discriminant) <-- `lir_op`
  //   cf.br()[^case0_test_bb]
  //
  // ^case0_test_bb:
  //   %case0_token = jslir.switch_statement_case_start {0} (%switch_token)
  //   %test0 = ...
  //   jslir.switch_statement_case_test (%test0)
  //   %cmp0 = jsir.binary_expression {"==="} (%discriminant, %test0)
  //   cond_br(%cmp0) [%case0_consequent_bb, %case1_test_bb]
  //
  // ^case1_test_bb:
  //   %case1_token = jslir.switch_statement_case_start {1} (%switch_token)
  //   %test1 = ...
  //   jslir.switch_statement_case_test (%test1)
  //   %cmp1 = jsir.binary_expression {"==="} (%discriminant, %test1)
  //   cond_br(%cmp1) [%case1_consequent_bb, %default_test_bb]
  //
  // ^default_test_bb:
  //   %default_token = jslir.switch_statement_case_start {2} (%switch_token)
  //   br [^default_consequent_bb]
  //
  // ^case0_consequent_bb:
  //   jslir.control_flow_marker {SwitchStatementCaseBody} (%case0_token)
  //   ...  // body
  //   br [^case1_consequent_bb] (or ^end_bb if break)
  //
  // ^case1_consequent_bb:
  //   jslir.control_flow_marker {SwitchStatementCaseBody} (%case1_token)
  //   ...  // body
  //   br [^default_consequent_bb] (or ^end_bb if break)
  //
  // ^default_consequent_bb:
  //   jslir.control_flow_marker {SwitchStatementCaseBody} (%default_token)
  //   ...  // body
  //   br [^end_bb]
  //
  // ^end_bb:
  //  jslir.control_flow_marker {SwitchStatementEnd} (%switch_token)
  //  ...  <------------------------------------------------------- return value
  mlir::Operation *VisitSwitchStatementStart(
      JslirSwitchStatementStartOp lir_op) override;
  mlir::Operation *VisitSwitchStatementCaseTest(
      JslirSwitchStatementCaseTestOp lir_op) override;

  // Given the starting operation of a try-statement in JSLIR, converts all
  // JSLIR operations related to this try-statement into a JshirTryStatementOp.
  //
  // Returns the next JSLIR operation after the try-statement.
  //
  //   %token = jslir.control_flow_starter {TryStatement} <------------ `lir_op`
  //   cf.br [^body]
  //
  // ^body:
  //   jslir.control_flow_marker {TryStatementBody} (%token)
  //   ...
  //   cf.br [^finalizer]
  //
  // ^handler:
  //   jslir.control_flow_marker {TryStatementHandler} (%token)
  //   %param = ...
  //   jslir.catch_clause_start (%param)
  //   ...
  //   cf.br [^finalizer]
  //
  // ^finalizer:
  //   jslir.control_flow_marker {TryStatementFinalizer} (%token)
  //   ...
  //   cf.br [^end]
  //
  // ^end:
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitTryStatementStart(JslirControlFlowStarterOp lir_op);
  mlir::Operation *VisitCatchClauseStart(
      JslirCatchClauseStartOp lir_op) override;

  // Given the starting operation of a break-statement in JSLIR, converts all
  // JSLIR operations related to this break-statement into a
  // JshirBreakStatementOp.
  //
  // Returns the next JSLIR operation after the break-statement.
  //
  // jslir.break_statement <------------------------------------------- `lir_op`
  // cf.br [^break_location]
  // ...  <-------------------------------------------------------- return value
  mlir::Operation *VisitBreakStatement(JslirBreakStatementOp lir_op) override;

  // Given the starting operation of a continue-statement in JSLIR, converts all
  // JSLIR operations related to this continue-statement into a
  // JshirContinueStatementOp.
  //
  // Returns the next JSLIR operation after the continue-statement.
  //
  // jslir.continue_statement <---------------------------------------- `lir_op`
  // cf.br [^continue_location]
  // ...  <-------------------------------------------------------- return value
  mlir::Operation *VisitContinueStatement(
      JslirContinueStatementOp lir_op) override;

  // Given the starting operation of a while-statement in JSLIR, converts all
  // JSLIR operations related to this while-statement into a
  // JshirWhileStatementOp.
  //
  // Returns the next JSLIR operation after the while-statement.
  //
  //   %token = jslir.control_flow_starter {WhileStatement} <---------- `lir_op`
  //   cf.br [^test_bb]
  //
  // ^test_bb:
  //   jslir.control_flow_marker {WhileStatementTest} (%token)
  //   %test = ...
  //   cf.cond_br (%test) [^body_bb, ^end_bb]
  //
  // ^body_bb:
  //   jslir.control_flow_marker {WhileStatementBody} (%token)
  //   ...  // body
  //   cf.br [^test_bb]
  //
  // ^end_bb:
  //   jslir.control_flow_marker {WhileStatementEnd} (%token)
  // ...  <-------------------------------------------------------- return value
  mlir::Operation *VisitWhileStatementStart(JslirControlFlowStarterOp lir_op);

  // Given the starting operation of a do-while-statement in JSLIR, converts all
  // JSLIR operations related to this do-while-statement into a
  // JshirDoWhileStatementOp.
  //
  // Returns the next JSLIR operation after the do-while-statement.
  //
  //   %token = jslir.control_flow_starter {DoWhileStatement}  <------- `lir_op`
  //   cf.br [^body_bb]
  //
  // ^body_bb:
  //   ...  // body
  //   cf.br [^test_bb]
  //
  // ^test_bb:
  //   jslir.control_flow_marker {DoWhileStatementTest} (%token)
  //   %test = ...
  //   cf.cond_br (%test) [^body_bb, ^end_bb]
  //
  // ^end_bb:
  //   jslir.control_flow_marker {DoWhileStatementEnd} (%token)
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitDoWhileStatementStart(JslirControlFlowStarterOp lir_op);

  // Given the starting operation of a for-statement in JSLIR, converts all
  // JSLIR operations related to this for-statement into a
  // JshirForStatementOp.
  //
  // Returns the next JSLIR operation after the for-statement.
  //
  //   %token = jslir.control_flow_starter {ForStatement}  <----------- `lir_op`
  //   cf.br [^init_bb]
  //
  // ^init_bb:
  //   jslir.control_flow_marker {ForStatementInit} (%token)
  //   %init = ...
  //   cf.br [^test_bb]
  //
  // ^test_bb:
  //   jslir.control_flow_marker {ForStatementTest} (%token)
  //   %test = ...
  //   cf.cond_br (%test) [^body_bb, ^end_bb]
  //
  // ^body_bb:
  //   jslir.control_flow_marker {ForStatementBody} (%token)
  //   ...  // body
  //   cf.br [^update_bb]
  //
  // ^update_bb:
  //   jslir.control_flow_marker {ForStatementUpdate} (%token)
  //   ...  // update
  //   cf.br [^test_bb]
  //
  // ^end_bb:
  //   jslir.control_flow_marker {ForStatementEnd} (%token)
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitForStatementStart(JslirControlFlowStarterOp lir_op);

  // Given the starting operation of a for-{in,of}-statement in JSLIR, converts
  // all JSLIR operations related to this for-{in,of}-statement into a
  // JshirFor{In,Of}StatementOp.
  //
  // Returns the next JSLIR operation after the for-{in,of}-statement.
  //
  // ^init:
  //   ...
  //   %left = ...
  //   %right = ...
  //   %iter = jslir.for_in_statement_start(%left, %right) <----------- `lir_op`
  //                                   - OR -
  //   %iter = jslir.for_of_statement_start(%left, %right) {await}
  //   cf.br()[^next]
  //
  // ^next:  // 2 preds: ^init, ^body
  //   %has_next = jslir.for_in_of_statement_has_next(%iter)
  //   cf.cond_br(%has_next)[^body, ^end]
  //
  // ^body:  // pred: ^next
  //   jslir.for_in_of_statement_get_next(%iter)
  //   ...
  //   cf.br()[^next]
  //
  // ^end:  // pred: ^next
  //   jslir.for_in_of_statement_end (%iter)
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitForInOfStatementStart(
      mlir::Operation *lir_op, mlir::Value lir_iterator, JsirForInOfKind kind,
      mlir::StringAttr left_declaration_kind, mlir::Value lir_left_lval,
      mlir::Value lir_right, std::optional<bool> await);

  mlir::Operation *VisitForInStatementStart(
      JslirForInStatementStartOp lir_op) override;

  mlir::Operation *VisitForOfStatementStart(
      JslirForOfStatementStartOp lir_op) override;

  // Given the starting operation of a logical-expression in JSLIR, converts all
  // JSLIR operations related to this logical-expression into a
  // JshirLogicalExpressionOp.
  //
  // Returns the next JSLIR operation after the logical-expression.
  //
  //   %token = jslir.logical_expression_start {"||"} (%a) <----------- `lir_op`
  //   cf.cond_br(%should_short_circuit) [^end_bb(%a), ^right_bb]
  //
  // ^right_bb:
  //   jslir.control_flow_marker {LogicalExpressionRight} (%token)
  //   %b = ...
  //   cf.br (%b) [^end_bb]
  //
  // ^end_bb(%result):
  //   jslir.control_flow_marker {LogicalExpressionEnd} (%token)
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitLogicalExpressionStart(
      JslirLogicalExpressionStartOp lir_op) override;

  // Given the starting operation of a conditional-expression in JSLIR, converts
  // all JSLIR operations related to this conditional-expression into a
  // JshirConditionalExpressionOp.
  //
  // Returns the next JSLIR operation after the conditional-expression.
  //
  //   ...
  //   %token = jslir.control_flow_starter {ConditionalExpression}  <-- `lir_op`
  //   cf.cond_br(%test) [^alternate_bb, ^consequent_bb]
  //
  // ^alternate_bb:
  //   jslir.control_flow_marker {ConditionalExpressionAlternate} (%token)
  //   %alternate = ...
  //   cf.br (%alternate) [^end_bb]
  //
  // ^consequent_bb:
  //   jslir.control_flow_marker {ConditionalExpressionConsequent} (%token)
  //   %consequent = ...
  //   cf.br (%consequent) [^end_bb]
  //
  // ^end_bb(%result):
  //   jslir.control_flow_marker {ConditionalExpressionEnd} (%token)
  //   ...  <------------------------------------------------------ return value
  mlir::Operation *VisitConditionalExpressionStart(
      JslirControlFlowStarterOp lir_op);

 private:
  // Converts all JSLIR operations from `first_lir_op` to the first "end marker"
  // of the current scope. The converted operations are inserted in
  // `hir_region`.
  //
  // This is done by iteratively visiting each operation from `first_lir_op`.
  // Each `Visit*` function returns the next JSLIR operation.
  void CloneIntoRegion(mlir::Operation *first_lir_op, mlir::Region &hir_region);

  // Mappings of mlir::Values in JSLIR to mlir::Values in JSHIR.
  //
  // This way, when we convert each JSLIR op to a JSHIR op, we know what the
  // JSHIR operands should be given the JSLIR operands.
  mlir::IRMapping lir_to_hir_mappings_;
  // The builder used to generate JSHIR.
  mlir::OpBuilder *builder_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_JSLIR_TO_JSHIR_H_

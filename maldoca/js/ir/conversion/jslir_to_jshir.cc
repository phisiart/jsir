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

#include "maldoca/js/ir/conversion/jslir_to_jshir.h"

#include <cstddef>
#include <optional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/base/status_macros.h"
#include "maldoca/js/ir/ir.h"
#include "maldoca/js/ir/jslir_visitor.h"

namespace maldoca {

mlir::Value GetCondBranchCondition(mlir::cf::CondBranchOp op) {
  // TODO(b/204592400): CondBranchOp::condition must be of type I1.
  mlir::Value condition = op.getConditionMutable().get();

  auto cast_op = condition.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (cast_op != nullptr) {
    CHECK_EQ(cast_op.getInputs().size(), 1);
    condition = cast_op.getInputs().front();
  }

  return condition;
}

absl::StatusOr<mlir::Value> GetCondBranchCondition(mlir::Block *block) {
  if (block->empty()) {
    return absl::InvalidArgumentError("empty block");
  }

  auto op = llvm::dyn_cast<mlir::cf::CondBranchOp>(block->back());
  if (op == nullptr) {
    return absl::InvalidArgumentError("terminator is not cond_br");
  }

  return GetCondBranchCondition(op);
}

mlir::Operation *JslirToJshir::VisitOperation(mlir::Operation *lir_op) {
  llvm::TypeSwitch<mlir::Operation *, mlir::Operation *> type_switch{lir_op};

  AttachJslirVisitor(type_switch, *this);

  return type_switch
      .Case([&](mlir::UnrealizedConversionCastOp lir_op) {
        CHECK_EQ(lir_op.getInputs().size(), lir_op.getOutputs().size());

        for (auto [input, output] :
             llvm::zip_equal(lir_op.getInputs(), lir_op.getOutputs())) {
          auto hir_value = lir_to_hir_mappings_.lookup(input);
          lir_to_hir_mappings_.map(output, hir_value);
        }

        return lir_op->getNextNode();
      })
      .Case([&](mlir::cf::CondBranchOp lir_op) {
        VisitCondBranch(lir_op);
        return nullptr;
      })
      .Case([&](mlir::cf::BranchOp lir_op) {
        VisitBranch(lir_op);
        return nullptr;
      })
      .Default([&](mlir::Operation *lir_op) {
        return VisitOperationDefault(lir_op);
      });
}

void JslirToJshir::VisitCondBranch(mlir::cf::CondBranchOp lir_op) {
  mlir::Value lir_condition = GetCondBranchCondition(lir_op);

  builder_->create<JsirExprRegionEndOp>(
      lir_op.getLoc(), lir_to_hir_mappings_.lookup(lir_condition));
}

void JslirToJshir::VisitBranch(mlir::cf::BranchOp lir_op) {
  switch (lir_op.getDestOperands().size()) {
    case 0:
      // This is the end of an HIR stmt region.
      break;
    case 1:
      mlir::Value lir_dest_operand = lir_op.getDestOperands().front();
      builder_->create<JsirExprRegionEndOp>(
          lir_op.getLoc(), lir_to_hir_mappings_.lookup(lir_dest_operand));
      break;
  }
}

mlir::Operation *JslirToJshir::VisitOperationDefault(mlir::Operation *lir_op) {
  CHECK(lir_op != nullptr);

  auto *hir_op = builder_->cloneWithoutRegions(*lir_op, lir_to_hir_mappings_);

  for (unsigned int i = 0, e = lir_op->getNumRegions(); i != e; ++i) {
    auto &lir_region = lir_op->getRegion(i);
    if (lir_region.empty()) {
      // If there's no block in lir_region, there's nothing to clone.
      continue;
    }

    // Always create a block in hir_block.
    // Even if hir_region only has an empty block, we still want an exact copy.
    auto &hir_region = hir_op->getRegion(i);
    auto &hir_block = hir_region.emplaceBlock();

    auto &lir_entry_block = lir_region.front();
    if (lir_entry_block.empty()) {
      continue;
    }

    mlir::OpBuilder::InsertionGuard insertion_guard(*builder_);
    builder_->setInsertionPointToEnd(&hir_block);
    for (auto *lir_region_op = &lir_entry_block.front();
         lir_region_op != nullptr;) {
      lir_region_op = VisitOperation(lir_region_op);
    }
  }

  return lir_op->getNextNode();
}

mlir::Operation *JslirToJshir::VisitControlFlowStarter(
    JslirControlFlowStarterOp lir_op) {
  switch (lir_op.getKind()) {
    case JsirControlFlowKind::BlockStatement:
      return VisitBlockStatementStart(lir_op);
    case JsirControlFlowKind::IfStatement:
      return VisitIfStatementStart(lir_op);
    case JsirControlFlowKind::TryStatement:
      return VisitTryStatementStart(lir_op);
    case JsirControlFlowKind::WhileStatement:
      return VisitWhileStatementStart(lir_op);
    case JsirControlFlowKind::DoWhileStatement:
      return VisitDoWhileStatementStart(lir_op);
    case JsirControlFlowKind::ForStatement:
      return VisitForStatementStart(lir_op);
    case JsirControlFlowKind::ConditionalExpression:
      return VisitConditionalExpressionStart(lir_op);
    default:
      return nullptr;
  }
}

static JslirControlFlowMarkerOp FindMarker(mlir::Value token,
                                           JsirControlFlowMarkerKind kind) {
  for (mlir::Operation *user : token.getUsers()) {
    auto marker = llvm::dyn_cast<JslirControlFlowMarkerOp>(user);
    if (marker == nullptr) {
      continue;
    }
    if (marker.getKind() == kind) {
      return marker;
    }
  }
  return nullptr;
}

void JslirToJshir::CloneIntoRegion(mlir::Operation *first_lir_op,
                                   mlir::Region &hir_region) {
  if (first_lir_op != nullptr) {
    mlir::Block &hir_directives_block = hir_region.emplaceBlock();
    mlir::OpBuilder::InsertionGuard insertion_guard(*builder_);
    builder_->setInsertionPointToStart(&hir_directives_block);

    for (mlir::Operation *lir_op = first_lir_op; lir_op != nullptr;) {
      lir_op = VisitOperation(lir_op);
    }
  }
}

mlir::Operation *JslirToJshir::VisitBlockStatementStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto directives_marker =
      FindMarker(token, JsirControlFlowMarkerKind::BlockStatementDirectives);
  auto body_marker =
      FindMarker(token, JsirControlFlowMarkerKind::BlockStatementBody);
  auto end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::BlockStatementEnd);

  auto hir_op = builder_->create<JshirBlockStatementOp>(lir_op.getLoc());

  if (directives_marker != nullptr) {
    CloneIntoRegion(directives_marker->getNextNode(), hir_op.getDirectives());
  }

  if (body_marker != nullptr) {
    CloneIntoRegion(body_marker->getNextNode(), hir_op.getBody());
  }

  if (end_marker == nullptr) {
    return nullptr;
  }
  return end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitWithStatementStart(
    JslirWithStatementStartOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_body_marker =
      FindMarker(token, JsirControlFlowMarkerKind::WithStatementBody);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::WithStatementEnd);

  auto lir_object = lir_op.getObject();
  auto hir_op = builder_->create<JshirWithStatementOp>(
      lir_op.getLoc(), lir_to_hir_mappings_.lookup(lir_object));

  if (lir_body_marker != nullptr) {
    CloneIntoRegion(lir_body_marker->getNextNode(), hir_op.getBody());
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitLabeledStatementStart(
    JslirLabeledStatementStartOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::LabeledStatementEnd);

  auto hir_op = builder_->create<JshirLabeledStatementOp>(
      lir_op.getLoc(), lir_op.getLabelAttr());
  CloneIntoRegion(lir_op->getNextNode(), hir_op.getBody());

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitIfStatementStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_consequent_marker =
      FindMarker(token, JsirControlFlowMarkerKind::IfStatementConsequent);
  auto lir_alternate_marker =
      FindMarker(token, JsirControlFlowMarkerKind::IfStatementAlternate);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::IfStatementEnd);

  MALDOCA_ASSIGN_OR_RETURN(mlir::Value lir_test,
                           GetCondBranchCondition(lir_op->getBlock()), nullptr);

  auto hir_op = builder_->create<JshirIfStatementOp>(
      lir_op.getLoc(), lir_to_hir_mappings_.lookup(lir_test));

  if (lir_consequent_marker != nullptr) {
    CloneIntoRegion(lir_consequent_marker->getNextNode(),
                    hir_op.getConsequent());
  }

  if (lir_alternate_marker != nullptr) {
    CloneIntoRegion(lir_alternate_marker->getNextNode(), hir_op.getAlternate());
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitSwitchStatementStart(
    JslirSwitchStatementStartOp lir_op) {
  mlir::Value token = lir_op.getToken();

  struct LirCase {
    size_t index;

    // Source location for the entire case.
    mlir::Location loc;

    // The first op after the JslirSwitchStatementCaseStartOp marker.
    std::optional<mlir::Operation *> lir_test_op;

    // The first op after the SwitchStatementCaseBody or
    // JslirSwitchStatementDefaultStartOp marker.
    mlir::Operation *lir_body_op;
  };

  std::vector<LirCase> lir_cases;
  for (mlir::Operation *op : token.getUsers()) {
    if (auto lir_test_marker =
            llvm::dyn_cast<JslirSwitchStatementCaseStartOp>(op)) {
      auto lir_body_marker = FindMarker(
          lir_test_marker, JsirControlFlowMarkerKind::SwitchStatementCaseBody);

      lir_cases.push_back(LirCase{
          .index = lir_test_marker.getCaseIdx(),
          .loc = lir_test_marker.getLoc(),
          .lir_test_op = lir_test_marker->getNextNode(),
          .lir_body_op = lir_body_marker->getNextNode(),
      });

    } else if (auto lir_default_marker =
                   llvm::dyn_cast<JslirSwitchStatementDefaultStartOp>(op)) {
      lir_cases.push_back(LirCase{
          .index = lir_default_marker.getCaseIdx(),
          .loc = lir_default_marker.getLoc(),
          .lir_test_op = std::nullopt,
          .lir_body_op = lir_default_marker->getNextNode(),
      });
    }
  }

  // Note that the blocks for the cases do not necessarily appear in the
  // original order.
  //
  // ```Source
  // switch (...) {
  //   case test0:
  //     ...
  //   default:
  //     ...
  //   case test1:
  //     ...
  // }
  // ```
  //
  // ```CFG
  //       |
  //       v
  // +-- test0 -------> body0
  // |                    | fallthrough
  // |                    v
  // |  default ------> body2
  // |     ^              | fallthrough
  // |     |              v
  // |   test1 -------> body1
  // |     ^              |
  // |     |              |
  // +-----+              |
  //                      |
  //       +--------------+
  //       |
  //       v
  // ```
  //
  // Therefore, we rely on JslirSwitchStatementCaseStartOp::case_idx to
  // determine the order of the cases.
  absl::c_sort(lir_cases,
               [&](LirCase lhs, LirCase rhs) { return lhs.index < rhs.index; });

  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::SwitchStatementEnd);

  auto hir_discriminant = lir_to_hir_mappings_.lookup(lir_op.getDiscriminant());
  auto hir_op = builder_->create<JshirSwitchStatementOp>(lir_op.getLoc(),
                                                         hir_discriminant);

  mlir::Block *hir_cases_block = &hir_op.getCases().emplaceBlock();
  {
    mlir::OpBuilder::InsertionGuard insertion_guard(*builder_);
    builder_->setInsertionPointToStart(hir_cases_block);

    for (const LirCase &lir_case : lir_cases) {
      auto hir_case_op = builder_->create<JshirSwitchCaseOp>(lir_case.loc);

      if (lir_case.lir_test_op.has_value()) {
        CloneIntoRegion(*lir_case.lir_test_op, hir_case_op.getTest());
      }

      CloneIntoRegion(lir_case.lir_body_op, hir_case_op.getConsequent());
    }
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitSwitchStatementCaseTest(
    JslirSwitchStatementCaseTestOp lir_op) {
  builder_->create<JsirExprRegionEndOp>(
      lir_op.getLoc(), lir_to_hir_mappings_.lookup(lir_op.getTest()));
  return nullptr;
}

mlir::Operation *JslirToJshir::VisitTryStatementStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_body_marker =
      FindMarker(token, JsirControlFlowMarkerKind::TryStatementBody);
  auto lir_handler_marker =
      FindMarker(token, JsirControlFlowMarkerKind::TryStatementHandler);
  auto lir_finalizer_marker =
      FindMarker(token, JsirControlFlowMarkerKind::TryStatementFinalizer);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::TryStatementEnd);

  auto hir_op = builder_->create<JshirTryStatementOp>(lir_op.getLoc());

  if (lir_body_marker != nullptr) {
    CloneIntoRegion(lir_body_marker->getNextNode(), hir_op.getBlock());
  }

  if (lir_handler_marker != nullptr) {
    CloneIntoRegion(lir_handler_marker->getNextNode(), hir_op.getHandler());
  }

  if (lir_finalizer_marker != nullptr) {
    CloneIntoRegion(lir_finalizer_marker->getNextNode(), hir_op.getFinalizer());
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitCatchClauseStart(
    JslirCatchClauseStartOp lir_op) {
  mlir::Value hir_param = nullptr;
  if (auto lir_param = lir_op.getParam(); lir_param != nullptr) {
    hir_param = lir_to_hir_mappings_.lookup(lir_param);
  }

  auto hir_op =
      builder_->create<JshirCatchClauseOp>(lir_op.getLoc(), hir_param);

  CloneIntoRegion(lir_op->getNextNode(), hir_op.getBody());

  return nullptr;
}

mlir::Operation *JslirToJshir::VisitBreakStatement(
    JslirBreakStatementOp lir_op) {
  builder_->create<JshirBreakStatementOp>(lir_op.getLoc(),
                                          lir_op.getLabelAttr());
  return nullptr;
}

mlir::Operation *JslirToJshir::VisitContinueStatement(
    JslirContinueStatementOp lir_op) {
  builder_->create<JshirContinueStatementOp>(lir_op.getLoc(),
                                             lir_op.getLabelAttr());
  return nullptr;
}

mlir::Operation *JslirToJshir::VisitWhileStatementStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_test_marker =
      FindMarker(token, JsirControlFlowMarkerKind::WhileStatementTest);
  auto lir_body_marker =
      FindMarker(token, JsirControlFlowMarkerKind::WhileStatementBody);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::WhileStatementEnd);

  auto hir_op = builder_->create<JshirWhileStatementOp>(lir_op.getLoc());

  if (lir_test_marker != nullptr) {
    CloneIntoRegion(lir_test_marker->getNextNode(), hir_op.getTest());
  }

  if (lir_body_marker != nullptr) {
    CloneIntoRegion(lir_body_marker->getNextNode(), hir_op.getBody());
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitDoWhileStatementStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_body_marker =
      FindMarker(token, JsirControlFlowMarkerKind::DoWhileStatementBody);
  auto lir_test_marker =
      FindMarker(token, JsirControlFlowMarkerKind::DoWhileStatementTest);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::DoWhileStatementEnd);

  auto hir_op = builder_->create<JshirDoWhileStatementOp>(lir_op.getLoc());

  if (lir_body_marker != nullptr) {
    CloneIntoRegion(lir_body_marker->getNextNode(), hir_op.getBody());
  }

  if (lir_test_marker != nullptr) {
    CloneIntoRegion(lir_test_marker->getNextNode(), hir_op.getTest());
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitForStatementStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_init_marker =
      FindMarker(token, JsirControlFlowMarkerKind::ForStatementInit);
  auto lir_test_marker =
      FindMarker(token, JsirControlFlowMarkerKind::ForStatementTest);
  auto lir_body_marker =
      FindMarker(token, JsirControlFlowMarkerKind::ForStatementBody);
  auto lir_update_marker =
      FindMarker(token, JsirControlFlowMarkerKind::ForStatementUpdate);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::ForStatementEnd);

  auto hir_op = builder_->create<JshirForStatementOp>(lir_op.getLoc());

  if (lir_init_marker != nullptr) {
    CloneIntoRegion(lir_init_marker->getNextNode(), hir_op.getInit());
  }

  if (lir_test_marker != nullptr) {
    CloneIntoRegion(lir_test_marker->getNextNode(), hir_op.getTest());
  }

  if (lir_body_marker != nullptr) {
    CloneIntoRegion(lir_body_marker->getNextNode(), hir_op.getBody());
  }

  if (lir_update_marker != nullptr) {
    CloneIntoRegion(lir_update_marker->getNextNode(), hir_op.getUpdate());
  }

  if (lir_end_marker == nullptr) {
    return nullptr;
  }
  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitForInOfStatementStart(
    mlir::Operation *lir_op, mlir::Value lir_iterator, JsirForInOfKind kind,
    JsirForInOfDeclarationAttr left_declaration, mlir::Value lir_left_lval,
    mlir::Value lir_right, std::optional<bool> await) {
  mlir::Value hir_left_lval = lir_to_hir_mappings_.lookup(lir_left_lval);
  mlir::Value hir_right = lir_to_hir_mappings_.lookup(lir_right);
  mlir::Value iterator = lir_iterator;

  JslirForInOfStatementHasNextOp lir_has_next_op = nullptr;
  JslirForInOfStatementGetNextOp lir_get_next_op = nullptr;
  JslirForInOfStatementEndOp lir_end_op = nullptr;
  for (mlir::Operation *user : iterator.getUsers()) {
    if (auto marker = llvm::dyn_cast<JslirForInOfStatementHasNextOp>(user)) {
      if (lir_has_next_op != nullptr) {
        lir_op->emitOpError("duplicate JslirForInOfStatementHasNextOp");
        return nullptr;
      }
      lir_has_next_op = marker;
    }

    if (auto marker = llvm::dyn_cast<JslirForInOfStatementGetNextOp>(user)) {
      if (lir_get_next_op != nullptr) {
        lir_op->emitOpError("duplicate JslirForInOfStatementGetNextOp");
        return nullptr;
      }
      lir_get_next_op = marker;
    }

    if (auto marker = llvm::dyn_cast<JslirForInOfStatementEndOp>(user)) {
      if (lir_end_op != nullptr) {
        lir_op->emitOpError("duplicate JslirForInOfStatementEndOp");
        return nullptr;
      }
      lir_end_op = marker;
    }
  }

  mlir::Region *hir_body_region;
  switch (kind) {
    case JsirForInOfKind::ForIn: {
      auto hir_op = builder_->create<JshirForInStatementOp>(
          lir_op->getLoc(), left_declaration, hir_left_lval, hir_right);
      hir_body_region = &hir_op.getBody();
      break;
    }
    case JsirForInOfKind::ForOf: {
      if (!await.has_value()) {
        lir_op->emitOpError("expect await to be defined, got nullopt");
        return nullptr;
      }

      auto hir_op = builder_->create<JshirForOfStatementOp>(
          lir_op->getLoc(), left_declaration, hir_left_lval, hir_right,
          await.value());
      hir_body_region = &hir_op.getBody();
      break;
    }
  }

  if (lir_get_next_op != nullptr) {
    CloneIntoRegion(lir_get_next_op->getNextNode(), *hir_body_region);
  }

  if (lir_end_op == nullptr) {
    return nullptr;
  }
  return lir_end_op->getNextNode();
}

mlir::Operation *JslirToJshir::VisitForInStatementStart(
    JslirForInStatementStartOp lir_op) {
  return VisitForInOfStatementStart(
      lir_op, lir_op.getIterator(), JsirForInOfKind::ForIn,
      lir_op.getLeftDeclarationAttr(), lir_op.getLeftLval(), lir_op.getRight(),
      /*await=*/std::nullopt);
}

mlir::Operation *JslirToJshir::VisitForOfStatementStart(
    JslirForOfStatementStartOp lir_op) {
  return VisitForInOfStatementStart(
      lir_op, lir_op.getIterator(), JsirForInOfKind::ForOf,
      lir_op.getLeftDeclarationAttr(), lir_op.getLeftLval(), lir_op.getRight(),
      lir_op.getAwait());
}

mlir::Operation *JslirToJshir::VisitLogicalExpressionStart(
    JslirLogicalExpressionStartOp lir_op) {
  mlir::Value token = lir_op.getToken();

  mlir::Value lir_left = lir_op.getLeft();
  mlir::Value hir_left = lir_to_hir_mappings_.lookup(lir_left);

  auto lir_right_marker =
      FindMarker(token, JsirControlFlowMarkerKind::LogicalExpressionRight);
  if (lir_right_marker == nullptr) {
    lir_op->emitError("LogicalExpression must have a right marker.");
    return nullptr;
  }

  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::LogicalExpressionEnd);
  if (lir_end_marker == nullptr) {
    lir_op->emitError("LogicalExpression must have an end marker.");
    return nullptr;
  }

  mlir::Block *lir_end_block = lir_end_marker->getBlock();
  if (lir_end_block->getNumArguments() != 1) {
    lir_op->emitError("LogicalExpression end block should have 1 argument.");
    return nullptr;
  }
  mlir::Value lir_result = lir_end_block->getArgument(0);

  auto hir_op = builder_->create<JshirLogicalExpressionOp>(
      lir_op.getLoc(), lir_op.getOperator_(), hir_left);
  lir_to_hir_mappings_.map(lir_result, hir_op);

  CloneIntoRegion(lir_right_marker->getNextNode(), hir_op.getRight());

  return lir_end_marker->getNextNode();
}

mlir::Operation *JslirToJshir::VisitConditionalExpressionStart(
    JslirControlFlowStarterOp lir_op) {
  mlir::Value token = lir_op.getToken();
  auto lir_alternate_marker = FindMarker(
      token, JsirControlFlowMarkerKind::ConditionalExpressionAlternate);
  auto lir_consequent_marker = FindMarker(
      token, JsirControlFlowMarkerKind::ConditionalExpressionConsequent);
  auto lir_end_marker =
      FindMarker(token, JsirControlFlowMarkerKind::ConditionalExpressionEnd);
  if (lir_end_marker == nullptr) {
    LOG(ERROR) << "ConditionalExpression must have an end marker.";
    return nullptr;
  }

  MALDOCA_ASSIGN_OR_RETURN(mlir::Value lir_condition,
                           GetCondBranchCondition(lir_op->getBlock()), nullptr);

  auto hir_op = builder_->create<JshirConditionalExpressionOp>(
      lir_op.getLoc(), lir_to_hir_mappings_.lookup(lir_condition));

  if (lir_alternate_marker != nullptr) {
    CloneIntoRegion(lir_alternate_marker->getNextNode(), hir_op.getAlternate());
  }

  if (lir_consequent_marker != nullptr) {
    CloneIntoRegion(lir_consequent_marker->getNextNode(),
                    hir_op.getConsequent());
  }

  mlir::Block *lir_end_block = lir_end_marker->getBlock();
  if (lir_end_block->getNumArguments() != 1) {
    LOG(ERROR) << "ConditionalExpression end block should have 1 argument.";
    return nullptr;
  }
  mlir::Value lir_result = lir_end_block->getArgument(0);
  lir_to_hir_mappings_.map(lir_result, hir_op);

  return lir_end_marker->getNextNode();
}

}  // namespace maldoca

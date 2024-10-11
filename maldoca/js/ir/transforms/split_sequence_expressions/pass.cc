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

#include "maldoca/js/ir/transforms/split_sequence_expressions/pass.h"

#include <cassert>
#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// Infers if this block should contain a single statement.
static bool IsStatementBlock(mlir::Block &block) {
  mlir::Region *region = block.getParent();
  if (region == nullptr) {
    return false;
  }

  return llvm::TypeSwitch<mlir::Operation *, bool>(region->getParentOp())
      .Case([&](JshirWithStatementOp parent_op) {
        // interface WithStatement <: Statement {
        //   object: Expression;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirLabeledStatementOp parent_op) {
        // interface LabeledStatement <: Statement {
        //   label: Identifier;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirIfStatementOp parent_op) {
        // interface IfStatement <: Statement {
        //   test: Expression;
        //   consequent: Statement;
        //   alternate: Statement | null;
        // }
        return region == &parent_op.getConsequent() ||
               region == &parent_op.getAlternate();
      })
      .Case([&](JshirWhileStatementOp parent_op) {
        // interface WhileStatement <: Statement {
        //   test: Expression;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirDoWhileStatementOp parent_op) {
        // interface DoWhileStatement <: Statement {
        //   body: Statement;
        //   test: Expression;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirForStatementOp parent_op) {
        // interface ForStatement <: Statement {
        //   init: VariableDeclaration | Expression | null;
        //   test: Expression | null;
        //   update: Expression | null;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirForInStatementOp parent_op) {
        // interface ForInStatement <: Statement {
        //   left: VariableDeclaration | LVal;
        //   right: Expression;
        //   body: Statement;
        // }
        return region == &parent_op.getBody();
      })
      .Case([&](JshirForOfStatementOp parent_op) {
        // interface ForOfStatement <: Statement {
        //   left: VariableDeclaration | LVal;
        //   right: Expression;
        //   body: Statement;
        //   await: boolean;
        // }
        return region == &parent_op.getBody();
      })
      .Default(false);
}

// If the statement is a child of another statement, then we can't split it into
// two statements. For example, consider this `with` statement:
//
// ```
// with (x)
//   a, b;
//   ~~~~~ body
// ```
//
// The `body` is a single statement, and we can't replace it with two
// statements.
//
// More specifically, the `with` statement looks like this in JSHIR:
//
// ```
// %x = jsir.identifier {"x"}
// jshir.with_statement (%x) {
//   %a = jsir.identifier {"a"}
//   %b = jsir.identifier {"b"}
//   %expr = jsir.sequence_expression(%a, %b)
//   jsir.expression_statement(%expr)
// }
// ```
//
// If we simplify split `body` into two statements like this:
//
// ```
// %x = jsir.identifier {"x"}
// jshir.with_statement (%x) {
//   %a = jsir.identifier {"a"}
//   jsir.expression_statement(%a)
//   %b = jsir.identifier {"b"}
//   jsir.expression_statement(%b)
// }
// ```
//
// Then we can't correctly convert `body` into a `JsStatement` node in the AST.
// In the implementation, only one of the two statements gets kept.
//
// Therefore, we need to wrap the two statements in a block:
//
// ```
// with (x) {
//   a;
//   b;
// }
// ```
//
// Or, in JSHIR, like this:
//
// ```
// %x = jsir.identifier {"x"}
// jshir.with_statement (%x) {
//   jshir.block_statement {
//     %a = jsir.identifier {"a"}
//     jsir.expression_statement(%a)
//     %b = jsir.identifier {"b"}
//     jsir.expression_statement(%b)
//   }
// }
// ```
//
// TODO(tzx) Implement a standalone pass to add `jshir.block_statement`.
//
// We shouldn't require each individual pass to maintain the invariant that
// certain `mlir::Block`s should only contain a single statement - a
// `mlir::Block` should always allow multiple statements, and we should
// automatically add `jshir.block_statement`s when lifting JSHIR to AST.

void WrapBlockContentWithBlockStatement(mlir::Block &block) {
  mlir::Region *region = block.getParent();
  mlir::MLIRContext *context = region->getContext();

  // +-------------+-------------+
  // | Before      | After       |
  // +-------------+-------------+
  // | ^block:     | ^block:     |
  // |   op1       |             |
  // |   op2       | ^new_block: |
  // |   ...       |   op1       |
  // |             |   op2       |
  // |             |   ...       |
  // +-------------+-------------+
  mlir::Block *new_block = block.splitBlock(block.begin());
  assert(block.empty());

  // After:
  //
  //  ^block:
  //    JshirBlockStatement(/*directives=*/{}, /*body=*/{})
  //
  //  ^new_block:
  //    op1
  //    op2
  //    ...
  mlir::OpBuilder builder{context};
  builder.setInsertionPointToStart(&block);
  auto block_stmt_op = builder.create<JshirBlockStatementOp>(region->getLoc());

  // `directives` is empty, but we need to keep an empty block in the region.
  block_stmt_op.getDirectives().emplaceBlock();

  // After:
  //
  //  ^block:
  //    JshirBlockStatement(
  //      /*directives=*/{
  //      ^empty_block:
  //      },
  //      /*body=*/{
  //      ^new_block:
  //        op1
  //        op2
  //        ...
  //      })
  new_block->moveBefore(&block_stmt_op.getBody(),
                        block_stmt_op.getBody().end());
}

// From:
// %a = ...
// %b = ...
// %c = ...
// %expr = jsir.sequence_expression(%a, %b, %c)
// jsir.return_statement(%expr)
//
// To:
// %a = ...
// jsir.expression_statement(%a)
// %b = ...
// jsir.expression_statement(%b)
// %c = ...
// jsir.return_statement(%c)

void SplitSequenceExpressions(mlir::Operation *root) {
  mlir::MLIRContext *context = root->getContext();
  mlir::OpBuilder builder{context};

  std::vector<mlir::Block *> modified_blocks;
  root->walk([&](JsirSequenceExpressionOp op) {
    mlir::Block *parent_block = op->getBlock();

    for (mlir::Operation *user : op->getUsers()) {
      if (!(llvm::isa<JsirReturnStatementOp>(user) ||
            llvm::isa<JsirExpressionStatementOp>(user))) {
        return;
      }
    }

    for (mlir::Value expr : op.getExpressions().drop_back(1)) {
      builder.setInsertionPointAfterValue(expr);
      builder.create<JsirExpressionStatementOp>(expr.getLoc(), expr);
    }

    mlir::Value last_expr = op.getExpressions().back();
    op.replaceAllUsesWith(last_expr);
    op.erase();

    if (parent_block != nullptr) {
      modified_blocks.push_back(parent_block);
    }
  });

  for (mlir::Block *block : modified_blocks) {
    if (IsStatementBlock(*block)) {
      WrapBlockContentWithBlockStatement(*block);
    }
  }
}

}  // namespace maldoca

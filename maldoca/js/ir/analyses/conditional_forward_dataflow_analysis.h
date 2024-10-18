// Copyright 2023 Google LLC
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

#ifndef MALDOCA_JS_IR_ANALYSES_COND_FORWARD_DATAFLOW_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_COND_FORWARD_DATAFLOW_ANALYSIS_H_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "maldoca/js/ir/analyses/dataflow_analysis.h"
#include "maldoca/js/ir/analyses/state.h"

namespace maldoca {

// =============================================================================
// JsirConditionalForwardDataFlowAnalysis
// =============================================================================
// A forward dataflow analysis API that attaches lattices to operations.
//
// Added IsExecutable states for each operation and block, so that some blocks
// will not be visited if it is under a false conditional branch.
// This enables built-in support for dead code analysis.
template <typename ValueT, typename StateT>
class JsirConditionalForwardDataFlowAnalysis
    : public JsirForwardDataFlowAnalysis<ValueT, StateT> {
 public:
  using Base = JsirForwardDataFlowAnalysis<ValueT, StateT>;
  using DenseBase = JsirDenseForwardDataFlowAnalysis<StateT>;

  explicit JsirConditionalForwardDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : JsirForwardDataFlowAnalysis<ValueT, StateT>(solver) {}

  // Gets the information about whether a block is executable.
  JsirStateRef<JsirExecutable> GetIsExecutable(mlir::Block *block);

  // Gets the information about whether a CFG edge is executable.
  JsirStateRef<JsirExecutable> GetIsExecutable(mlir::dataflow::CFGEdge *edge);

  // Inherit the same transfer function from base class.
  virtual void VisitOp(mlir::Operation *op,
                       llvm::ArrayRef<const ValueT *> operands,
                       const StateT *before,
                       llvm::MutableArrayRef<JsirStateRef<ValueT>> results,
                       JsirStateRef<StateT> after) = 0;

  // For conditional dataflow analyses, whether the successor of the basic block
  // is executable may be updated after applying transfer functions. This
  // virtual method checks whether the operation is a conditional branch, and
  // returns the possibly executable successors.
  virtual std::optional<std::vector<mlir::Block *>> InferExecutableSuccessors(
      mlir::Operation *op, llvm::ArrayRef<const ValueT *> operands) = 0;

  void PrintAtBlockEntry(mlir::Block &block, size_t num_indents,
                         llvm::raw_ostream &os) override {
    os.indent(num_indents + 2);
    os << "// ";
    auto executable_ref = GetIsExecutable(&block);
    executable_ref.value().print(os);
    os << "\n";

    Base::PrintAtBlockEntry(block, num_indents, os);
  }

 private:
  // We override the three methods from base classes to add IsExecutable info.
  using Base::GetCFGEdge;
  void VisitOp(mlir::Operation *op) override;
  void VisitBlock(mlir::Block *block) override;
  void InitializeBlockDependencies(mlir::Block *block) override;
  using Base::VisitCFGEdge;
};

template <typename ValueT, typename StateT>
JsirStateRef<JsirExecutable>
JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::GetIsExecutable(
    mlir::Block *block) {
  return Base::template GetStateImpl<JsirExecutable>(
      Base::getProgramPointBefore(block));
}

template <typename ValueT, typename StateT>
JsirStateRef<JsirExecutable>
JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::GetIsExecutable(
    mlir::dataflow::CFGEdge *edge) {
  return Base::template GetStateImpl<JsirExecutable>(edge);
}

template <typename ValueT, typename StateT>
void JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::VisitOp(
    mlir::Operation *op) {
  auto *block = op->getBlock();

  JsirStateRef<StateT> before_state_ref = DenseBase::GetStateBefore(op);
  const StateT *before = &before_state_ref.value();

  JsirStateRef after_state_ref = DenseBase::GetStateAfter(op);

  auto [operands, result_state_refs] = Base::GetValueStateRefs(op);

  VisitOp(op, operands, before, result_state_refs, after_state_ref);

  // For terminator operations, we should also update whether the outgoing edges
  // are still executable.
  if (op->getNextNode() == nullptr) {
    std::optional<std::vector<mlir::Block *>> optional_successors;
    // Find the successors that may execute.
    optional_successors = InferExecutableSuccessors(op, operands);

    mlir::BlockRange successors = op->getSuccessors();
    if (optional_successors.has_value()) {
      successors = *optional_successors;
    }

    // Flip CFG edges as live.
    for (mlir::Block *succ : successors) {
      auto *edge = GetCFGEdge(block, succ);
      JsirStateRef<JsirExecutable> edge_executable_ref = GetIsExecutable(edge);
      edge_executable_ref.Write(JsirExecutable{true});
    }
  }
}

template <typename ValueT, typename StateT>
void JsirConditionalForwardDataFlowAnalysis<
    ValueT, StateT>::InitializeBlockDependencies(mlir::Block *block) {
  // The block depends on its incoming CFG edges.
  //
  // In particular, when an incoming CFG is marked as live, the block is
  // visited.
  for (mlir::Block *pred : block->getPredecessors()) {
    auto *edge = Base::GetCFGEdge(pred, block);
    JsirStateRef<JsirExecutable> edge_executable_ref = GetIsExecutable(edge);
    edge_executable_ref.AddDependent(Base::getProgramPointBefore(block));
  }

  // The first time the block is marked as executable, visit all ops.
  //
  // This is because some ops (e.g. constant) do not have other dependencies.
  JsirStateRef<JsirExecutable> block_executable_ref = GetIsExecutable(block);
  for (mlir::Operation &op : *block) {
    block_executable_ref.AddDependent(Base::getProgramPointAfter(&op));
  }

  if (block->isEntryBlock()) {
    // Entry blocks are always executable.
    // This also triggers all ops to be visited.
    block_executable_ref.Write(JsirExecutable{true});
  }
}

template <typename ValueT, typename StateT>
void JsirConditionalForwardDataFlowAnalysis<ValueT, StateT>::VisitBlock(
    mlir::Block *block) {
  // Iterate over the predecessors of the non-entry block.
  for (auto pred_it = block->pred_begin(), pred_end = block->pred_end();
       pred_it != pred_end; ++pred_it) {
    mlir::Block *pred = *pred_it;

    // If the edge from the predecessor block to the current block is not
    // live, bail out.
    auto *edge = GetCFGEdge(pred, block);
    JsirStateRef<JsirExecutable> edge_executable_ref = GetIsExecutable(edge);
    if (!*edge_executable_ref.value()) {
      continue;
    }

    // If this is a flip, it causes all ops in the block to be visited.
    JsirStateRef<JsirExecutable> block_executable_ref = GetIsExecutable(block);
    block_executable_ref.Write(JsirExecutable{true});

    VisitCFGEdge(*pred_it, pred_it.getSuccessorIndex(), block);
  }
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_COND_FORWARD_DATAFLOW_ANALYSIS_H_

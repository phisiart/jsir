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

#ifndef MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_
#define MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "maldoca/js/ir/analyses/state.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// The actual data stored on each mlir::ProgramPoint.
// Users should use JsirStateRef instead of this.
namespace detail {
template <typename T>
class JsirStateElement;
}  // namespace detail

// An accessor to the state stored to each mlir::ProgramPoint.
// Writes cause dependents to be visited.
template <typename T>
class JsirStateRef;

namespace detail {

template <typename T>
class JsirStateElement : public mlir::AnalysisState {
 public:
  static_assert(std::is_base_of_v<JsirState<T>, T>,
                "Must use the CRTP type JsirState. "
                "E.g. class MyState : public JsirState<MyState> {};");

  explicit JsirStateElement(mlir::LatticeAnchor anchor)
      : AnalysisState(anchor) {}

  // Read-only. Please use JsirStateRef to modify the value.
  const T &value() const { return value_; }

  void print(llvm::raw_ostream &os) const override { value_.print(os); }

 private:
  friend class JsirStateRef<T>;
  T value_;
};

}  // namespace detail
}  // namespace maldoca

namespace llvm {

template <typename RangeT> hash_code hash_combine_range(RangeT &&R) {
  return hash_combine_range(adl_begin(R), adl_end(R));
}

// Provide DenseMapInfo for SmallVector of a type which has info.
template <typename T, unsigned N> struct DenseMapInfo<llvm::SmallVector<T, N>> {
  static SmallVector<T, N> getEmptyKey() {
    return {DenseMapInfo<T>::getEmptyKey()};
  }

  static SmallVector<T, N> getTombstoneKey() {
    return {DenseMapInfo<T>::getTombstoneKey()};
  }

  static unsigned getHashValue(const SmallVector<T, N> &V) {
    return static_cast<unsigned>(hash_combine_range(V));
  }

  static bool isEqual(const SmallVector<T, N> &LHS,
                      const SmallVector<T, N> &RHS) {
    return LHS == RHS;
  }
};

}  // namespace llvm

namespace maldoca {

class JsirGeneralCfgEdge
    : public mlir::GenericLatticeAnchorBase<
          JsirGeneralCfgEdge,
          std::tuple<mlir::ProgramPoint *, mlir::ProgramPoint *, unsigned int,
                     mlir::SmallVector<mlir::Value>,
                     mlir::SmallVector<mlir::Value>>> {
 public:
  using Base::Base;

  mlir::ProgramPoint *getPred() const { return std::get<0>(getValue()); }

  mlir::ProgramPoint *getSucc() const { return std::get<1>(getValue()); }

  // TODO Storing the value vectors in the CFG edge might make this unneeded?
  unsigned int getBranchSuccIndex() const { return std::get<2>(getValue()); }

  const mlir::SmallVector<mlir::Value> &getPredValues() const {
    return std::get<3>(getValue());
  }

  const mlir::SmallVector<mlir::Value> &getSuccValues() const {
    return std::get<4>(getValue());
  }

  void print(llvm::raw_ostream &os) const override {
    getPred()->print(os);
    os << "\n -> \n";
    getSucc()->print(os);
    os << " with index " << getBranchSuccIndex() << "\n";
  }

  mlir::Location getLoc() const override {
    return mlir::FusedLoc::get(getPred()->getBlock()->getParent()->getContext(),
                               {getPred()->getBlock()->getParent()->getLoc(),
                                getSucc()->getBlock()->getParent()->getLoc()});
  }
};

template <typename T>
class JsirStateRef {
 public:
  static_assert(std::is_base_of_v<JsirState<T>, T>,
                "Must use the CRTP type JsirState. "
                "E.g. class MyState : public JsirState<MyState> {};");

  explicit JsirStateRef()
      : element_(nullptr), solver_(nullptr), analysis_(nullptr) {}

  explicit JsirStateRef(detail::JsirStateElement<T> *element,
                        mlir::DataFlowSolver *solver,
                        mlir::DataFlowAnalysis *analysis)
      : element_(element), solver_(solver), analysis_(analysis) {}

  bool operator==(std::nullptr_t) const { return element_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return element_ != nullptr; }

  detail::JsirStateElement<T> *element() { return element_; }

  const T &value() const { return element_->value(); }

  // Marks a program point as depending on this state.
  // This means that whenever this state is updated, we trigger a visit() of
  // that program point.
  void AddDependent(mlir::ProgramPoint *point);

  // Writes the state and triggers visit()s of its dependents.
  void Write(absl::FunctionRef<mlir::ChangeResult(T *)> write_fn);

  // Writes the state and triggers visit()s of its dependents.
  void Write(T &&lattice);
  void Write(const T &lattice);

  // Joins the state and triggers visit()s of its dependents.
  void Join(const T &lattice);

 private:
  // Points to the actual data attached to the program point.
  detail::JsirStateElement<T> *element_;

  // The solver that drives the worklist algorithm.
  // We need this to access the solver APIs to propagate changes.
  mlir::DataFlowSolver *solver_;

  // The analysis that this state belongs to.
  // When we schedule a new program point to be visited, we need to specify the
  // analysis, hence the need of this field.
  mlir::DataFlowAnalysis *analysis_;
};

// A lattice that represents if a piece of code is executable.
// Join(executable, non-executable) = executable
class JsirExecutable : public JsirState<JsirExecutable> {
 public:
  explicit JsirExecutable(bool executable = false) : executable_(executable) {}

  mlir::ChangeResult Join(const JsirExecutable &other) override;

  const bool &operator*() const { return executable_; }

  bool operator==(const JsirExecutable &rhs) const override {
    return executable_ == rhs.executable_;
  }

  bool operator!=(const JsirExecutable &rhs) const override {
    return !(operator==(rhs));
  }

  void print(llvm::raw_ostream &os) const override;

 private:
  bool executable_ = false;
};

template <typename T>
class JsirDenseStates {
 public:
  virtual ~JsirDenseStates() = default;

  // Gets the state attached before an op.
  virtual T GetStateBefore(mlir::Operation *op) = 0;

  // Gets the state attached after an op.
  virtual T GetStateAfter(mlir::Operation *op) = 0;

  // Gets the state attached at the entry of a block.
  virtual T GetStateAtEntryOf(mlir::Block *block) = 0;

  // Gets the state attached at the end of a block.
  virtual T GetStateAtEndOf(mlir::Block *block) = 0;
};

template <typename T>
class JsirSparseStates {
 public:
  virtual ~JsirSparseStates() = default;

  // Gets the state at an SSA value.
  virtual T GetStateAt(mlir::Value value) = 0;
};

class JsirDataFlowAnalysisPrinter {
 public:
  virtual ~JsirDataFlowAnalysisPrinter() = default;

  // Format:
  //
  // ^block_name:
  //   <AtBlockEntry>%result0 = an_op (%arg0, %arg1, ...)<AfterOp>
  //   %result1 = another_op (%arg0, %arg1, ...)<AfterOp>
  //   ...
  virtual void PrintOp(mlir::Operation *op, size_t num_indents,
                       mlir::AsmState &asm_state, llvm::raw_ostream &os) = 0;

  std::string PrintOp(mlir::Operation *op) {
    std::string output;
    llvm::raw_string_ostream os(output);
    mlir::AsmState asm_state(op);
    PrintOp(op, /*num_indents=*/0, asm_state, os);
    os.flush();
    return output;
  }
};

enum class DataflowDirection { kForward, kBackward };

// =============================================================================
// JsirDenseDataFlowAnalysis
// =============================================================================
// A dataflow analysis API that attaches lattices to operations. This analysis
// supports both forward and backward analysis.
template <typename StateT, DataflowDirection direction>
class JsirDenseDataFlowAnalysis : public mlir::DataFlowAnalysis,
                                  public JsirDataFlowAnalysisPrinter,
                                  public JsirDenseStates<JsirStateRef<StateT>> {
 public:
  explicit JsirDenseDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : mlir::DataFlowAnalysis(solver), solver_(solver) {
    registerAnchorKind<JsirGeneralCfgEdge>();
  }

  // Set the initial state of an entry block for forward analysis or exit block
  // for backward analysis.
  virtual void InitializeBoundaryBlock(mlir::Block *block,
                                       JsirStateRef<StateT> boundary_state) = 0;

  // This virtual method is the transfer function for an operation. It is called
  // by its overloaded protected method. Remember, what the input and output
  // should come from is different in forward and backward analysis. In forward
  // analysis, the input should be the state before the op. In backward
  // analysis, it should be the state after the op.
  // +--------+-------------------+-------------------+
  // |        | Forward Analysis  | Backward Analysis |
  // +--------+-------------------+-------------------+
  // | Input  |       Before      |       After       |
  // +--------+-------------------+-------------------+
  // | Output |       After       |       Before      |
  // +--------+-------------------+-------------------+
  //
  // Input:
  // - The analysis state (lattice value) input for the op
  //
  // Output:
  // - The analysis state (lattice value) output for the op
  virtual void VisitOp(mlir::Operation *op, const StateT *input,
                       JsirStateRef<StateT> output) = 0;

  // Gets the state attached before an op.
  JsirStateRef<StateT> GetStateBefore(mlir::Operation *op) final;

  // Gets the state attached after an op.
  JsirStateRef<StateT> GetStateAfter(mlir::Operation *op) final;

  // Gets the state attached at the entry of a block.
  JsirStateRef<StateT> GetStateAtEntryOf(mlir::Block *block) final;

  // Gets the state attached at the end of a block.
  JsirStateRef<StateT> GetStateAtEndOf(mlir::Block *block) final;

  // Format:
  //
  // ^block_name:
  //   <AtBlockEntry>%result0 = an_op (%arg0, %arg1, ...)<AfterOp>
  //   %result1 = another_op (%arg0, %arg1, ...)<AfterOp>
  //   ...
  void PrintOp(mlir::Operation *op, size_t num_indents,
               mlir::AsmState &asm_state, llvm::raw_ostream &os) override;

  using JsirDataFlowAnalysisPrinter::PrintOp;

  void PrintRegion(mlir::Region &region, size_t num_indents,
                   mlir::AsmState &asm_state, llvm::raw_ostream &os);

  bool IsEntryBlock(mlir::Block *block);

  // When we visit the op, visit all the CFG edges associated with that op.
  absl::flat_hash_map<mlir::Operation *, std::vector<JsirGeneralCfgEdge *>>
      op_to_cfg_edges_;

  absl::flat_hash_map<mlir::Block *, std::vector<JsirGeneralCfgEdge *>>
      block_to_cfg_edges_;

 protected:
  JsirGeneralCfgEdge *GetCFGEdge(mlir::ProgramPoint *pred,
                                 mlir::ProgramPoint *succ,
                                 unsigned int branch_succ_index);

  void EmplaceCFGEdge(mlir::ProgramPoint *from, mlir::ProgramPoint *to,
                      unsigned int branch_succ_index, mlir::Operation *op);

  void EmplaceCFGEdgesFromRegion(mlir::Region &from_exits_of,
                                 mlir::ProgramPoint *to,
                                 unsigned int branch_succ_index,
                                 mlir::Operation *op);

  void EmplaceCFGEdgesBetweenRegions(mlir::Region &from_exits_of,
                                     mlir::Region &to_entry_of,
                                     unsigned int branch_succ_index,
                                     mlir::Operation *op);

  // TODO(b/425421947) Eventually we won't need this function.
  bool IsBlockCfg(JsirGeneralCfgEdge *edge) {
    return edge->getPred()->isBlockEnd() && edge->getSucc()->isBlockStart() &&
           edge->getPred()->getBlock()->mightHaveTerminator() &&
           (edge->getPred()->getBlock()->getParent() ==
            edge->getSucc()->getBlock()->getParent());
  }

  // Gets the state at the program point.
  template <typename T>
  JsirStateRef<T> GetStateImpl(mlir::LatticeAnchor anchor);

  mlir::LogicalResult initialize(mlir::Operation *op) override;
  void InitializeBlock(mlir::Block *block);

  // Since our analysis algorithm is based on MLIR's dataflow analysis, we need
  // to set up the dependency information between basic blocks so that the
  // fixpoint algorithm works.
  // Different analyses may have different strategies. For instance, conditional
  // forward analysis requires to mark whether each successor basic block is
  // executable and selectively add executable basic blocks as successors. Here,
  // we provide a vanilla (unconditional) dependency initialization that
  // provides all successors as dependencies.
  // This method is called inside `InitializeBlock`.
  virtual void InitializeBlockDependencies(mlir::Block *block);
  virtual void VisitBlock(mlir::Block *block);
  virtual void VisitOp(mlir::Operation *op);

  // This method mainly serves to "join" states from blocks. i.e., this method
  // should implement the "join" operation in a dataflow analysis. It should
  // join the states from the end of the predecessor into the entry of the
  // successor for a forward analysis, or join the states from the entry of a
  // block to the end of the predecessor for a backward analysis.
  // `branch_succ_index` is the index of the CFG edge among all edges from the
  // predecessor. This is necessary for sparse values.
  virtual void VisitCFGEdge(JsirGeneralCfgEdge *edge);

  // Callbacks for `PrintOp`. See comments of `PrintOp` for the format.
  virtual void PrintAtBlockEntry(mlir::Block &block, size_t num_indents,
                                 llvm::raw_ostream &os);
  virtual void PrintAfterOp(mlir::Operation *op, size_t num_indents,
                            mlir::AsmState &asm_state, llvm::raw_ostream &os);

  mlir::DataFlowSolver &solver_;

 private:
  // Override `mlir::DataFlowAnalysis::visit` and redirect to `Visit{Op,Block}`.
  mlir::LogicalResult visit(mlir::ProgramPoint *point) override;
};

template <typename StateT>
using JsirDenseForwardDataFlowAnalysis =
    JsirDenseDataFlowAnalysis<StateT, DataflowDirection::kForward>;

template <typename StateT>
using JsirDenseBackwardDataFlowAnalysis =
    JsirDenseDataFlowAnalysis<StateT, DataflowDirection::kBackward>;

// =============================================================================
// JsirDataFlowAnalysis
// =============================================================================
// A dataflow analysis API that attaches lattices to both values and operations.
// This analysis supports both forward and backward analysis.
template <typename ValueT, typename StateT, DataflowDirection direction>
class JsirDataFlowAnalysis
    : public JsirDenseDataFlowAnalysis<StateT, direction>,
      public JsirSparseStates<JsirStateRef<ValueT>> {
 public:
  using Base = JsirDenseDataFlowAnalysis<StateT, direction>;

  explicit JsirDataFlowAnalysis(mlir::DataFlowSolver &solver)
      : JsirDenseDataFlowAnalysis<StateT, direction>(solver) {}

  // The initial state on a boundary `mlir::Value`, e.g. a parameter of an entry
  // block. This is used in both backward and forward analysis, when visiting
  // the CFG edges.
  virtual ValueT BoundaryInitialValue() const = 0;

  // Sets the initial state on a boundary `mlir::Block`, i.e. the entry state of
  // an entry block for a forward analysis, or the exit state of an exit block
  // for a backward analysis.
  virtual void InitializeBoundaryBlock(
      mlir::Block *block, JsirStateRef<StateT> boundary_state,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> arg_states) = 0;

  using Base::InitializeBoundaryBlock;
  void InitializeBoundaryBlock(mlir::Block *block,
                               JsirStateRef<StateT> boundary_state) override {
    std::vector<JsirStateRef<ValueT>> arg_states;
    for (mlir::Value arg : block->getArguments()) {
      arg_states.push_back(GetStateAt(arg));
    }
    return InitializeBoundaryBlock(block, boundary_state, arg_states);
  }

  // This virtual method is the transfer function for an operation. It is called
  // by its overloaded protected method. Same as its version in dense analysis,
  // what the input and output of sparse states (`ValueT`) should come from
  // is different in forward and backward analysis.
  //
  // Generally, a transfer function in a dataflow analysis can be represented in
  // a form of
  //
  //   output = gen ∪ (input - kill)
  //
  // where ∪ is the lattice join operation. Usually, for forward analysis, the
  // `gen` set comes from the `results` in a JSIR `Operation`, and `kill` set
  // comes from the `operands`. For backward analysis, it is the opposite case.
  //
  // For sparse values, we would update the values in `gen` set, and read values
  // from `kill` set. Thus, we have the following table for sparse values:
  // +--------+-------------------+-------------------+
  // |        | Forward Analysis  | Backward Analysis |
  // +--------+-------------------+-------------------+
  // | Input  |      Operands     |     Results       |
  // +--------+-------------------+-------------------+
  // | Output |      Results      |     Operands      |
  // +--------+-------------------+-------------------+
  virtual void VisitOp(
      mlir::Operation *op, llvm::ArrayRef<const ValueT *> sparse_input,
      const StateT *dense_input,
      llvm::MutableArrayRef<JsirStateRef<ValueT>> sparse_output,
      JsirStateRef<StateT> dense_output) = 0;

  // Gets the state at an SSA value.
  JsirStateRef<ValueT> GetStateAt(mlir::Value value) final;

 protected:
  using Base::GetCFGEdge;
  using Base::InitializeBlockDependencies;
  using Base::VisitBlock;
  using Base::VisitOp;
  void VisitCFGEdge(JsirGeneralCfgEdge *edge) override;

  // Helper function to get the `StateRef`s for the operands and results of an
  // op. For forward analysis, the input should be the operands and the output
  // should be the results. For backward analysis, the input should be the
  // results and the output should be the operands.
  struct ValueStateRefs {
    std::vector<const ValueT *> inputs;
    std::vector<JsirStateRef<ValueT>> outputs;
  };
  ValueStateRefs GetValueStateRefs(mlir::Operation *op);

  void PrintAfterOp(mlir::Operation *op, size_t num_indents,
                    mlir::AsmState &asm_state, llvm::raw_ostream &os) override;

 private:
  mlir::LogicalResult initialize(mlir::Operation *op) override;

  // Override the transfer function in `JsirDenseDataFlowAnalysis` and
  // redirect to the transfer function supporting sparse values in
  // `JsirDataFlowAnalysis`.
  void VisitOp(mlir::Operation *op, const StateT *input,
               JsirStateRef<StateT> output) override;

  using Base::solver_;
};

template <typename StateT, typename ValueT>
using JsirForwardDataFlowAnalysis =
    JsirDataFlowAnalysis<StateT, ValueT, DataflowDirection::kForward>;

template <typename StateT, typename ValueT>
using JsirBackwardDataFlowAnalysis =
    JsirDataFlowAnalysis<StateT, ValueT, DataflowDirection::kBackward>;

// =============================================================================
// JsirStateRef
// =============================================================================

template <typename T>
void JsirStateRef<T>::AddDependent(mlir::ProgramPoint *point) {
  element_->addDependency(point, analysis_);
}

template <typename T>
void JsirStateRef<T>::Write(
    absl::FunctionRef<mlir::ChangeResult(T *)> write_fn) {
  mlir::ChangeResult changed = write_fn(&element_->value_);
  solver_->propagateIfChanged(element_, changed);
}

template <typename T>
void JsirStateRef<T>::Write(T &&lattice) {
  if (element_->value_ == lattice) {
    return;
  }

  element_->value_ = std::move(lattice);
  solver_->propagateIfChanged(element_, mlir::ChangeResult::Change);
}

template <typename T>
void JsirStateRef<T>::Write(const T &lattice) {
  T lattice_copy = lattice;
  Write(std::move(lattice_copy));
}

template <typename T>
void JsirStateRef<T>::Join(const T &lattice) {
  mlir::ChangeResult changed = element_->value_.Join(lattice);
  solver_->propagateIfChanged(element_, changed);
}

// =============================================================================
// JsirDenseDataFlowAnalysis
// =============================================================================

template <typename StateT, DataflowDirection direction>
JsirGeneralCfgEdge *JsirDenseDataFlowAnalysis<StateT, direction>::GetCFGEdge(
    mlir::ProgramPoint *pred, mlir::ProgramPoint *succ,
    unsigned int branch_succ_index) {
  llvm::SmallVector<mlir::Value> pred_values;
  llvm::SmallVector<mlir::Value> succ_values;

  // Is this a traditional CFG edge (as opposed to a general one)?
  if (pred->isBlockEnd() && succ->isBlockStart() &&
      pred->getBlock()->mightHaveTerminator() &&
      (pred->getBlock()->getParent() == succ->getBlock()->getParent())) {
    auto term_op = pred->getBlock()->getTerminator();
    auto pred_branch = llvm::dyn_cast<mlir::BranchOpInterface>(term_op);
    pred_values = pred_branch.getSuccessorOperands(branch_succ_index)
                      .getForwardedOperands();
    succ_values = {succ->getBlock()->getArguments().begin(),
                   succ->getBlock()->getArguments().end()};
  }

  return getLatticeAnchor<JsirGeneralCfgEdge>(pred, succ, branch_succ_index,
                                              pred_values, succ_values);
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::EmplaceCFGEdge(
    mlir::ProgramPoint *from, mlir::ProgramPoint *to,
    unsigned int branch_succ_index, mlir::Operation *op) {
  op_to_cfg_edges_[op].push_back(GetCFGEdge(from, to, branch_succ_index));
  auto from_state = GetStateImpl<StateT>(from);
  from_state.AddDependent(getProgramPointAfter(op));
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::EmplaceCFGEdgesFromRegion(
    mlir::Region &from_exits_of, mlir::ProgramPoint *to,
    unsigned int branch_succ_index, mlir::Operation *op) {
  for (mlir::Block &block : from_exits_of) {
    if (block.getSuccessors().empty()) {
      mlir::ProgramPoint *after_block = getProgramPointAfter(&block);
      EmplaceCFGEdge(after_block, to, branch_succ_index, op);
    }
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::
    EmplaceCFGEdgesBetweenRegions(mlir::Region &from_exits_of,
                                  mlir::Region &to_entry_of,
                                  unsigned int branch_succ_index,
                                  mlir::Operation *op) {
  mlir::ProgramPoint *entry = getProgramPointBefore(&to_entry_of.front());
  EmplaceCFGEdgesFromRegion(from_exits_of, entry, branch_succ_index, op);
}

template <typename StateT, DataflowDirection direction>
template <typename T>
JsirStateRef<T> JsirDenseDataFlowAnalysis<StateT, direction>::GetStateImpl(
    mlir::LatticeAnchor anchor) {
  auto *element =
      mlir::DataFlowAnalysis::getOrCreate<detail::JsirStateElement<T>>(anchor);
  return JsirStateRef<T>{element, &solver_, this};
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateBefore(
    mlir::Operation *op) {
  if (auto *prev_op = op->getPrevNode()) {
    return GetStateAfter(prev_op);
  } else {
    return GetStateImpl<StateT>(getProgramPointBefore(op->getBlock()));
  }
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateAfter(
    mlir::Operation *op) {
  return GetStateImpl<StateT>(getProgramPointAfter(op));
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateAtEntryOf(
    mlir::Block *block) {
  return GetStateImpl<StateT>(getProgramPointBefore(block));
}

template <typename StateT, DataflowDirection direction>
JsirStateRef<StateT>
JsirDenseDataFlowAnalysis<StateT, direction>::GetStateAtEndOf(
    mlir::Block *block) {
  if (block->empty()) {
    return GetStateAtEntryOf(block);
  } else {
    return GetStateAfter(&block->back());
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintOp(
    mlir::Operation *op, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  size_t num_results = op->getNumResults();
  size_t num_operands = op->getNumOperands();
  size_t num_attributes = op->getAttrs().size();
  size_t num_regions = op->getNumRegions();

  for (size_t i = 0; i != num_results; ++i) {
    if (i != 0) {
      os << ", ";
    }
    op->getResult(i).printAsOperand(os, asm_state);
  }

  if (num_results != 0) {
    os << " = ";
  }

  os << op->getName();

  if (num_operands != 0) {
    os << " (";
    for (size_t i = 0; i != num_operands; ++i) {
      if (i != 0) {
        os << ", ";
      }
      op->getOperand(i).printAsOperand(os, asm_state);
    }
    os << ")";
  }

  if (num_attributes != 0) {
    os << " {";
    for (size_t i = 0; i != num_attributes; ++i) {
      if (i != 0) {
        os << ", ";
      }
      op->getAttrs()[i].getValue().print(os);
    }
    os << "}";
  }

  if (num_regions != 0) {
    os << " (";
    for (size_t i = 0; i != num_regions; ++i) {
      if (i != 0) {
        os << ", ";
      }
      PrintRegion(op->getRegion(i), num_indents, asm_state, os);
    }
    os << ")";
  }

  PrintAfterOp(op, num_indents, asm_state, os);
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintRegion(
    mlir::Region &region, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  os << "{\n";
  {
    llvm::SaveAndRestore<size_t> num_indents_in_region{num_indents,
                                                       num_indents + 2};

    for (mlir::Block &block : region.getBlocks()) {
      os.indent(num_indents);
      block.printAsOperand(os, asm_state);
      os << ":\n";

      llvm::SaveAndRestore<size_t> num_indents_in_block{num_indents,
                                                        num_indents + 2};

      PrintAtBlockEntry(block, num_indents, os);

      for (mlir::Operation &op : block) {
        os.indent(num_indents);
        PrintOp(&op, num_indents, asm_state, os);
        os << "\n";
      }
    }
  }
  os.indent(num_indents);
  os << "}";
}

template <typename StateT, DataflowDirection direction>
bool JsirDenseDataFlowAnalysis<StateT, direction>::IsEntryBlock(
    mlir::Block *block) {
  mlir::Operation *parent_op = block->getParentOp();
  if (llvm::isa<JsirProgramOp>(parent_op)) {
    return true;
  }
  if (llvm::isa<JsirFileOp>(parent_op)) {
    return true;
  }

  // TODO(tzx): Rethink this. Function parameters should be UNKNOWN but
  // other variables should be UNINIT.
  if (llvm::isa<JsirFunctionDeclarationOp>(parent_op)) {
    return true;
  }
  if (llvm::isa<JsirFunctionExpressionOp>(parent_op)) {
    return true;
  }
  if (llvm::isa<JsirObjectMethodOp>(parent_op)) {
    return true;
  }
  if (llvm::isa<JsirClassMethodOp>(parent_op)) {
    return true;
  }
  if (parent_op->getParentOp() == nullptr) {
    return true;
  }
  return false;
}

// Initializes states on all program points:
// - On every `mlir::Value`:
//   ValueT.
// - After every `mlir::Operation`:
//   StateT.
// - At the entry of every `mlir::Block`:
//   StateT.
// - On every `mlir::Block`:
//   JsirExecutable.
// - On every CFG edge (Block -> Block):
//   JsirExecutable.
template <typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDenseDataFlowAnalysis<StateT, direction>::initialize(
    mlir::Operation *op) {
  // Register `op`'s dependent state.
  if (op->getParentOp() != nullptr) {
    if constexpr (direction == DataflowDirection::kForward) {
      JsirStateRef<StateT> before_state_ref = GetStateBefore(op);
      before_state_ref.AddDependent(getProgramPointAfter(op));
    } else if constexpr (direction == DataflowDirection::kBackward) {
      JsirStateRef<StateT> after_state_ref = GetStateAfter(op);
      after_state_ref.AddDependent(getProgramPointAfter(op));
    }
  }

  // ┌─────◄
  // │     jshir.if_statement (
  // ├─────► ┌───────────────┐
  // │       │ true region   │
  // │  ┌──◄ └───────────────┘
  // └──│──► ┌───────────────┐
  //    │    │ false region  │
  //    ├──◄ └───────────────┘
  //    │  );
  //    └──►
  if (auto if_op = mlir::dyn_cast<JshirIfStatementOp>(op); if_op != nullptr) {
    mlir::ProgramPoint *before_if = getProgramPointBefore(if_op);
    mlir::ProgramPoint *after_if = getProgramPointAfter(if_op);

    const auto regions = {&if_op.getConsequent(), &if_op.getAlternate()};
    for (auto [i, region] : llvm::enumerate(regions)) {
      if (!region->empty()) {
        mlir::ProgramPoint *to = getProgramPointBefore(&region->front());

        EmplaceCFGEdge(before_if, to, i, if_op);
        EmplaceCFGEdgesFromRegion(*region, after_if, 0, if_op);
      }
    }
  }

  // ┌─────◄
  // │     jshir.block_statement (
  // └─────► ┌───────────────┐
  //         │ directives    │
  //    ┌──◄ └───────────────┘
  //    └──► ┌───────────────┐
  //         │ body region   │
  // ┌─────◄ └───────────────┘
  // │     );
  // └─────►
  if (auto block_op = mlir::dyn_cast<JshirBlockStatementOp>(op);
      block_op != nullptr) {
    mlir::ProgramPoint *before_block = getProgramPointBefore(block_op);
    mlir::ProgramPoint *after_block = getProgramPointAfter(block_op);
    mlir::ProgramPoint *before_directives =
        getProgramPointBefore(&block_op.getDirectives().front());

    EmplaceCFGEdge(before_block, before_directives, 0, block_op);
    EmplaceCFGEdgesBetweenRegions(block_op.getDirectives(), block_op.getBody(),
                                  0, block_op);
    EmplaceCFGEdgesFromRegion(block_op.getBody(), after_block, 0, block_op);
  }

  // ┌─────◄
  // │     jshir.while_statement (
  // ├─────► ┌───────────────┐
  // │       │ test region   │
  // │  ┌──◄ └───────────────┘
  // │  ├──► ┌───────────────┐
  // │  │    │ body region   │
  // └──│──◄ └───────────────┘
  //    │  );
  //    └──►
  if (auto while_op = mlir::dyn_cast<JshirWhileStatementOp>(op);
      while_op != nullptr) {
    mlir::ProgramPoint *before_while = getProgramPointBefore(while_op);
    mlir::ProgramPoint *after_while = getProgramPointAfter(while_op);
    mlir::ProgramPoint *before_test =
        getProgramPointBefore(&while_op.getTest().front());

    EmplaceCFGEdge(before_while, before_test, 0, while_op);
    // TODO(b/425421947): The edges {test -> body} and {test -> after_while} are
    // conditional.
    EmplaceCFGEdgesBetweenRegions(while_op.getTest(), while_op.getBody(), 0,
                                  while_op);
    EmplaceCFGEdgesBetweenRegions(while_op.getBody(), while_op.getTest(), 0,
                                  while_op);
    EmplaceCFGEdgesFromRegion(while_op.getTest(), after_while, 1, while_op);
  }

  // ┌─────◄
  // │     jshir.do_while_statement (
  // ├─────► ┌───────────────┐
  // │       │ body region   │
  // │  ┌──◄ └───────────────┘
  // │  └──► ┌───────────────┐
  // │       │ test region   │
  // ├─────◄ └───────────────┘
  // │     );
  // └─────►
  if (auto do_while_op = mlir::dyn_cast<JshirDoWhileStatementOp>(op);
      do_while_op != nullptr) {
    mlir::ProgramPoint *before_do_while = getProgramPointBefore(do_while_op);
    mlir::ProgramPoint *after_do_while = getProgramPointAfter(do_while_op);
    mlir::ProgramPoint *before_body =
        getProgramPointBefore(&do_while_op.getBody().front());

    EmplaceCFGEdge(before_do_while, before_body, 0, do_while_op);
    EmplaceCFGEdgesBetweenRegions(do_while_op.getBody(), do_while_op.getTest(),
                                  0, do_while_op);
    // TODO(b/425421947): The edges {test -> body} and {test -> after_do_while}
    // are conditional.
    EmplaceCFGEdgesBetweenRegions(do_while_op.getTest(), do_while_op.getBody(),
                                  1, do_while_op);
    EmplaceCFGEdgesFromRegion(do_while_op.getTest(), after_do_while, 0,
                              do_while_op);
  }

  //    ┌─────◄
  //    │     jshir.for_statement (
  //    └─────► ┌───────────────┐
  //            │ init region   │
  // ┌────────◄ └───────────────┘
  // ├────────► ┌───────────────┐
  // │          │ test region   │
  // │  ┌─────◄ └───────────────┘
  // │  ├─────► ┌───────────────┐
  // │  │       │ body region   │
  // │  │  ┌──◄ └───────────────┘
  // │  │  └──► ┌───────────────┐
  // │  │       │ update region │
  // └──│─────◄ └───────────────┘
  //    │     );
  //    └─────►
  if (auto for_op = mlir::dyn_cast<JshirForStatementOp>(op);
      for_op != nullptr) {
    mlir::ProgramPoint *before_for = getProgramPointBefore(for_op);
    mlir::ProgramPoint *after_for = getProgramPointAfter(for_op);
    mlir::ProgramPoint *before_init =
        getProgramPointBefore(&for_op.getInit().front());

    EmplaceCFGEdge(before_for, before_init, 0, for_op);
    EmplaceCFGEdgesBetweenRegions(for_op.getInit(), for_op.getTest(), 0,
                                  for_op);
    // TODO(b/425421947): The edges {test -> body} and {test -> after_for} are
    // conditional.
    EmplaceCFGEdgesBetweenRegions(for_op.getTest(), for_op.getBody(), 0,
                                  for_op);
    EmplaceCFGEdgesBetweenRegions(for_op.getBody(), for_op.getUpdate(), 0,
                                  for_op);
    EmplaceCFGEdgesBetweenRegions(for_op.getUpdate(), for_op.getTest(), 0,
                                  for_op);
    EmplaceCFGEdgesFromRegion(for_op.getTest(), after_for, 1, for_op);
  }

  // ┌─────◄
  // │     jshir.for_in_statement (
  // └──┬──► ┌───────────────┐
  //    │    │ body region   │
  // ┌──┴──◄ └───────────────┘
  // │     );
  // └─────►
  if (auto for_in_op = mlir::dyn_cast<JshirForInStatementOp>(op);
      for_in_op != nullptr) {
    mlir::ProgramPoint *before_for_in = getProgramPointBefore(for_in_op);
    mlir::ProgramPoint *after_for_in = getProgramPointAfter(for_in_op);

    mlir::ProgramPoint *before_body =
        getProgramPointBefore(&for_in_op.getBody().front());

    EmplaceCFGEdge(before_for_in, before_body, 0, for_in_op);
    EmplaceCFGEdgesBetweenRegions(for_in_op.getBody(), for_in_op.getBody(), 0,
                                  for_in_op);
    EmplaceCFGEdgesFromRegion(for_in_op.getBody(), after_for_in, 1, for_in_op);
  }

  // ┌─────◄
  // │     jshir.for_of_statement (
  // └──┬──► ┌───────────────┐
  //    │    │ body region   │
  // ┌──┴──◄ └───────────────┘
  // │     );
  // └─────►
  if (auto for_of_op = mlir::dyn_cast<JshirForOfStatementOp>(op);
      for_of_op != nullptr) {
    mlir::ProgramPoint *before_for_of = getProgramPointBefore(for_of_op);
    mlir::ProgramPoint *after_for_of = getProgramPointAfter(for_of_op);

    mlir::ProgramPoint *before_body =
        getProgramPointBefore(&for_of_op.getBody().front());

    EmplaceCFGEdge(before_for_of, before_body, 0, for_of_op);
    EmplaceCFGEdgesBetweenRegions(for_of_op.getBody(), for_of_op.getBody(), 0,
                                  for_of_op);
    EmplaceCFGEdgesFromRegion(for_of_op.getBody(), after_for_of, 1, for_of_op);
  }

  // Recursively initialize.
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      InitializeBlock(&block);
    }
  }

  return mlir::success();
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::InitializeBlock(
    mlir::Block *block) {
  // Initialize all inner ops.
  for (mlir::Operation &op : *block) {
    initialize(&op);
  }
  InitializeBlockDependencies(block);
  if constexpr (direction == DataflowDirection::kForward) {
    // TODO(b/425421947): Switch this to IsEntryBlock()? Currently these
    // functions do not always agree with each other.
    if (block->isEntryBlock()) {
      JsirStateRef<StateT> block_state_ref = GetStateAtEntryOf(block);
      InitializeBoundaryBlock(block, block_state_ref);
    }

    // Iterate over the predecessors of the non-entry block.
    for (auto pred_it = block->pred_begin(), pred_end = block->pred_end();
         pred_it != pred_end; ++pred_it) {
      auto *edge =
          GetCFGEdge(getProgramPointAfter(*pred_it),
                     getProgramPointBefore(block), pred_it.getSuccessorIndex());
      block_to_cfg_edges_[block].push_back(edge);
    }

    solver_.enqueue(
        mlir::DataFlowSolver::WorkItem{getProgramPointBefore(block), this});
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // The definition below is copied from https://reviews.llvm.org/D154713.
    auto is_exit_block = [](mlir::Block *b) {
      // Treat empty and terminator-less blocks as exit blocks.
      if (b->empty() ||
          !b->back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
        return true;

      // There may be a weird case where a terminator may be transferring
      // control either to the parent or to another block, so exit blocks and
      // successors are not mutually exclusive.
      mlir::Operation *terminator = b->getTerminator();
      return terminator && terminator->hasTrait<mlir::OpTrait::ReturnLike>();
    };

    if (is_exit_block(block)) {
      JsirStateRef<StateT> block_state_ref = GetStateAtEndOf(block);
      InitializeBoundaryBlock(block, block_state_ref);
    }

    // Iterate over the successors of the non-exit block.
    for (auto succ_it = block->succ_begin(), succ_end = block->succ_end();
         succ_it != succ_end; ++succ_it) {
      // TODO: Design a unit test to cover multiple branches.
      auto *edge =
          GetCFGEdge(getProgramPointAfter(block),
                     getProgramPointBefore(*succ_it), succ_it.getIndex());
      block_to_cfg_edges_[block].push_back(edge);
    }

    solver_.enqueue(
        mlir::DataFlowSolver::WorkItem{getProgramPointAfter(block), this});
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::InitializeBlockDependencies(
    mlir::Block *block) {
  if constexpr (direction == DataflowDirection::kForward) {
    // For each block, we should update its successor blocks when the state
    // at the end of the block updates. Thus, we enumerate each predecessor's
    // end state and link it to the block.
    for (mlir::Block *pred : block->getPredecessors()) {
      JsirStateRef<StateT> pred_state_ref = GetStateAtEndOf(pred);
      pred_state_ref.AddDependent(getProgramPointBefore(block));
    }
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // For each block, we should update its predecessor blocks when the state
    // at the end of the block updates. Thus, we enumerate each successor's
    // end state and link it to the block.
    for (mlir::Block *succ : block->getSuccessors()) {
      JsirStateRef<StateT> succ_state_ref = GetStateAtEntryOf(succ);
      succ_state_ref.AddDependent(getProgramPointBefore(block));
    }
  }
}

template <typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDenseDataFlowAnalysis<StateT, direction>::visit(
    mlir::ProgramPoint *point) {
  if (!point->isBlockStart()) {
    VisitOp(point->getPrevOp());
  } else if (!point->isNull()) {
    VisitBlock(point->getBlock());
  }
  return mlir::success();
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::VisitOp(
    mlir::Operation *op) {
  if constexpr (direction == DataflowDirection::kForward) {
    JsirStateRef<StateT> before_state_ref = GetStateBefore(op);
    const StateT *before = &before_state_ref.value();

    JsirStateRef after_state_ref = GetStateAfter(op);

    VisitOp(op, before, after_state_ref);
  } else if constexpr (direction == DataflowDirection::kBackward) {
    JsirStateRef<StateT> after_state_ref = GetStateAfter(op);
    const StateT *after = &after_state_ref.value();

    JsirStateRef before_state_ref = GetStateBefore(op);

    VisitOp(op, after, before_state_ref);
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::VisitBlock(
    mlir::Block *block) {
  for (auto *edge : block_to_cfg_edges_[block]) {
    VisitCFGEdge(edge);
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::VisitCFGEdge(
    JsirGeneralCfgEdge *edge) {
  JsirStateRef<StateT> pred_state_ref = GetStateImpl<StateT>(edge->getPred());
  JsirStateRef<StateT> succ_state_ref = GetStateImpl<StateT>(edge->getSucc());

  if constexpr (direction == DataflowDirection::kForward) {
    // Merge the predecessor into the successor.
    pred_state_ref.AddDependent(edge->getSucc());
    succ_state_ref.Join(pred_state_ref.value());
  } else if constexpr (direction == DataflowDirection::kBackward) {
    // Merge the successor into the predecessor.
    succ_state_ref.AddDependent(edge->getPred());
    pred_state_ref.Join(succ_state_ref.value());
  }
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintAtBlockEntry(
    mlir::Block &block, size_t num_indents, llvm::raw_ostream &os) {
  os.indent(num_indents + 2);
  os << "// ";
  GetStateAtEntryOf(&block).value().print(os);
  os << "\n";
}

template <typename StateT, DataflowDirection direction>
void JsirDenseDataFlowAnalysis<StateT, direction>::PrintAfterOp(
    mlir::Operation *op, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  os << "\n";
  os.indent(num_indents + 2);
  os << "// ";
  GetStateAfter(op).value().print(os);
}

// =============================================================================
// JsirDataFlowAnalysis
// =============================================================================

template <typename ValueT, typename StateT, DataflowDirection direction>
JsirStateRef<ValueT>
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetStateAt(mlir::Value value) {
  return Base::template GetStateImpl<ValueT>(value);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::PrintAfterOp(
    mlir::Operation *op, size_t num_indents, mlir::AsmState &asm_state,
    llvm::raw_ostream &os) {
  for (mlir::Value result : op->getResults()) {
    auto result_state_ref = GetStateAt(result);

    os << "\n";
    os.indent(num_indents + 2);
    os << "// ";
    result.printAsOperand(os, asm_state);
    os << " = ";
    result_state_ref.value().print(os);
  }

  Base::PrintAfterOp(op, num_indents, asm_state, os);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
mlir::LogicalResult JsirDataFlowAnalysis<ValueT, StateT, direction>::initialize(
    mlir::Operation *op) {
  // The op depends on its input operands.
  for (mlir::Value operand : op->getOperands()) {
    JsirStateRef<ValueT> operand_state_ref = GetStateAt(operand);
    operand_state_ref.AddDependent(Base::getProgramPointAfter(op));
  }

  return Base::initialize(op);
}

template <typename ValueT, typename StateT, DataflowDirection direction>
typename JsirDataFlowAnalysis<ValueT, StateT, direction>::ValueStateRefs
JsirDataFlowAnalysis<ValueT, StateT, direction>::GetValueStateRefs(
    mlir::Operation *op) {
  if constexpr (direction == DataflowDirection::kForward) {
    std::vector<const ValueT *> operands;
    for (mlir::Value operand : op->getOperands()) {
      auto operand_state_ref = GetStateAt(operand);
      operands.push_back(&operand_state_ref.value());
    }

    std::vector<JsirStateRef<ValueT>> result_state_refs;
    for (size_t i = 0; i != op->getNumResults(); ++i) {
      mlir::Value result = op->getResult(i);
      JsirStateRef<ValueT> result_state_ref = GetStateAt(result);
      result_state_refs.push_back(std::move(result_state_ref));
    }

    return ValueStateRefs{
        .inputs = std::move(operands),
        .outputs = std::move(result_state_refs),
    };
  } else if constexpr (direction == DataflowDirection::kBackward) {
    std::vector<const ValueT *> results;
    for (size_t i = 0; i != op->getNumResults(); ++i) {
      mlir::Value result = op->getResult(i);
      auto result_state_ref = GetStateAt(result);
      results.push_back(&result_state_ref.value());
    }
    std::vector<JsirStateRef<ValueT>> operand_state_refs;
    for (mlir::Value operand : op->getOperands()) {
      JsirStateRef<ValueT> operand_state_ref = GetStateAt(operand);
      operand_state_refs.push_back(std::move(operand_state_ref));
    }

    return ValueStateRefs{
        .outputs = std::move(operand_state_refs),
        .inputs = std::move(results),
    };
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::VisitOp(
    mlir::Operation *op, const StateT *input, JsirStateRef<StateT> output) {
  if constexpr (direction == DataflowDirection::kForward) {
    auto [operands, result_state_refs] = GetValueStateRefs(op);
    return VisitOp(op, operands, input, result_state_refs, output);
  } else if constexpr (direction == DataflowDirection::kBackward) {
    auto [results, operand_state_refs] = GetValueStateRefs(op);
    return VisitOp(op, results, input, operand_state_refs, output);
  }
}

template <typename ValueT, typename StateT, DataflowDirection direction>
void JsirDataFlowAnalysis<ValueT, StateT, direction>::VisitCFGEdge(
    JsirGeneralCfgEdge *edge) {
  // Match arguments from the predecessor to the successor.
  for (const auto [pred_value, succ_value] :
       llvm::zip(edge->getPredValues(), edge->getSuccValues())) {
    CHECK(pred_value != nullptr);

    JsirStateRef<ValueT> succ_state_ref = GetStateAt(succ_value);
    JsirStateRef<ValueT> pred_state_ref = GetStateAt(pred_value);

    if constexpr (direction == DataflowDirection::kForward) {
      succ_state_ref.Join(pred_state_ref.value());
    } else if constexpr (direction == DataflowDirection::kBackward) {
      pred_state_ref.Join(succ_state_ref.value());
    }
  }

  Base::VisitCFGEdge(edge);
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_ANALYSES_DATAFLOW_ANALYSIS_H_

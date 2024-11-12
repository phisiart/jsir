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

#ifndef MALDOCA_JS_IR_JSLIR_VISITOR_H_
#define MALDOCA_JS_IR_JSLIR_VISITOR_H_

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

#define FOR_EACH_JSLIR_OP(LIR_OP_HANDLER)                                      \
  FOR_EACH_JSIR_CLASS(/*CIR_OP=*/JSIR_CLASS_IGNORE,                            \
                      /*HIR_OP=*/JSIR_CLASS_IGNORE, /*LIR_OP=*/LIR_OP_HANDLER, \
                      /*REF_OP=*/JSIR_CLASS_IGNORE,                            \
                      /*ATTRIB=*/JSIR_CLASS_IGNORE)

template <typename Ret, typename... Args>
class JslirAbstractVisitor {
 public:
  virtual ~JslirAbstractVisitor() = default;

  // The following code expands into:
  //
  // virtual Ret VisitControlFlowStarter(JslirControlFlowStarterOp op,
  //                                     Args ...args) = 0;
  // ...

#define DECLARE_JSLIR_ABSTRACT_VISITOR_VISIT_FUNC(OP) \
  virtual Ret Visit##OP(Jslir##OP##Op op, Args ...args) = 0;

  FOR_EACH_JSLIR_OP(DECLARE_JSLIR_ABSTRACT_VISITOR_VISIT_FUNC)
};

template <typename Ret, typename... VisitorArgs, typename... Args>
void AttachJslirVisitor(llvm::TypeSwitch<mlir::Operation *, Ret> &type_switch,
                        JslirAbstractVisitor<Ret, VisitorArgs...> &visitor,
                        Args &&...args) {
  // The following code expands into:
  //
  // type_switch.Case([&](JslirControlFlowStarterOp op) {
  //   return VisitControlFlowStarter(op, std::forward<Args>(args)...);
  // });
  // type_switch.Case([&](JslirControlFlowMarkerOp op)) {
  //   return VisitControlFlowMarker(op, std::forward<Args>(args)...);
  // });
  // ...

#define ADD_JSLIR_VISITOR_TYPE_SWITCH_CASE(OP)                 \
  type_switch.Case([&](Jslir##OP##Op op) {                     \
    return visitor.Visit##OP(op, std::forward<Args>(args)...); \
  });

  FOR_EACH_JSLIR_OP(ADD_JSLIR_VISITOR_TYPE_SWITCH_CASE)
}

template <typename Ret, typename... Args>
class JslirVisitor : public JslirAbstractVisitor<Ret, Args...> {
 public:
  virtual ~JslirVisitor() = default;

  virtual Ret VisitJslirOpDefault(mlir::Operation *op, Args &&...args) = 0;

  // The following code expands into:
  //
  // Ret VisitControlFlowStarter(JslirControlFlowStarterOp op,
  //                             Args ...args) override {
  //   return VisitJslirOpDefault(op, std::forward<Args>(args)...);
  // }
  // ...

#define DECLARE_JSLIR_VISITOR_VISIT_FUNC(OP)                     \
  Ret Visit##OP(Jslir##OP##Op op, Args... args) override {       \
    return VisitJslirOpDefault(op, std::forward<Args>(args)...); \
  }

  FOR_EACH_JSLIR_OP(DECLARE_JSLIR_VISITOR_VISIT_FUNC)
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_JSLIR_VISITOR_H_

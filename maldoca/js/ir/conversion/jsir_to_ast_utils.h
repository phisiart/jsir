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

#ifndef MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_UTILS_H_
#define MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// Including the `JsSourceLocation` struct, and the `start` and `end` fields.
struct AstSourceLocationInfo {
  std::optional<std::unique_ptr<JsSourceLocation>> loc;
  std::optional<int64_t> start;
  std::optional<int64_t> end;
  std::optional<int64_t> scope_uid;
  std::optional<std::vector<std::unique_ptr<JsComment>>> leading_comments;
  std::optional<std::vector<std::unique_ptr<JsComment>>> trailing_comments;
  std::optional<std::vector<std::unique_ptr<JsComment>>> inner_comments;
};

AstSourceLocationInfo GetAstLocationFromIrLocationAttr(
    JsirCommentsAndLocationAttr mlir_loc);

AstSourceLocationInfo GetAstLocationFromIr(mlir::Operation *op);

AstSourceLocationInfo GetAstLocationFromIr(mlir::Attribute attr);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CONVERSION_JSIR_TO_AST_UTILS_H_

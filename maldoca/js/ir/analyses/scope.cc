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

#include "maldoca/js/ir/analyses/scope.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

std::optional<int64_t> FindSymbol(const BabelScopes &scopes, int64_t scope_uid,
                                  absl::string_view name) {
  auto scope_it = scopes.scopes().find(scope_uid);
  if (scope_it == scopes.scopes().end()) {
    return std::nullopt;
  }
  const auto &scope = scope_it->second;

  auto binding_it = scope.bindings().find(name);
  if (binding_it != scope.bindings().end()) {
    return scope_uid;
  }

  // If this scope has no parent, then this is the root, stop searching.
  if (!scope.has_parent_uid()) {
    return std::nullopt;
  }
  // Stop-gap: If parent_uid() defaults to 0, then we will be stuck in an
  // infinite loop, so also check whether this scope is the root (0).
  if (scope_uid == 0) {
    return std::nullopt;
  }

  return FindSymbol(scopes, scope.parent_uid(), name);
}

std::optional<int64_t> FindSymbol(const BabelScopes &scopes,
                                  mlir::Operation *op, absl::string_view name) {
  auto loc = op->getLoc().dyn_cast<JsirCommentsAndLocationAttr>();
  if (loc == nullptr) {
    return std::nullopt;
  }

  if (!loc.getLoc().getScopeUid().has_value()) {
    return std::nullopt;
  }

  return FindSymbol(scopes, *loc.getLoc().getScopeUid(), name);
}

JsirSymbolId GetSymbolId(const BabelScopes &scopes, int64_t scope_uid,
                         absl::string_view name) {
  return JsirSymbolId{name, FindSymbol(scopes, scope_uid, name).value_or(0)};
}

JsirSymbolId GetSymbolId(const BabelScopes &scopes, mlir::Operation *op,
                         absl::string_view name) {
  return JsirSymbolId{name, FindSymbol(scopes, op, name).value_or(0)};
}

JsirSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierOp op) {
  return GetSymbolId(scopes, op, op.getName());
}

JsirSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierRefOp op) {
  return GetSymbolId(scopes, op, op.getName());
}

JsirSymbolId GetSymbolId(const BabelScopes &scopes, JsirIdentifierAttr attr) {
  int64_t scope_uid = [&]() -> int64_t {
    JsirCommentsAndLocationAttr loc = attr.getLoc();
    if (loc == nullptr) {
      return 0;
    }
    return loc.getLoc().getScopeUid().value_or(0);
  }();

  return GetSymbolId(scopes, scope_uid, attr.getName().strref());
}

}  // namespace maldoca

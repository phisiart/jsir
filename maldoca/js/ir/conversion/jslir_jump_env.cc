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

#include "maldoca/js/ir/conversion/jslir_jump_env.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace maldoca {
namespace internal {

absl::StatusOr<mlir::Block *> JslirJumpEnvScope::break_target() const {
  for (const auto &info : llvm::reverse(targets_stack_)) {
    if (info.break_target.has_value()) {
      return info.break_target.value();
    }
  }
  return absl::NotFoundError("Cannot find unlabeled break target.");
}

absl::StatusOr<mlir::Block *> JslirJumpEnvScope::break_target(
    mlir::StringAttr label) const {
  auto it = matched_labels_.find(label);
  if (it == matched_labels_.end()) {
    if (unmatched_labels_.contains(label)) {
      return absl::NotFoundError(
          "Label is not matched to a control flow structure.");
    }
    return absl::NotFoundError("Cannot find label.");
  }

  const JslirLabeledJumpTargets &info = it->second;
  return info.break_target;
}

absl::StatusOr<mlir::Block *> JslirJumpEnvScope::continue_target() const {
  for (const auto &info : llvm::reverse(targets_stack_)) {
    if (info.continue_target.has_value()) {
      return info.continue_target.value();
    }
  }
  return absl::NotFoundError("Cannot find unlabeled break target.");
}

absl::StatusOr<mlir::Block *> JslirJumpEnvScope::continue_target(
    mlir::StringAttr label) const {
  auto it = matched_labels_.find(label);
  if (it == matched_labels_.end()) {
    if (unmatched_labels_.contains(label)) {
      return absl::NotFoundError(
          "Label is not matched to a control flow structure.");
    }
    return absl::NotFoundError("Cannot find label.");
  }

  JslirLabeledJumpTargets info = it->second;
  if (!info.continue_target.has_value()) {
    return absl::NotFoundError("Labeled statement does not support continue.");
  }
  return info.continue_target.value();
}

}  // namespace internal
}  // namespace maldoca

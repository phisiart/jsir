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

#include "maldoca/js/ir/jsir_utils.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

absl::StatusOr<mlir::Operation *> GetStmtRegionOperation(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  return &block.back();
}

absl::StatusOr<mlir::Block *> GetStmtsRegionBlock(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  return &block;
}

absl::StatusOr<mlir::Value> GetExprRegionValue(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  auto expr_region_end = llvm::dyn_cast<JsirExprRegionEndOp>(block.back());
  if (expr_region_end == nullptr) {
    return absl::InvalidArgumentError(
        "Block should end with JsirExprRegionEndOp.");
  }
  return expr_region_end.getArgument();
}

absl::StatusOr<mlir::ValueRange> GetExprsRegionValues(mlir::Region &region) {
  if (!region.hasOneBlock()) {
    return absl::InvalidArgumentError("Region should have exactly one block.");
  }
  mlir::Block &block = region.front();
  if (block.empty()) {
    return absl::InvalidArgumentError("Block cannot be empty.");
  }
  auto exprs_region_end = llvm::dyn_cast<JsirExprsRegionEndOp>(block.back());
  if (exprs_region_end == nullptr) {
    return absl::InvalidArgumentError(
        "Block should end with JsirExprsRegionEndOp.");
  }
  return exprs_region_end.getArguments();
}

}  // namespace maldoca

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

#include "maldoca/js/ir/conversion/jsir_to_ast_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

std::unique_ptr<JsPosition> GetJsPositionFromIr(JsirPositionAttr ir_pos) {
  const auto loc_start_line = ir_pos.getLine();
  const auto loc_start_column = ir_pos.getColumn();
  return absl::make_unique<JsPosition>(loc_start_line, loc_start_column);
}

std::vector<std::unique_ptr<JsComment>> GetJsirCommentsFromIrs(
    std::vector<JsirCommentAttr> ir_comments) {
  std::vector<std::unique_ptr<JsComment>> comments =
      std::vector<std::unique_ptr<JsComment>>();
  for (JsirCommentAttr ir_comment : ir_comments) {
    std::unique_ptr<JsSourceLocation> loc;
    JsirLocationAttr mlir_loc = ir_comment.getLoc();
    {
      std::optional<std::unique_ptr<JsPosition>> start;
      if (JsirPositionAttr mlir_start = mlir_loc.getStart();
          mlir_start != nullptr) {
        start = GetJsPositionFromIr(mlir_start);
      }

      std::optional<std::unique_ptr<JsPosition>> end;
      if (JsirPositionAttr mlir_end = mlir_loc.getEnd(); mlir_end != nullptr) {
        end = GetJsPositionFromIr(mlir_end);
      }

      std::optional<std::string> identifier_name;
      if (mlir::StringAttr mlir_identifier_name = mlir_loc.getIdentifierName();
          mlir_identifier_name != nullptr) {
        identifier_name = mlir_identifier_name.str();
      }

      if (start.has_value() && end.has_value()) {
        loc = absl::make_unique<JsSourceLocation>(
            /*start=*/std::move(*start),
            /*end=*/std::move(*end),
            /*identifier_name=*/std::move(identifier_name));
      }
    }

    std::string value = ir_comment.getValue().str();
    int64_t start = mlir_loc.getStartIndex().value();
    int64_t end = mlir_loc.getEndIndex().value();

    if (ir_comment.getCommentType().str() == "CommentLine") {
      comments.push_back(
          absl::make_unique<JsCommentLine>(std::move(loc), value, start, end));
    } else if (ir_comment.getCommentType().str() == "CommentBlock") {
      comments.push_back(
          absl::make_unique<JsCommentBlock>(std::move(loc), value, start, end));
    }
  }
  return comments;
}

std::optional<JsirCommentsAndLocationAttr> GetJsirCommentsAndLocationAttr(
    mlir::Attribute attr) {
  return llvm::TypeSwitch<mlir::Attribute,
                          std::optional<JsirCommentsAndLocationAttr>>(attr)
      .Case([&](JsirStringLiteralAttr attr) { return attr.getLoc(); })
      .Case([&](JsirNumericLiteralAttr attr) { return attr.getLoc(); })
      .Case([&](JsirIdentifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirPrivateNameAttr attr) { return attr.getLoc(); })
      .Case([&](JsirImportSpecifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirImportDefaultSpecifierAttr attr) { return attr.getLoc(); })
      .Case(
          [&](JsirImportNamespaceSpecifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirExportSpecifierAttr attr) { return attr.getLoc(); })
      .Case([&](JsirInterpreterDirectiveAttr attr) { return attr.getLoc(); })
      .Default([&](mlir::Attribute attr) {
        LOG(INFO) << "Unexpected mlir::Attribute to get source location from. "
                  << "Maybe we missed a type cast here!";
        return std::nullopt;
      });
}

AstSourceLocationInfo GetAstLocationFromIr(mlir::Operation *op) {
  const auto milr_loc =
      llvm::dyn_cast<JsirCommentsAndLocationAttr>(op->getLoc());
  if (milr_loc == nullptr) {
    return AstSourceLocationInfo{};
  }
  return GetAstLocationFromIrLocationAttr(milr_loc);
}

AstSourceLocationInfo GetAstLocationFromIrLocationAttr(
    JsirCommentsAndLocationAttr mlir_loc) {
  std::optional<std::unique_ptr<JsSourceLocation>> loc;
  {
    std::optional<std::unique_ptr<JsPosition>> start;
    if (JsirPositionAttr mlir_start = mlir_loc.getLoc().getStart();
        mlir_start != nullptr) {
      start = GetJsPositionFromIr(mlir_start);
    }

    std::optional<std::unique_ptr<JsPosition>> end;
    if (JsirPositionAttr mlir_end = mlir_loc.getLoc().getEnd();
        mlir_end != nullptr) {
      end = GetJsPositionFromIr(mlir_end);
    }

    std::optional<std::string> identifier_name;
    if (mlir::StringAttr mlir_identifier_name =
            mlir_loc.getLoc().getIdentifierName();
        mlir_identifier_name != nullptr) {
      identifier_name = mlir_identifier_name.str();
    }

    if (start.has_value() && end.has_value()) {
      loc = absl::make_unique<JsSourceLocation>(
          /*start=*/std::move(*start),
          /*end=*/std::move(*end),
          /*identifier_name=*/std::move(identifier_name));
    }
  }

  std::optional<std::vector<std::unique_ptr<JsComment>>> leading_comments;
  if (llvm::ArrayRef<JsirCommentAttr> mlir_leading_comments =
          mlir_loc.getLeadingComments();
      !mlir_leading_comments.empty()) {
    leading_comments = GetJsirCommentsFromIrs(mlir_leading_comments.vec());
  }
  std::optional<std::vector<std::unique_ptr<JsComment>>> trailing_comments;
  if (llvm::ArrayRef<JsirCommentAttr> mlir_trailing_comments =
          mlir_loc.getTrailingComments();
      !mlir_trailing_comments.empty()) {
    trailing_comments = GetJsirCommentsFromIrs(mlir_trailing_comments.vec());
  }
  std::optional<std::vector<std::unique_ptr<JsComment>>> inner_comments;
  if (llvm::ArrayRef<JsirCommentAttr> mlir_inner_comments =
          mlir_loc.getInnerComments();
      !mlir_inner_comments.empty()) {
    inner_comments = GetJsirCommentsFromIrs(mlir_inner_comments.vec());
  }
  return AstSourceLocationInfo{
      .loc = std::move(loc),
      .start = mlir_loc.getLoc().getStartIndex(),
      .end = mlir_loc.getLoc().getEndIndex(),
      .scope_uid = mlir_loc.getLoc().getScopeUid(),
      .leading_comments = std::move(leading_comments),
      .trailing_comments = std::move(trailing_comments),
      .inner_comments = std::move(inner_comments),
  };
}

AstSourceLocationInfo GetAstLocationFromIr(mlir::Attribute attr) {
  auto ir_loc_attr = GetJsirCommentsAndLocationAttr(attr);
  if (ir_loc_attr.has_value()) {
    return GetAstLocationFromIrLocationAttr(ir_loc_attr.value());
  } else {
    return AstSourceLocationInfo{};
  }
}

}  // namespace maldoca

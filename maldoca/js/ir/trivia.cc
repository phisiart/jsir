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

#include "maldoca/js/ir/trivia.h"

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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {
namespace {

JsirPositionAttr JsPosition2JsirPositionAttr(mlir::MLIRContext *context,
                                             const JsPosition &pos) {
  return JsirPositionAttr::get(context, pos.line(), pos.column());
}

std::unique_ptr<JsPosition> JsirPositionAttr2JsPosition(JsirPositionAttr attr) {
  const auto loc_start_line = attr.getLine();
  const auto loc_start_column = attr.getColumn();
  return absl::make_unique<JsPosition>(loc_start_line, loc_start_column);
}

JsirSymbolIdAttr GetJsirSymbolIdAttr(mlir::MLIRContext *context,
                                     const JsSymbolId &symbol_id) {
  return JsirSymbolIdAttr::get(context,
                               mlir::StringAttr::get(context, symbol_id.name()),
                               symbol_id.def_scope_uid());
}

JsirLocationAttr GetJsirLocationAttr(mlir::MLIRContext *context,
                                     const JsSourceLocation *loc,
                                     std::optional<int64_t> start_index,
                                     std::optional<int64_t> end_index,
                                     std::optional<int64_t> scope_uid) {
  JsirPositionAttr start = JsPosition2JsirPositionAttr(context, *loc->start());
  JsirPositionAttr end = JsPosition2JsirPositionAttr(context, *loc->end());
  const auto identifier_name = loc->identifier_name();
  mlir::StringAttr jsir_identifier_name = nullptr;
  if (identifier_name.has_value()) {
    jsir_identifier_name = mlir::StringAttr::get(context, *identifier_name);
  }

  return JsirLocationAttr::get(context, start, end, jsir_identifier_name,
                               start_index, end_index, scope_uid);
}

std::vector<JsirCommentAttr> JsComments2JsirCommentAttrs(
    mlir::MLIRContext *context,
    std::optional<const std::vector<std::unique_ptr<JsComment>> *> js_comments,
    std::optional<int64_t> scope_uid) {
  if (!js_comments.has_value()) {
    return {};
  }

  std::vector<JsirCommentAttr> mlir_comments;
  for (const auto &comment : **js_comments) {
    JsirLocationAttr jsir_loc = GetJsirLocationAttr(
        context, comment->loc(), comment->start(), comment->end(), scope_uid);

    absl::string_view type;
    switch (comment->comment_type()) {
      case JsCommentType::kCommentLine:
        type = "CommentLine";
        break;
      case JsCommentType::kCommentBlock:
        type = "CommentBlock";
        break;
    }

    auto comment_attr = JsirCommentAttr::get(
        context, jsir_loc, mlir::StringAttr::get(context, comment->value()),
        mlir::StringAttr::get(context, type));

    mlir_comments.push_back(comment_attr);
  }

  return mlir_comments;
}

std::vector<std::unique_ptr<JsComment>> JsirCommentAttrs2JsComments(
    absl::Span<const JsirCommentAttr> ir_comments) {
  std::vector<std::unique_ptr<JsComment>> comments;
  for (JsirCommentAttr ir_comment : ir_comments) {
    std::unique_ptr<JsSourceLocation> loc;
    JsirLocationAttr mlir_loc = ir_comment.getLoc();
    {
      std::optional<std::unique_ptr<JsPosition>> start;
      if (JsirPositionAttr mlir_start = mlir_loc.getStart();
          mlir_start != nullptr) {
        start = JsirPositionAttr2JsPosition(mlir_start);
      }

      std::optional<std::unique_ptr<JsPosition>> end;
      if (JsirPositionAttr mlir_end = mlir_loc.getEnd(); mlir_end != nullptr) {
        end = JsirPositionAttr2JsPosition(mlir_end);
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

JsirTriviaAttr GetJsirTriviaAttr(mlir::Attribute attr) {
  return llvm::TypeSwitch<mlir::Attribute, JsirTriviaAttr>(attr)
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
        return nullptr;
      });
}

}  // namespace

JsirTriviaAttr GetJsirTriviaAttr(mlir::MLIRContext *context,
                                 const JsNode &node) {
  JsirLocationAttr jsir_location = nullptr;
  if (node.loc().has_value()) {
    jsir_location =
        GetJsirLocationAttr(context, node.loc().value(), node.start(),
                            node.end(), node.scope_uid());
  }

  std::vector<JsirCommentAttr> mlir_leading_comments =
      JsComments2JsirCommentAttrs(context, node.leading_comments(),
                                  node.scope_uid());

  std::vector<JsirCommentAttr> mlir_trailing_comments =
      JsComments2JsirCommentAttrs(context, node.trailing_comments(),
                                  node.scope_uid());

  std::vector<JsirCommentAttr> mlir_inner_comments =
      JsComments2JsirCommentAttrs(context, node.inner_comments(),
                                  node.scope_uid());

  JsirSymbolIdAttr mlir_referenced_symbol = nullptr;
  if (node.referenced_symbol().has_value()) {
    mlir_referenced_symbol =
        GetJsirSymbolIdAttr(context, **node.referenced_symbol());
  }

  std::vector<JsirSymbolIdAttr> mlir_defined_symbols;
  if (node.defined_symbols().has_value()) {
    for (const auto &defined_symbol : **node.defined_symbols()) {
      mlir_defined_symbols.push_back(
          GetJsirSymbolIdAttr(context, *defined_symbol));
    }
  }

  return JsirTriviaAttr::get(context, jsir_location, mlir_leading_comments,
                             mlir_trailing_comments, mlir_inner_comments,
                             mlir_referenced_symbol, mlir_defined_symbols);
}

JsTrivia GetJsTrivia(mlir::Operation *op) {
  const auto mlir_trivia = llvm::dyn_cast<JsirTriviaAttr>(op->getLoc());
  if (mlir_trivia == nullptr) {
    return JsTrivia{};
  }
  return JsirTriviaAttr2JsTrivia(mlir_trivia);
}

JsTrivia JsirTriviaAttr2JsTrivia(JsirTriviaAttr attr) {
  CHECK(attr.getLoc() != nullptr);

  std::optional<std::unique_ptr<JsSourceLocation>> loc;
  {
    std::optional<std::unique_ptr<JsPosition>> start;
    if (JsirPositionAttr mlir_start = attr.getLoc().getStart();
        mlir_start != nullptr) {
      start = JsirPositionAttr2JsPosition(mlir_start);
    }

    std::optional<std::unique_ptr<JsPosition>> end;
    if (JsirPositionAttr mlir_end = attr.getLoc().getEnd();
        mlir_end != nullptr) {
      end = JsirPositionAttr2JsPosition(mlir_end);
    }

    std::optional<std::string> identifier_name;
    if (mlir::StringAttr mlir_identifier_name =
            attr.getLoc().getIdentifierName();
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
          attr.getLeadingComments();
      !mlir_leading_comments.empty()) {
    leading_comments = JsirCommentAttrs2JsComments(mlir_leading_comments);
  }

  std::optional<std::vector<std::unique_ptr<JsComment>>> trailing_comments;
  if (llvm::ArrayRef<JsirCommentAttr> mlir_trailing_comments =
          attr.getTrailingComments();
      !mlir_trailing_comments.empty()) {
    trailing_comments = JsirCommentAttrs2JsComments(mlir_trailing_comments);
  }

  std::optional<std::vector<std::unique_ptr<JsComment>>> inner_comments;
  if (llvm::ArrayRef<JsirCommentAttr> mlir_inner_comments =
          attr.getInnerComments();
      !mlir_inner_comments.empty()) {
    inner_comments = JsirCommentAttrs2JsComments(mlir_inner_comments);
  }

  std::optional<std::unique_ptr<JsSymbolId>> referenced_symbol;
  if (JsirSymbolIdAttr mlir_referenced_symbol = attr.getReferencedSymbol()) {
    referenced_symbol =
        std::make_unique<JsSymbolId>(mlir_referenced_symbol.getName().str(),
                                     mlir_referenced_symbol.getDefScopeId());
  }

  std::optional<std::vector<std::unique_ptr<JsSymbolId>>> defined_symbols;
  if (llvm::ArrayRef<JsirSymbolIdAttr> mlir_defined_symbols =
          attr.getDefinedSymbols();
      !mlir_defined_symbols.empty()) {
    std::vector<std::unique_ptr<JsSymbolId>> defined_symbols_vec;
    for (JsirSymbolIdAttr mlir_defined_symbol : mlir_defined_symbols) {
      defined_symbols_vec.push_back(std::make_unique<JsSymbolId>(
          mlir_defined_symbol.getName().str(),
          mlir_defined_symbol.getDefScopeId()));
    }
    defined_symbols = std::move(defined_symbols_vec);
  }

  return JsTrivia{
      .loc = std::move(loc),
      .start = attr.getLoc().getStartIndex(),
      .end = attr.getLoc().getEndIndex(),
      .scope_uid = attr.getLoc().getScopeUid(),
      .referenced_symbol = std::move(referenced_symbol),
      .defined_symbols = std::move(defined_symbols),
      .leading_comments = std::move(leading_comments),
      .trailing_comments = std::move(trailing_comments),
      .inner_comments = std::move(inner_comments),
  };
}

JsTrivia GetJsTrivia(mlir::Attribute attr) {
  JsirTriviaAttr mlir_trivia = GetJsirTriviaAttr(attr);
  if (mlir_trivia == nullptr) {
    return JsTrivia{};
  }

  return JsirTriviaAttr2JsTrivia(mlir_trivia);
}

}  // namespace maldoca

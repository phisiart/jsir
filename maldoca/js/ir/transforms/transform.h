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

#ifndef MALDOCA_JS_IR_TRANSFORMS_TRANSFORM_H_
#define MALDOCA_JS_IR_TRANSFORMS_TRANSFORM_H_

#include <memory>
#include <vector>

#include "mlir/Pass/Pass.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.pb.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

// Creates the corresponding pass based on the given transform config.
absl::StatusOr<std::unique_ptr<mlir::Pass>> CreateJsirTransformPass(
    const JsAnalysisOutputs &analysis_outputs, const BabelScopes *scopes,
    JsirTransformConfig config, absl::Nullable<Babel *> babel);

// Performs a single transform on a JSHIR or JSLIR module.
absl::Status TransformJsir(const JsAnalysisOutputs &analysis_outputs,
                           JsirFileOp jsir_file, const BabelScopes &scopes,
                           JsirTransformConfig config,
                           absl::Nullable<Babel *> babel);

// Performs the given list of transforms on a JSHIR or JSLIR module.
absl::Status TransformJsir(const JsAnalysisOutputs &analysis_outputs,
                           JsirFileOp jsir_file, const BabelScopes &scopes,
                           std::vector<JsirTransformConfig> configs,
                           absl::Nullable<Babel *> babel);

// Converts the AST into JSHIR, performs the given list of transforms, and
// converts back to an AST.
absl::StatusOr<std::unique_ptr<JsFile>> TransformJsAst(
    const JsFile &ast, const BabelScopes &scopes,
    std::vector<JsirTransformConfig> configs, absl::Nullable<Babel *> babel);

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_TRANSFORM_H_

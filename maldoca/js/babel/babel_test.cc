// Copyright 2021 Google LLC
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

#include "maldoca/js/babel/babel_test.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "maldoca/base/testing/protocol-buffer-matchers.h"
#include "maldoca/base/testing/status_matchers.h"
#include "maldoca/js/babel/babel.h"

namespace maldoca {

using ::maldoca::testing::EqualsProto;
using ::testing::IsEmpty;
using ::testing::StrEq;
using ::testing::StrNe;

static constexpr char kSource[] = R"(console.log("Hello, Babel!");)";

static constexpr char kVarDef[] = R"(var a = 1;)";

TEST_P(BabelTest, ParseSimpleCode) {
  std::unique_ptr<Babel> babel = GetParam().babel_factory();
  BabelParseRequest request;
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      BabelParseResult result,
      babel->Parse(kSource, request, absl::InfiniteDuration()));

  EXPECT_THAT(result.ast_string.value(), StrNe(""));
  EXPECT_THAT(result.errors, EqualsProto(""));
}

TEST_P(BabelTest, ParseVarDef) {
  std::unique_ptr<Babel> babel = GetParam().babel_factory();
  BabelParseRequest request;
  request.set_compute_scopes(true);
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      BabelParseResult result,
      babel->Parse(kVarDef, request, absl::InfiniteDuration()));

  EXPECT_THAT(result.ast_string.value(), StrNe(""));
  EXPECT_THAT(result.ast_string.scopes(), EqualsProto(R"pb(
                scopes {
                  key: 0
                  value {
                    uid: 0
                    bindings {
                      key: "a"
                      value { kind: KIND_VAR name: "a" }
                    }
                  }
                })pb"));
}

TEST_P(BabelTest, ParseBrokenCode) {
  constexpr char kExpectedErrorMessage[] = "Unexpected token (1:3)";

  const auto expected_response =
      absl::StrFormat(R"pb(
                        errors {
                          name: "SyntaxError"
                          message: "%s"
                          loc { line: 1 column: 3 }
                        }
                      )pb",
                      absl::CEscape(kExpectedErrorMessage));

  std::unique_ptr<Babel> babel = GetParam().babel_factory();
  BabelParseRequest request;
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      BabelParseResult result,
      babel->Parse("-_-", request, absl::InfiniteDuration()));

  EXPECT_THAT(result.ast_string.value(), StrEq(""));
  EXPECT_THAT(result.errors, EqualsProto(expected_response));
}

TEST_P(BabelTest, ErrorRecovery) {
  constexpr char kRecoverableSource[] = R"(
let a = {
  __proto__: x,
  __proto__: y
}
)";

  constexpr char kExpectedErrorMessage[] =
      "Redefinition of __proto__ property. (4:2)";

  std::unique_ptr<Babel> babel = GetParam().babel_factory();
  BabelParseRequest request;
  {
    const auto expected_response =
        absl::StrFormat(R"pb(
                          errors {
                            name: "SyntaxError"
                            message: "%s"
                            loc { line: 4 column: 2 }
                          }
                        )pb",
                        absl::CEscape(kExpectedErrorMessage));

    MALDOCA_ASSERT_OK_AND_ASSIGN(
        BabelParseResult result,
        babel->Parse(kRecoverableSource, request, absl::InfiniteDuration()));

    EXPECT_THAT(result.ast_string.value(), StrEq(""));
    EXPECT_THAT(result.errors, EqualsProto(expected_response));
  }

  request.set_error_recovery(true);
  {
    MALDOCA_ASSERT_OK_AND_ASSIGN(
        BabelParseResult result,
        babel->Parse(kRecoverableSource, request, absl::InfiniteDuration()));

    EXPECT_THAT(result.ast_string.value(), StrNe(""));
    EXPECT_THAT(result.errors.errors(), IsEmpty());
  }
}

TEST_P(BabelTest, GenerateSimpleCode) {
  std::unique_ptr<Babel> babel = GetParam().babel_factory();
  BabelParseRequest request;
  MALDOCA_ASSERT_OK_AND_ASSIGN(
      BabelParseResult parse_result,
      babel->Parse(kSource, request, absl::InfiniteDuration()));

  MALDOCA_ASSERT_OK_AND_ASSIGN(
      BabelGenerateResult generate_result,
      babel->Generate(parse_result.ast_string, {}, absl::InfiniteDuration()));

  EXPECT_EQ(generate_result.source_code, kSource);
  EXPECT_EQ(generate_result.error, std::nullopt);
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(BabelTest);

}  // namespace maldoca

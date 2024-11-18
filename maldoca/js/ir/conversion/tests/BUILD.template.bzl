# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//bazel:lit.bzl", "glob_lit_tests")

package(default_applicable_licenses = ["//:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        "ast.json",
        "input.js",
        "jshir.mlir",
        "jslir.mlir",
        "output.js",
        "//maldoca/js/ir:lit_test_files",
    ],
    test_file_exts = [
        "lit",
    ],
)

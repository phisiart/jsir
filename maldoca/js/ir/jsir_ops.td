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

#ifndef MALDOCA_JS_IR_JSIR_TD_
#define MALDOCA_JS_IR_JSIR_TD_

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "maldoca/js/ir/interfaces.td"
include "maldoca/js/ir/jsir_attrs.td"
include "maldoca/js/ir/jsir_dialect.td"

// Import the generated ops.
include "maldoca/js/ir/jsir_ops.generated.td"

def JsirNoneOp : Jsir_Op<"none", [Pure]> {
  let results = (outs
    JsirAnyType
  );
}

def JslirControlFlowStarterOp : Jslir_Op<
    "control_flow_starter", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    JsirControlFlowKindAttr: $kind
  );

  let results = (outs
    JsirAnyType: $token
  );
}

def JslirControlFlowMarkerOp : Jslir_Op<
    "control_flow_marker", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    JsirControlFlowMarkerKindAttr: $kind,
    JsirAnyType: $token
  );
}

// BlockStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %token = jslir.control_flow_starter {BlockStatement}
//   jslir.control_flow_marker {BlockStatementDirectives} (%token)
//   ...  // directives
//   jslir.control_flow_marker {BlockStatementBody} (%token)
//   ...  // body
//   jslir.control_flow_marker {BlockStatementEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.block_statement ( {
//   ...  // body
// },  {
//   ...  // directives
// })

// WithStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %object = ...
//   %token = jslir.with_statement_start (%object)
//   cf.br [^body_bb]
//
// ^body_bb:
//   jslir.control_flow_marker {WithStatementBody} (%token)
//   ...  // body
//   cf.br [^end_bb]
//
// ^end_bb:
//   jslir.control_flow_marker {WithStatementEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// %object = ...
// jshir.with_statement (%object) ( {
//   ...  // body
// })

def JslirWithStatementStartOp : Jslir_Op<
    "with_statement_start", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $object
  );

  let results = (outs
    JsirAnyType: $token
  );
}

// LabeledStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
// %token = jslir.labeled_statement {label}
// ...  // body
// jslir.control_flow_marker {LabeledStatementEnd} (%token)
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.labeled_statement {label} ( {
//   ...  // body
// })

def JslirLabeledStatementStartOp : Jslir_Op<
    "labeled_statement_start", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    JsirIdentifierAttr: $label
  );

  let results = (outs
    JsirAnyType: $token
  );
}

// IfStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %token = jslir.control_flow_starter {IfStatement}
//   cf.cond_br (%test) [^consequent_bb, ^alternate_bb]
// ^consequent_bb:
//   jslir.control_flow_marker {IfStatementConsequent} (%token)
//   ...  // consequent
//   cf.br [^end_bb]
//
// ^alternate_bb:
//   jslir.control_flow_marker {IfStatementAlternate} (%token)
//   ...  // alternate
//   cf.br [^end_bb]
//
// ^end_bb:
//   jslir.control_flow_marker {IfStatementEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.if_statement (%test) ( {
//   ...  // consequent
// }, {
//   ...  // alternate
// })

// SwitchCase-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
// note: default branch doesn't need to occur at end
//
//   %switch_token = jslir.switch_statement_start (%discriminant)
//   cf.br()[^case0_test_bb]
//
// ^case0_test_bb:
//   %case0_token = jslir.switch_statement_case_start {0} (%switch_token)
//   %test0 = ...
//   jslir.switch_statement_case_test (%test0)
//   %cmp0 = jsir.binary_expression {"==="} (%discriminant, %test0)
//   cond_br(%cmp0) [%case0_consequent_bb, %case1_test_bb]
//
// ^case1_test_bb:
//   %case1_token = jslir.switch_statement_case_start {1} (%switch_token)
//   %test1 = ...
//   jslir.switch_statement_case_test (%test1)
//   %cmp1 = jsir.binary_expression {"==="} (%discriminant, %test1)
//   cond_br(%cmp1) [%case1_consequent_bb, %default_test_bb]
//
// ^case0_consequent_bb:
//   jslir.control_flow_marker {SwitchStatementCaseBody} (%case0_token)
//   ...  // body
//   br [^case1_consequent_bb] (or ^end_bb if break)
//
// ^case1_consequent_bb:
//   jslir.control_flow_marker {SwitchStatementCaseBody} (%case1_token)
//   ...  // body
//   br [^default_consequent_bb] (or ^end_bb if break)
//
// ^default_consequent_bb:
//   jslir.switch_statement_default_start {2} (%switch_token)
//   ...  // body
//   br [^end_bb]
//
// ^end_bb:
//  jslir.control_flow_marker {SwitchStatementEnd} (%switch_token)
//  ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.switch_statement (%discriminant) ({
//   jshir.switch_case() ({
//     %test0 = ...
//     jsir.expr_region_end (%test0)
//   }, {
//     ...  // body
//   })
//   jshir.switch_case() ({
//     %test1 = ...
//     jsir.expr_region_end (%test1)
//   }, {
//     ...  // body
//   })
//   jshir.switch_case() ({
//   }, {
//     ...  // body
//   })
// })

def JslirSwitchStatementStartOp : Jslir_Op<
    "switch_statement_start", [
        NoTerminator,
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $discriminant
  );
  let results = (outs
    JsirAnyType: $token
  );
}

def JslirSwitchStatementCaseStartOp : Jslir_Op<
    "switch_statement_case_start", [
        NoTerminator,
        JsirStatementOpInterfaceTraits,
    ]> {
  let arguments = (ins
    AnyType: $switch_token,
    UI32Attr: $case_idx
  );
  let results = (outs
    JsirAnyType: $case_token
  );
}

def JslirSwitchStatementDefaultStartOp : Jslir_Op<
    "switch_statement_default_start", [
        NoTerminator,
        JsirStatementOpInterfaceTraits,
    ]> {
  let arguments = (ins
    AnyType: $switch_token,
    UI32Attr: $case_idx
  );
}

def JslirSwitchStatementCaseTestOp : Jslir_Op<
    "switch_statement_case_test", [
        JsirStatementOpInterfaceTraits,
    ]> {
  let arguments = (ins
    AnyType: $test
  );
}

// TryStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %token = jslir.control_flow_starter (TryStatement)
//   cf.br [^body]
//
// ^body:
//   jslir.control_flow_marker (TryStatementBody)
//   ...
//   cf.br [^finalizer]
//
// ^handler:
//   jslir.control_flow_marker (TryStatementHandler)
//   %param = ...
//   jslir.catch_clause_start (%param)
//   ...
//   cf.br [^finalizer]
//
// ^finalizer:
//   jslir.control_flow_marker (TryStatementFinalizer)
//   ...
//   cf.br [^end]
//
// ^end:
//   ...

def JslirCatchClauseStartOp : Jslir_Op<
    "catch_clause_start", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    Optional<AnyType>: $param
  );
}

// BreakStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
// jslir.break_statement (%label optional argument)
// cf.br [^break_location]
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.break_statement (%label optional argument)

def JshirBreakStatementOp : Jshir_Op<
    "break_statement", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $label
  );
}

def JslirBreakStatementOp : Jslir_Op<
    "break_statement", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $label
  );
}

// ContinueStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
// jslir.continue_statement (%label optional argument)
// cf.br [^break_location]
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.continue_statement (%label optional argument)

def JshirContinueStatementOp : Jshir_Op<
    "continue_statement", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $label
  );
}

def JslirContinueStatementOp : Jslir_Op<
    "continue_statement", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $label
  );
}

// WhileStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %token = jslir.control_flow_starter {WhileStatement}
//   cf.br [^test_bb]
//
// ^test_bb:
//   jslir.control_flow_marker {WhileStatementTest} (%token)
//   %test = ...
//   cf.cond_br (%test) [^body_bb, ^end_bb]
//
// ^body_bb:
//   jslir.control_flow_marker {WhileStatementBody} (%token)
//   ...  // body
//   cf.br [^test_bb]
//
// ^end_bb:
//   jslir.control_flow_marker {WhileStatementEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.while_statement ( {
//   %test = ...
//   jsir.expr_region_end (%test)
// },  {
//   ...  // body
// })

// DoWhileStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %token = jslir.control_flow_starter {DoWhileStatement}
//   cf.br [^body_bb]
//
// ^body_bb:
//   ...  // body
//   cf.br [^test_bb]
//
// ^test_bb:
//   jslir.control_flow_marker {DoWhileStatementTest} (%token)
//   %test = ...
//   cf.cond_br (%test) [^body_bb, ^end_bb]
//
// ^end_bb:
//   jslir.control_flow_marker {DoWhileStatementEnd} (%token)
// ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.do_while_statement () ({
//   ...  // body
// }, {
//   %test = ...
//   jsir.expr_region_end (%test)
// })

// ForStatement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   %token = jslir.control_flow_starter {ForStatement}
//   cf.br [^init_bb]
//
// ^init_bb:
//   jslir.control_flow_marker {ForStatementInit} (%token)
//   %init = ...
//   cf.br [^test_bb]
//
// ^test_bb:
//   jslir.control_flow_marker {ForStatementTest} (%token)
//   %test = ...
//   cf.cond_br (%test) [^body_bb, ^end_bb]
//
// ^body_bb:
//   jslir.control_flow_marker {ForStatementBody} (%token)
//   ...  // body
//   cf.br [^update_bb]
//
// ^update_bb:
//   jslir.control_flow_marker {ForStatementUpdate} (%token)
//   ...  // update
//   cf.br [^test_bb]
//
// ^end_bb:
//   jslir.control_flow_marker {ForStatementEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// jshir.for_statement ( {
//   %init = ...
//   jsir.expr_region_end (%init)
// },  {
//   %test = ...
//   jsir.expr_region_end (%test)
// },  {
//   %update = ...
//   jsir.expr_region_update (%update)
// },  {
//   ...  // body
// })

def JshirForStatementOp : Jshir_Op<
    "for_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator
    ]> {
  let regions = (region
    OptionalRegion<UnknownRegion>: $init,
    OptionalRegion<ExprRegion>: $test,
    OptionalRegion<ExprRegion>: $update,
    StmtRegion: $body
  );
}

// For{In,Of}Statement-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
// ^init:
//   ...
//   %left = ...
//   %right = ...
//   %iter = jslir.for_in_statement_start(%left, %right)
//                                   - OR -
//   %iter = jslir.for_of_statement_start(%left, %right) {await}
//   cf.br()[^next]
//
// ^next:  // 2 preds: ^init, ^body
//   %has_next = jslir.for_in_of_statement_has_next(%iter)
//   cf.cond_br(%has_next)[^body, ^end]
//
// ^body:  // pred: ^next
//   jslir.for_in_of_statement_get_next(%iter)
//   ...
//   cf.br()[^next]
//
// ^end:  // pred: ^next
//   jslir.for_in_of_statement_end (%iter)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// %left = ...
// %right = ...
// jshir.for_in_statement (%left, %right) {
//   ...  // body
// })
//                                   - OR -
// jshir.for_of_statement (%left, %right) {
//   ...  // body
// }) {await = true}

def JshirForInStatementOp : Jshir_Op<
    "for_in_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirForInOfDeclarationAttr>: $left_declaration,
    AnyType: $left_lval,  // JsLVal
    AnyType: $right  // JsExpression
  );

  let regions = (region
    StmtRegion: $body
  );
}

def JshirForOfStatementOp : Jshir_Op<
    "for_of_statement", [
        JsirStatementOpInterfaceTraits,
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirForInOfDeclarationAttr>: $left_declaration,
    AnyType: $left_lval,  // JsLVal
    AnyType: $right,  // JsExpression
    BoolAttr: $await
  );

  let regions = (region
    StmtRegion: $body
  );
}

def JslirForInStatementStartOp : Jslir_Op<
    "for_in_statement_start", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirForInOfDeclarationAttr>: $left_declaration,
    AnyType: $left_lval,
    AnyType: $right
  );

  let results = (outs
    JsirAnyType: $iterator
  );
}

def JslirForOfStatementStartOp : Jslir_Op<
    "for_of_statement_start", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    OptionalAttr<JsirForInOfDeclarationAttr>: $left_declaration,
    AnyType: $left_lval,
    AnyType: $right,
    BoolAttr: $await
  );

  let results = (outs
    JsirAnyType: $iterator
  );
}

def JslirForInOfStatementHasNextOp : Jslir_Op<
    "for_in_of_statement_has_next", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $iterator
  );

  let results = (outs
    JsirAnyType
  );
}

def JslirForInOfStatementGetNextOp : Jslir_Op<
    "for_in_of_statement_get_next", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $iterator
  );
}

def JslirForInOfStatementEndOp : Jslir_Op<
    "for_in_of_statement_end", [
        JsirStatementOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $iterator
  );
}

def JsirArrowFunctionExpressionOp : Jsir_Op<
    "arrow_function_expression", [
        JsirExpressionOpInterfaceTraits,
        AttrSizedOperandSegments,
        NoTerminator
    ]> {
  let arguments = (ins
    Optional<AnyType>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async
  );

  let regions = (region
    UnknownRegion: $body
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirObjectLiteralKeyType : OptionalAttr<AnyAttrOf<[
    JsirIdentifierAttr,
    JsirStringLiteralAttr,
    JsirNumericLiteralAttr,
    JsirBigIntLiteralAttr
]>>;

def JsirObjectPropertyOp : Jsir_Op<"object_property", []> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $shorthand,
    AnyType: $value  // Expression
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirObjectPropertyRefOp : Jsir_Op<"object_property_ref", []> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $shorthand,
    AnyType: $value  // Pattern
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirObjectMethodOp : Jsir_Op<
    "object_method", [
        AttrSizedOperandSegments,
        NoTerminator
    ]> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    OptionalAttr<JsirIdentifierAttr>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async,
    StrAttr: $kind
  );

  let regions = (region
    StmtRegion: $body
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirObjectExpressionOp : Jsir_Op<
    "object_expression", [
        JsirExpressionOpInterfaceTraits,
        NoTerminator
    ]> {
  let regions = (region
    ExprsRegion: $region
  );

  let results = (outs
    JsirAnyType
  );
}

// - If `computed` is `true`, the op corresponds to a computed (`a[b]`) member
//   expression.
//   - `literal_property` is nullptr.
//   - `computed_property` is an `Expression`.
// - If `computed` is `false`, the op corresponds to a static (`a.b`) member
//   expression.
//   - `literal_property` is an `Identifier` or a `PrivateName`.
//   - `computed_property` is nullptr.
def JsirMemberExpressionOp : Jsir_Op<
    "member_expression", [
        JsirExpressionOpInterfaceTraits
    ]> {
  let hasFolder = 1;

  let arguments = (ins
    AnyType: $object,
    OptionalAttr<AnyAttrOf<[JsirIdentifierAttr, JsirPrivateNameAttr]>>: $literal_property,
    Optional<AnyType>: $computed_property
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirMemberExpressionRefOp : Jsir_Op<
    "member_expression_ref", [
        JsirPatternRefOpInterfaceTraits,
        JsirLValRefOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $object,
    OptionalAttr<AnyAttrOf<[JsirIdentifierAttr, JsirPrivateNameAttr]>>: $literal_property,
    Optional<AnyType>: $computed_property
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirOptionalMemberExpressionOp : Jsir_Op<
    "optional_member_expression", [
        JsirExpressionOpInterfaceTraits
    ]> {
  let arguments = (ins
    AnyType: $object,
    OptionalAttr<AnyAttrOf<[JsirIdentifierAttr, JsirPrivateNameAttr]>>: $literal_property,
    Optional<AnyType>: $computed_property,
    BoolAttr: $optional
  );

  let results = (outs
    JsirAnyType
  );
}

// LogicalExpression-Related Ops
//
// Note: Conceptually, logical expressions are equivalent to conditional
// expressions.
//
// a && b =>      a      ? b : a
// a || b =>      a      ? a : b
// a ?? b => (a != null) ? a : b
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   ...
//   %token = jslir.logical_expression_start {"||"} (%a)
//   cf.cond_br(%should_short_circuit) [^end_bb(%a), ^right_bb]
//
// ^right_bb:
//   jslir.control_flow_marker {LogicalExpressionRight} (%token)
//   %b = ...
//   cf.br (%b) [^end_bb]
//
// ^end_bb(%result):
//   jslir.control_flow_marker {LogicalExpressionEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// %a = jsir.identifier {"a"}
// %result = "jshir.logical_expression"(%a) ({
//   %b = jsir.identifier {"b"}
//   jsir.expr_region_end(%b)
// }) {operator_ = "||"}

def JslirLogicalExpressionStartOp : Jslir_Op<
    "logical_expression_start", []> {
  let arguments = (ins
    StrAttr: $operator_,
    AnyType: $left
  );

  let results = (outs
    JsirAnyType: $token
  );
}

// ConditionalExpression-Related Ops
//
// =============================================================================
// JSLIR
// =============================================================================
//
//   ...
//   %token = jslir.control_flow_starter {ConditionalExpression}
//   cf.cond_br(%test) [^alternate_bb, ^consequent_bb]
//
// ^alternate_bb:
//   jslir.control_flow_marker {ConditionalExpressionAlternate} (%token)
//   %alternate = ...
//   cf.br (%alternate) [^end_bb]
//
// ^consequent_bb:
//   jslir.control_flow_marker {ConditionalExpressionConsequent} (%token)
//   %consequent = ...
//   cf.br (%consequent) [^end_bb]
//
// ^end_bb(%result):
//   jslir.control_flow_marker {ConditionalExpressionEnd} (%token)
//   ...
//
// =============================================================================
// Equivalent JSHIR representation
// =============================================================================
//
// %result = jshir.conditional_expression (%test, {
//   %alternate = ...
//   jsir.expr_region_end (%alternate)
// },  {
//   %consequent = ...
//   jsir.expr_region_end (%consequent)
// })

def JsirParenthesizedExpressionOp : Jsir_Op<
    "parenthesized_expression", [
        JsirExpressionOpInterfaceTraits,
        Pure
    ]> {
  let hasFolder = 1;

  let arguments = (ins
    AnyType: $expression
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirParenthesizedExpressionRefOp : Jsir_Op<
    "parenthesized_expression_ref", [
        JsirPatternRefOpInterfaceTraits,
        JsirLValRefOpInterfaceTraits,
        Pure
    ]> {
  let hasFolder = 1;

  let arguments = (ins
    AnyType: $expression
  );

  let results = (outs
    JsirAnyType
  );
}

def JsirClassMethodOp : Jsir_Op<
    "class_method", [
        AttrSizedOperandSegments,
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async,
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    StrAttr: $kind,
    BoolAttr: $static_
  );

  let regions = (region
    StmtRegion: $body
  );
}

def JsirClassPrivateMethodOp : Jsir_Op<
    "class_private_method", [
        NoTerminator
    ]> {
  let arguments = (ins
    OptionalAttr<JsirIdentifierAttr>: $id,
    Variadic<AnyType>: $params,
    BoolAttr: $generator,
    BoolAttr: $async,
    JsirPrivateNameAttr: $key,
    StrAttr: $kind,
    BoolAttr: $static_
  );

  let regions = (region
    StmtRegion: $body
  );
}

def JsirClassPropertyOp : Jsir_Op<"class_property", [NoTerminator]> {
  let arguments = (ins
    JsirObjectLiteralKeyType: $literal_key,
    Optional<AnyType>: $computed_key,
    BoolAttr: $static_
  );

  let regions = (region
    OptionalRegion<ExprRegion>: $value
  );
}

def JsirExportDefaultDeclarationOp : Jsir_Op<
    "export_default_declaration", [
        JsirModuleDeclarationOpInterfaceTraits,
        NoTerminator
    ]> {
  let regions = (region
    UnknownRegion: $declaration
  );
}

#endif  // MALDOCA_JS_IR_JSIR_HANDWRITTEN_TD_

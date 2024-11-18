// JSLIR:      "jsir.file"() ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.none"() : () -> !jsir.any
// JSLIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.array_pattern_ref"(%0, %1, %2) : (!jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %4 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     %5 = "jsir.assignment_expression"(%3, %4) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()

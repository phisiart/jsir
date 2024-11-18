// JSHIR:      "jsir.file"() ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %1 = "jsir.none"() : () -> !jsir.any
// JSHIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:     %3 = "jsir.array_pattern_ref"(%0, %1, %2) : (!jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:     %4 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:     %5 = "jsir.assignment_expression"(%3, %4) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSHIR-NEXT:     "jsir.expression_statement"(%5) : (!jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()

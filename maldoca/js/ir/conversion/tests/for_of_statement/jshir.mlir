// JSHIR:      "jsir.file"() ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     %0 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.for_of_statement"(%0, %1) <{await = false}> ({
// JSHIR-NEXT:       %6 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSHIR-NEXT:     %2 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.for_of_statement"(%2, %3) <{await = false, left_declaration = #jsir<for_in_of_declaration   <L 4 C 5>, <L 4 C 10>, 24, 29, 2,   <L 4 C 9>, <L 4 C 10>, 28, 29, 2,  "a", 2, "let">}> ({
// JSHIR-NEXT:       %6 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSHIR-NEXT:     %4 = "jsir.identifier_ref"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:     %5 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:     "jshir.for_of_statement"(%4, %5) <{await = false}> ({
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %6 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%6) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : (!jsir.any, !jsir.any) -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()

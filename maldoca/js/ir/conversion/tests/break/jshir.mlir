// JSHIR:      "jsir.file"() ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jshir.while_statement"() ({
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:         %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:         "jshir.if_statement"(%1) ({
// JSHIR-NEXT:           "jshir.break_statement"() : () -> ()
// JSHIR-NEXT:         }, {
// JSHIR-NEXT:         }) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier   <L 15 C 0>, <L 15 C 6>, "label0", 435, 441, 0, "label0">}> ({
// JSHIR-NEXT:       "jshir.while_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         "jshir.block_statement"() ({
// JSHIR-NEXT:           %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:           "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:           "jshir.labeled_statement"() <{label = #jsir<identifier   <L 17 C 2>, <L 17 C 8>, "label1", 462, 468, 4, "label1">}> ({
// JSHIR-NEXT:             "jshir.while_statement"() ({
// JSHIR-NEXT:               %1 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSHIR-NEXT:               "jsir.expr_region_end"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:             }, {
// JSHIR-NEXT:               %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:               "jshir.if_statement"(%1) ({
// JSHIR-NEXT:                 "jshir.break_statement"() <{label = #jsir<identifier   <L 19 C 12>, <L 19 C 18>, "label0", 503, 509, 5, "label0">}> : () -> ()
// JSHIR-NEXT:               }, {
// JSHIR-NEXT:               }) : (!jsir.any) -> ()
// JSHIR-NEXT:             }) : () -> ()
// JSHIR-NEXT:           }) : () -> ()
// JSHIR-NEXT:         }, {
// JSHIR-NEXT:         ^bb0:
// JSHIR-NEXT:         }) : () -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier   <L 26 C 0>, <L 26 C 5>, "label", 713, 718, 0, "label">}> ({
// JSHIR-NEXT:       "jshir.break_statement"() <{label = #jsir<identifier   <L 26 C 13>, <L 26 C 18>, "label", 726, 731, 0, "label">}> : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()

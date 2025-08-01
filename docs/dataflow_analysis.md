---
layout: page
title: Dataflow Analysis
permalink: /dataflow_analysis/
---

# Dataflow Analysis

<!-- DO NOT SUBMIT without changing the go-link -->

go/g3doc-canonical-go-links

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'tzx' reviewed: '2025-08-01' }
*-->

[TOC]

## Overview

JSIR provides an API for **flow-sensitive, conditional**, dataflow analysis.

It is built on top of the MLIR dataflow analysis API with usability
improvements:

  * We provide a single API that uses sparse and dense states at the same time,
    whereas using the MLIR API would require writing two separate analyses.

  * We have a util API for updating analysis states that automatically trigger
    op visits, so the user doesn't have to manually propagate states.

### **Flow sensitive analysis**

**Flow-sensitive** means for the following code, we can reason about the two
program paths, and determine that, at the end of the `if`-statement, `a` has the
value `“same”`, and `b` has an unknown value.

```js
if (some_unknown_condition) {
  a = "same";
  b = "different 1";
} else {
  a = "same";
  b = "different 2";
}
// a == "same"
// b == <Unknown>
```

### **Conditional analysis**

**Conditional** means for the following code, we can skip the traversal of the
`else`-branch of the `if`-statement and determine that at the end of the
`if`-statement, `b` has the value `“different 1”`.

```js
if (true) {
  a = "same";
  b = "different 1";
} else {
  a = "same";
  b = "different 2";
}
// a == "same"
// b == "different 1"
```

## Example 1: Straightline code

> In this example, we demonstrate the basic concepts of dataflow analysis:
>
>   * Program points and values - where we attach states;
>
>   * Transfer function - what computes the states;
>
>   * Constant propagation - how these concepts are used in action.

Imagine that we want to run constant propagation analysis on the following code:

```js
a = 1;
b = a + 2;
```

As shown below, the analysis calculates the state at each program point, which
contains the value in each variable. In particular, we should determine that `b`
has the value of `3` after the assignment.

```js
// {}
a = 1;
// {a: Const{1}}
b = a + 2;
// {a: Const{1}, b: Const{3}}
```

Since the dataflow analysis algorithm runs on JSIR, we first convert the code
into JSIR:

```
// IR for `a = 1;`:
%a_ref = jsir.identifier_ref{"a"}
%1 = jsir.numeric_literal{1}
%a_assign = jsir.assignment_expression (%a_ref, %1)
jsir.expression_statement (%a_assign)

// IR for `b = a + 2;`
%b_ref = jsir.identifier_ref{"b"}
%a = jsir.identifier{"a"}
%2 = jsir.numeric_literal{2}
%add = jsir.binary_expression{"+"} (%a, %2)
%b_assign = jsir.assignment_expression (%b_ref, %add)
jsir.expression_statement (%b_assign)
```

We can see that JSIR looks like the result of a **post-order traversal** of the
abstract syntax tree (AST), and it’s intentional - we designed JSIR to be as
close to the AST as possible.

Now, the dataflow analysis algorithm will compute a **state** for every
**program point** and every **value**.

> ### Concepts: program points and values
>
>   * A **program point** is before or after each MLIR operation
>     (`mlir::Operation`).
>
>   * A **value** is an MLIR SSA value (`mlir::Value`), denoted by `%value`.
>
> In constant propagation, we design our **value state** as either `Uninit`,
> `Const`, or `Unknown`, **program point state** as a map from symbols to value
> states.

We expect that, after constant propagation, the states on all program points and
values should be as follows:

```
// state  0 = {}

%a_ref = jsir.identifier_ref{"a"}
// state[%a_ref] = Uninit
// state  1 = {}

%1 = jsir.numeric_literal{1}
// state[%1] = Const{1}
// state  2 = {}

%a_assign = jsir.assignment_expression (%a_ref, %1)
// state[%a_assign] = Const{1}
// state  3 = {a: Const{1}}

jsir.expression_statement (%a_assign)
// state  4 = {a: Const{1}}

%b_ref = jsir.identifier_ref{"b"}
// state[%b_ref] = Uninit
// state  5 = {a: Const{1}}

%a = jsir.identifier{"a"}
// state[%a] = Const{1}
// state  6 = {a: Const{1}}

%2 = jsir.numeric_literal{2}
// state[%2] = Const{2}
// state  7 = {a: Const{1}}

%add = jsir.binary_expression{"+"} (%a, %2)
// state[%add] = Const{3}
// state  8 = {a: Const{1}}

%b_assign = jsir.assignment_expression (%b_ref, %add)
// state[%b_assign] = Const{3}
// state  9 = {a: Const{1}, b: Const{3}}

jsir.expression_statement (%b_assign)
// state 10 = {a: Const{1}, b: Const{3}}
```

The dataflow analysis algorithm traverses (organized by a work queue) the IR and
computes all these states.

During the traversal, when we visit an operation, we will:

  * **Read** the state on the program point **before** the op, and states on its
    **input values**; and
  * **Write** the state on the program point after the op, and states on its
    **output values**.

> **Example 1.1:** When we visit the `jsir.binary_expression` op, we will:
>
>   * Read the state before the op, i.e. state 7, and the states on its input
>     values %a and %2; and
>
>   * Write the state after the op, i.e. state 8, and the state on its output
>     value %add.
>
> These are all the reads and writes:
>
> ```
> // READ:  state[7]   == {a: Const{1}}
> // READ:  state[%a]  == Const{1}
> // READ:  state[%2]  == Const{2}
> %add = jsir.binary_expression{"+"} (%a, %2)
> // WRITE: state[8]    = {a: Const{1}}
> // WRITE: state[%add] = Const{3}
> ```

> **Example 1.2:** When we visit the `jsir.assignment_expression` op, we will:
>
>   * Read the state before the op, i.e. state 8, and the states on its input
>     values %a_ref and %add; and
>   * Write the state after the op, i.e. state 9, and the state on its output
>     value %b_assign.
>
> These are all the reads and writes:
>
> ```
> // READ: state[8]         == {a: Const{1}}
> // READ: state[%b_ref]    == Uninit
> // READ: state[%add]      == Const{3}
> %b_assign = jsir.assignment_expression (%b_ref, %add)
> // WRITE: state[9]         = {a: Const{1}, b: Const{3}}
> // WRITE: state[%b_assign] = Const{3}
> ```

> **Caveat:** Currently, we don’t store states on lvalues like `%b_ref`.
> Instead, when we visit the `jsir.assignment_expression` op, we fetch the
> defining op of `%b_ref` (i.e. the `jsir.identifier_ref` op) to get the
> variable name. This is because an lvalue represents a reference to an object,
> rather than a value, so it doesn't fit our definition of value states in
> constant propagation. This should be changed.

When we have completed the calculation of every state in the IR, the algorithm
terminates.

## Example 2: If statement

Through this example, we will demonstrate:

  * Control flow graphs (CFGs) - how control flow structures are represented in
    JSIR

  * Work queue - how CFG traversal works

Consider the following `if`-statement:

```js
...
// We don't know what cond is.
if (cond)
  a = 1;
else
  a = 2;
// a == Unknown
...
```

Imagine that we don’t know the value of `cond`, so we don’t know which branch we
will take. Therefore, we don't know what `a` will be after the `if`-statement.
The constant propagation analysis must properly infer this.

JSIR comes with two dialects, JSHIR and JSLIR. Dataflow analysis runs on JSLIR,
which looks like this:

```
          ...
          %cond = jsir.identifier{"cond"}
   +----- cf.cond_br (%cond) [^bb_true, ^bb_false]
   |
   +--> ^bb_true:
   |      %a_ref_true = jsir.identifier_ref{"a"}
   |      %1 = jsir.numeric_literal{1}
   |      %assign_true = jsir.assignment_expression (%a_ref_true, %1)
   |      jsir.expression_statement (%assign_true)
+--|----- cf.br [^bb_end]
|  |
|  +--> ^bb_false:
|         %a_ref_false = jsir.identifier_ref{"a"}
|         %2 = jsir.numeric_literal{2}
|         %assign_false = jsir.assignment_expression (%a_ref_false, %2)
|         jsir.expression_statement (%assign_false)
+-------- cf.br [^bb_end]
|
+-----> ^bb_end:
          ...
```

In JSLIR, we represent control flow structures using control flow graphs (CFGs).
We can see that the two branches of the `if`-statement are represented by two
**blocks** `^bb_true` and `^bb_false`.

  * We enter the `if`-statement with a `cf.cond_br`
    ("control_flow.conditional_branch") op, which says, if `%cond` is `true`,
    go to `^bb_true`, else go to `^bb_false`.

  * At the end of both `^bb_true` and `^bb_false`, we have a `cf.br`
    ("control_flow.branch") op that unconditionally jumps to `^bb_end`, which
    is the start of the code after the `if`-statement.

### Step 1: Up until the start of the `if`-statement

The dataflow analysis algorithm starts at the top, and we compute states like
before, until we see `cf.cond_br`.

<pre><code>          ...
          <b>// state B0 = {}</b>
          %cond = jsir.identifier{"cond"}
          <b>// state[%cond] = Unknown</b>
          <b>// state B1 = {}</b>
   +----- cf.cond_br (%cond) [^bb_true, ^bb_false]
   |
   +--> ^bb_true:
   |      &lt;IR for `a = 1;`&gt;
+--|----- cf.br [^bb_end]
|  |
|  +--> ^bb_false:
|         &lt;IR for `a = 2;`&gt;
+-------- cf.br [^bb_end]
|
+-----> ^bb_end:
          ...
</code></pre>

### Step 2: Propagating the state to both branches

The `cf.cond_br` op has two successors, so handling that is more complicated:

  * First, we still compute the state after `cf.cond_br` (i.e. `state B2`),
    which is equal to `state B1`.

  * Then, we propagate `state B2` to both `^bb_true` (`state T0`) and
    `^bb_false` (`state F0`).

  * We maintain a **work queue** such that we make sure to traverse ops in both
    `^bb_true` and `^bb_false`.

<pre><code>          ...
          // state B0 = {}
          %cond = jsir.identifier{"cond"}
          // state[%cond] = Unknown
          // state B1 = {}
          cf.cond_br (%cond) [^bb_true, ^bb_false]
   +----- <b>// state B2 = {}</b>
   |
   |    ^bb_true:
   +----> <b>// state T0 = {}</b>
   |      &lt;IR for `a = 1;`&gt;
+--|----- cf.br [^bb_end]
|  |
|  |    ^bb_false:
|  +----> <b>// state F0 = {}</b>
|         &lt;IR for `a = 2;`&gt;
+-------- cf.br [^bb_end]
|
+-----> ^bb_end:
          ...
</code></pre>

### (WIP) Concept: the work queue

### Step 3: Computing states in `^bb_true`

Now, assume that we will first traverse every op in `^bb_true`. Note that it’s
not how the work queue works in practice, but the order of traversal actually
doesn't matter in the algorithm, so the assumption here is only for
demonstration purposes.

<!-- disableFinding(HTML_OPEN) -->

<pre><code>          ...
          // state B0 = {}
          %cond = jsir.identifier{"cond"}
          // state[%cond] = Unknown
          // state B1 = {}
          cf.cond_br (%cond) [^bb_true, ^bb_false]
   +----- // state B2 = {}
   |
   |    ^bb_true:
   +----> // state T0 = {}
   |      %a_ref_true = jsir.identifier_ref{"a"}
   |      <b>// state[%a_ref_true] = Uninit</b>
   |      <b>// state T1 = {}</b>
   |      %1 = jsir.numeric_literal{1}
   |      <b>// state[%1] = Const{1}</b>
   |      <b>// state T2 = {}</b>
   |      %assign_true = jsir.assignment_expression (%a_ref_true, %1)
   |      <b>// state[%assign_true] = Const{1}</b>
   |      <b>// state T3 = {a: Const{1}}</b>
   |      jsir.expression_statement (%assign_true)
   |      <b>// state T4 = {a: Const{1}}</b>
+--|----- cf.br [^bb_end]
|  |
|  |    ^bb_false:
|  +----> // state F0 = {}
|         &lt;IR for `a = 2;`&gt;
+-------- cf.br [^bb_end]
|
+-----> ^bb_end:
          ...
</code></pre>

### Step 4: Propagating from `^bb_true` to `^bb_end`

Now we have reached the `cf.br` op in `^bb_true`, and we will:

  * Compute the state after the op, namely `state T5`;
  * Propagate `state T5` to the beginning of `^bb_end`, which is the successor
    of the `cf.br` op.

<!-- end list -->

<pre><code>          ...
          cf.cond_br (%cond) [^bb_true, ^bb_false]
   +----- // state B2 = {}
   |
   |    ^bb_true:
   +----> // state T0 = {}
   |      &lt;IR for `a = 1;`&gt;
   |      // state T4 = {a: Const{1}}
   |      cf.br [^bb_end]
+--|----- <b>// state T5 = {a: Const{1}}</b>
|  |
|  |    ^bb_false:
|  +----> // state F0 = {}
|         &lt;IR for `a = 2;`&gt;
+-------- cf.br [^bb_end]
|
|       ^bb_end:
+-------> <b>// state E0 = {a: Const{1}}</b>
          ...
</code></pre>

> \[Important\] Since we don’t know which branch in the if-statement we will
> take, we shouldn’t know what `a` is after the if-statement. However, right now
> `state E0` says `a` has constant `1`, which is incorrect. This is fine - this
> is an intermediate state in the dataflow analysis algorithm.

### Step 5: Computing states in `^bb_false`

Now, imagine that from the work queue, we decide to start traversing
`^bb_false`. We will compute states up to `state F4`:

<pre><code>          ...
          cf.cond_br (%cond) [^bb_true, ^bb_false]
   +----- // state B2 = {}
   |
   |    ^bb_true:
   +----> // state T0 = {}
   |      &lt;IR for `a = 1;`&gt;
   |      // state T4 = {a: Const{1}}
   |      cf.br [^bb_end]
+--|----- // state T5 = {a: Const{1}}
|  |
|  |    ^bb_false:
|  +----> // state F0 = {}
|         %a_ref_false = jsir.identifier_ref{"a"}
|         <b>// state[%a_ref_false] = Uninit</b>
|         <b>// state F1 = {}</b>
|         %2 = jsir.numeric_literal{2}
|         <b>// state[%2] = Const{2}</b>
|         <b>// state F2 = {}</b>
|         %assign_false = jsir.assignment_expression (%a_ref_false, %2)
|         <b>// state[%assign_false] = Const{2}</b>
|         <b>// state F3 = {a: Const{2}}</b>
|         jsir.expression_statement (%assign_false)
|         <b>// state F4 = {a: Const{2}}</b>
+-------- cf.br [^bb_end]
|
|       ^bb_end:
+-------> // state E0 = {a: Const{1}}
          ...
</code></pre>

### Step 6: Joining `^bb_false` into `^bb_end`

Now we have reached the `cf.br` op in `^bb_false`. Like before, we calculate the
state after the op (i.e. `state F5`). The important thing is we need to then
propagate `state F5` to `state E0`. We don't simply overwrite `state E0` with
`state F5`, but we **join `state F5` into `state E0`**:

<pre><code>          ...
          // state B1 = {}
          cf.cond_br (%cond) [^bb_true, ^bb_false]
   +----- // state B2 = {}
   |
   |    ^bb_true:
   +----> // state T0 = {}
   |      &lt;IR for `a = 1;`&gt;
   |      // state T4 = {a: Const{1}}
   |      cf.br [^bb_end]
+--|----- // state T5 = {a: Const{1}}
|  |
|  |    ^bb_false:
|  +----> // state F0 = {}
|         &lt;IR for `a = 2;`&gt;
|         // state F4 = {a: Const{2}}
|         cf.br [^bb_end]
+-------- <b>// state F5 = {a: Const{2}}</b>
|
|       ^bb_end:
+-------> <b>// state E0 = {a: <del>Const{1}</del> Unknown}</b>
          ...
</code></pre>

> ### Concepts: Lattice and Join
>
> The types of states, both on program points and on values, are mathematical
> “lattices”. The most important feature of a lattice is that you can join()
> two elements.
>
> For example, we already know intuitively that, for a state on a value:
>
> ```
> Join(Const{M}, Const{N}) = Unknown
> ```
>
> This means that if a %value might take two different constant values in two
> program paths, then when the program paths merge, we don’t know what value it
> takes, hence we have to say it’s Unknown.
>
> ```
> Join(Const{N}, Unknown ) = Unknown
> ```
>
> This means that if a %value takes constant literal in one path, but is Unknown
> in another path, then when the program paths merge, we have to be on the
> conservative side and say it’s Unknown.
>
> What’s less obvious is that, in addition to `Unknown` we also need another
> state value called `Uninit`, and that:
>
> ```
> Join(Uninit,   Const{N}) = Const{N}
> ```
>
> To explain this, let’s go back to the if-statement example. In \[step 4\] we
> first propagate state 8 into state 9, causing state 9 to become {a: Const{1}},
> and in \[step 6\] we propagate state 14 into state 9, causing state 9 to
> become {a: Unknown}.
>
> So what was state 9 before \[step 4\]? We need to pre-initialize state 9 with
> a value such that this value, when joined with {a: Const{1}}, yields {a:
> Const{1}}. This value is:
>
> ```
> {} default = Uninit
> ```
>
> which means: unless explicitly specified, all symbols have the value `Uninit`.
>
> ```
>          ...
>          // state B0 = [default = Unknown] {}
>          %cond = jsir.identifier{"cond"}
>          // state[%cond] = Unknown
>          // state B1 = [default = Unknown] {}
>          cf.cond_br (%cond) [^bb_true, ^bb_false]
>   +----- // state B2 = [default = Unknown] {}
>   |
>   |    ^bb_true:
>   +----> // state T0 = [default = Unknown] {}
>   |      %a_ref_true = jsir.identifier_ref{"a"}
>   |      // state[%a_ref_true] = Uninit
>   |      // state T1 = [default = Unknown] {}
>   |      %1 = jsir.numeric_literal{1}
>   |      // state[%1] = Const{1}
>   |      // state T2 = [default = Unknown] {}
>   |      %assign_true = jsir.assignment_expression (%a_ref_true, %1)
>   |      // state[%assign_true] = Const{1}
>   |      // state T3 = [default = Unknown] {a: Const{1}}
>   |      jsir.expression_statement (%assign_true)
>   |      // state T4 = [default = Unknown] {a: Const{1}}
> +--|----- cf.br [^bb_end]
> |  |      // state T5 = [default = Uninit] {}
> |  |
> |  |    ^bb_false:
> |  +----> // state F0 = [default = Unknown] {}
> |         %a_ref_false = jsir.identifier_ref{"a"}
> |         // state F1 = [default = Unknown] {}
> |         %2 = jsir.numeric_literal{2}
> |         // state F2 = [default = Unknown] {}
> |         %assign_false = jsir.assignment_expression (%a_ref_false, %2)
> |         jsir.expression_statement (%assign_false)
> +-------- cf.br [^bb_end]
> |
> +-----> ^bb_end:
>          ...
> ```

## Example 3: While loop

The control flow graph (CFG) of a `while`-statement contains a loop. The
dataflow analysis algorithm treats this CFG the same way, but in this example,
we demonstrate more explicitly:

  * How `Join` works in practice;

  * The importance of lattice design for the algorithm to terminate.

Consider the following code:

```js
a = 1;
while (cond()) {
  a = a + 2;
}
```

Let’s just assume that `cond()` returns a nondeterministic boolean (i.e. every
time it might return a different result). Therefore, we have to be conservative
and say we don't know what `a` is, both within and after the loop:

```js
// {}
a = 1;
// {a: Const{1}}
while (cond()) {
  // {a: Unknown}
  a = a + 2;
  // {a: Unknown}
}
// {a: Unknown}
```

Note that `a` is `Unknown` within the loop body because we are reasoning about
the combination of all iterations.

Now, we convert the code into the corresponding JSLIR.

```
          ...
          %a_ref_before = jsir.identifier_ref{"a"}
          %1 = jsir.numeric_literal{1}
          %assign_before = jsir.assignment_expression (%a_ref_before, %1)
          jsir.expression_statement (%assign_before)
+-------- cf.br [^bb_test]
|
+-----> ^bb_test:
|         // IR for `cond()`:
|         %cond = ...
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |
|  +--> ^bb_body:
|  |      // IR for `a = a + 2;`:
|  |      %a_ref_body = jsir.identifier_ref{"a"}
|  |      %a = jsir.identifier{"a"}
|  |      %2 = jsir.numeric_literal{2}
|  |      %add = jsir.binary_expression{"+"} (%a, %2)
|  |      %assign_body = jsir.assignment_expression (%a_ref_body, %add)
|  |      jsir.expression_statement (%assign_body)
+--|----- cf.br [^bb_test]
   |
   +--> ^bb_end:
          ...
```

### Step 1: Entering `bb_test` the first time

Similar to the handling of the `if`-statement, we compute the states up to the
start of the `while`-loop. Then, we propagate the state through the `cf.br` op
to `^bb_test`.

<pre><code>          ...
          <b>// state B0 = {}</b>
          &lt;IR for `a = 1;`&gt;
          <b>// state B4 = {a: Const{1}}</b>
          cf.br [^bb_test]
+-------- <b>// state B5 = {a: Const{1}}</b>
|
|       ^bb_test:
+-------> <b>// state T0 = {a: Const{1}}</b>
|         %cond = ...
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |
|  +--> ^bb_body:
|  |      &lt;&lt;&lt;IR for `a = a + 2;`&gt;&gt;&gt;
+--|----- cf.br [^bb_test]
   |
   +--> ^bb_end:
          ...
</code></pre>

We can see that, as we enter `^bb_test` for the first time, `a` holds the value
of `1`. Note that this does not match the expected final result (`Unknown`). We
will see how `a` eventually becomes `Unknown` as we progress through the
algorithm.

### Step 2: Running the loop test the first time

Now we will compute the states in `^bb_test`, which resembles evaluating the
loop condition for the first time. As stated in the assumption before, we don't
know the return value of `cond()`, so we can only assign `Unknown` to `%cond`.

<pre><code>          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> // state T0 = {a: Const{1}}
|         %cond = ...
|         <b>// NOTE: Computing the states in `^bb_test`.</b>
|         <b>// state[%cond] = Unknown</b>
|         <b>// state T1 = {a: Const{1}}</b>
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |
|  +--> ^bb_body:
|  |      &lt;&lt;&lt;IR for `a = a + 2;`&gt;&gt;&gt;
+--|----- cf.br [^bb_test]
   |
   +--> ^bb_end:
          ...
</code></pre>

Since `%cond` is `Unknown`, we have to conservatively assume that both branch
targets are possible, so we propagate `state T2` to both `state I0` (the loop
body) and `state E0` (after the loop).

<pre><code>          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> // state T0 = {a: Const{1}}
|         %cond = ...
|         // state[%cond] = Unknown
|         // state T1 = {a: Const{1}}
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |      <b>// NOTE: Since %cond is Unknown, we propagate to both states 9 and 10.</b>
|  |      <b>// state T2 = {a: Const{1}}</b>
|  |
|  |    ^bb_body:
|  +-->   <b>// state I0 = {a: Const{1}}</b>
|  |      &lt;&lt;&lt;IR for `a = a + 2;`&gt;&gt;&gt;
+--|----- cf.br [^bb_test]
   |
   +--> ^bb_end:
          <b>// state E0 = {a: Const{1}}</b>
          ...
</code></pre>

### Step 3: Running the loop body

Now, depending on the status of the work queue, the algorithm might decide to
visit ops in `^bb_body` or `^bb_end`, or both interleaved. Here, for simplicity
reasons, we assume that we will visit `^bb_body`. This represents the states
during the first iteration of the loop body, which changes `a` from `1` to `3`.

<pre><code>          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> // state T0 = {a: Const{1}}
|         %cond = ...
|         // state[%cond] = Unknown
|         // state T1 = {a: Const{1}}
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |      // state T2 = {a: Const{1}}
|  |
|  +--> ^bb_body:
|  |      // state I0 = {a: Const{1}}
|  |      &lt;&lt;&lt;IR for `a = a + 2;`&gt;&gt;&gt;
|  |      <b>// state I1 = {a: Const{3}}</b>
+--|----- cf.br [^bb_test]
   |
   +--> ^bb_end:
          // state E0 = {a: Const{1}}
          ...
</code></pre>

At the end of the loop body, we jump back to `^bb_test`, which `Join`s `state
I2` into `state T0`.

<pre><code>          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> <b>// state T0 = {a: <del>Const{1}</del> Unknown}</b>
|         %cond = ...
|         // state[%cond] = Unknown
|         // state T1 = {a: Const{1}}
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |      // state T2 = {a: Const{1}}
|  |
|  +--> ^bb_body:
|  |      // state I0 = {a: Const{1}}
|  |      &lt;&lt;&lt;IR for `a = a + 2;`&gt;&gt;&gt;
|  |      // state I1 = {a: Const{3}}
|  |      cf.br [^bb_test]
+--|----- <b>// state I2 = {a: Const{3}}</b>
   |
   +--> ^bb_end:
          // state E0 = {a: Const{1}}
          ...
</code></pre>

### Step 4: Running the loop test again

Going through ^bb_test again, and propagating the state to ^bb_body and
^bb_end.

<pre><code>          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> // state T0 = {a: <del>Const{1}</del> Unknown}
|         %cond = ...
|         <b>// state[%cond] = Unknown</b>
|         <b>// state T1 = {a: <del>Const{1}</del> Unknown}</b>
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |      <b>// state T2 = {a: <del>Const{1}</del> Unknown}</b>
|  |
|  +--> ^bb_body:
|  |      <b>// state I0 = {a: <del>Const{1}</del> Unknown}</b>
|  |      &lt;IR for `a = a + 2;`&gt;
|  |      // state I1 = {a: Const{3}}
|  |      cf.br [^bb_test]
+--|----- // state I2 = {a: Const{3}}
   |
   +--> ^bb_end:
          <b>// state E0 = {a: <del>Const{1}</del> Unknown}</b>
          ...
</code></pre>

### Step 5: Running the loop body again

<pre><code>          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> // state T0 = {a: <del>Const{1}</del> Unknown}
|         %cond = ...
|         // state[%cond] = Unknown
|         // state T1 = {a: <del>Const{1}</del> Unknown}
|  +----- cf.cond_br (%cond) [^bb_body, ^bb_end]
|  |      // state T2 = {a: <del>Const{1}</del> Unknown}
|  |
|  +--> ^bb_body:
|  |      // state I0 = {a: <del>Const{1}</del> Unknown}
|  |      &lt;IR for `a = a + 2;`&gt;
|  |      <b>// state I1 = {a: <del>Const{3}</del> Unknown}</b>
|  |      cf.br [^bb_test]
+--|----- // state I2 = {a: Const{3}}
   |
   +--> ^bb_end:
          // state E0 = {a: <del>Const{1}</del> Unknown}
          ...
</code></pre>

Propagate state 16 to state 6, fixpoint reached, algorithm terminates

<pre><code class="language-js prettyprint">          ...
          // state B0 = {}
          &lt;IR for `a = 1;`&gt;
          // state B4 = {a: Const{1}}
          cf.br [^bb_test]
+-------- // state B5 = {a: Const{1}}
|
|       ^bb_test:
+-------> <b>// state T0 = {a: <del>Const{1} Unknown</del> Unknown}</b>
|         %cond = ...
|         // state[%cond] = Unknown
|         // state T1 = {a: Const{1} Unknown}
|         cf.cond_br (%cond) [^bb_body, ^bb_end]
|  +----- // state T2 = {a: Const{1} Unknown}
|  |
|  |    ^bb_body:
|  +----> // state I0 = {a: Const{1} Unknown}
|  |      &lt;IR for `a = a + 2;`&gt;
|  |      // state 11 = {a: Const{3} Unknown}
|  |      cf.br [^bb_test]
+--|----- <b>// state I2 = {a: <del>Const{3}</del> Unknown}</b>
   |
   |    ^bb_end:
   +----> // state E0 = {a: <del>Const{1}</del> Unknown}
          ...
</code></pre>

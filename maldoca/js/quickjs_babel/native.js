/**
 * @license
 * Copyright 2024 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * @fileoverview The top-level file to run in V8.
 */

exports = {};

// =============================================================================

// NOTE: There are two encodings:
// +---------+--------------------+---------------------------+
// | node.js |      'base64'      |        'base64url'        |
// |---------+--------------------+---------------------------+
// |   C++   | absl::Base64Escape | absl::WebSafeBase64Escape |
// +---------+--------------------+---------------------------+
/**
 * Base64-encodes a string.
 *
 * @param {string} value
 * @return {string}
 */
function base64Encode(value) {
  return atob(value);
}

/**
 * Base64-decodes a string.
 *
 * @param {string} value
 * @return {string}
 */
function base64Decode(value) {
  return btoa(value);
}

/**
 * A StringLiteral looks like this:
 *
 * {
 *   type: 'StringLiteral',
 *   value: 'a',
 *   extra: {
 *     rawValue: 'a',
 *     raw: '"\u0061"',
 *   },
 * }
 *
 * We want to base64-encode/decode node.value and node.extra.rawValue.
 *
 * @param {!Object} node
 * @param {!Set<!Object>} visited
 * @param {function(string): string} mutate
 */
function mutateStringLiteral(node, visited, mutate) {
  if ('value' in node && typeof (node.value) === 'string') {
    node.value = mutate(node.value);
  }

  if ('extra' in node && typeof (node.extra) === 'object') {
    const extra = node.extra;

    // Sometimes the same object appears multiple times in the AST, probably to
    // reduce memory usage.
    if (visited.has(extra)) {
      return;
    }
    visited.add(extra);

    if ('rawValue' in extra && typeof (extra.rawValue) === 'string') {
      extra.rawValue = mutate(extra.rawValue);
    }
  }
}

/**
 * Base64-encode/decode all StringLiterals in the AST.
 *
 * @param {!Object} node
 * @param {!Set<!Object>} visited
 * @param {function(string): string} mutate
 */
function mutateStringLiterals(node, visited, mutate) {
  if (!(node instanceof Object)) {
    return;
  }

  // Sometimes the same object appears multiple times in the AST, probably to
  // reduce memory usage.
  if (visited.has(node)) {
    return;
  }
  visited.add(node);

  if ('type' in node && node.type === 'StringLiteral') {
    mutateStringLiteral(node, visited, mutate);
  }

  // Recursively walk the AST.
  // Note that this for-loop also works if node is an array.
  for (const element of Object.values(node)) {
    mutateStringLiterals(element, visited, mutate);
  }
}

/**
 * Base64-encode all StringLiterals in the AST.
 *
 * @param {!Object} node
 * @param {!Set<!Object>=} visited
 */
function base64EncodeStringLiterals(node, visited = new Set()) {
  mutateStringLiterals(node, visited, base64Encode);
}

/**
 * Base64-decode all StringLiterals in the AST.
 *
 * @param {!Object} node
 * @param {!Set<!Object>=} visited
 */
function base64DecodeStringLiterals(node, visited = new Set()) {
  mutateStringLiterals(node, visited, base64Decode);
}

// =============================================================================

/**
 * Replaces characteres in the range [U+D800, U+DFFF] with '�' (U+FFFD).
 *
 // copybara:strip_begin(internal comment)
 * See: b/235090893.
 // copybara:strip_end
 *
 * @param {string} source
 * @return {string}
 */
function replaceInvalidSurrogatePairs(source) {
  return [...source]
      .map(
          (str) => (str.codePointAt(0) ?? 0) >= 0xD800 &&
                  (str.codePointAt(0) ?? 0) <= 0xDFFF ?
              '\ufffd' :
              str)
      .join('');
}

/**
 * Parses JavaScript source and returns a stringified AST.
 * @param {string} source
 * @param {object?} options
 * @return {{ast: string, scopes: !Array<!Object>}}
 */
function parseInternal(source, options) {
  const ast = Babel.packages.parser.parse(source, options);

  if (options?.replaceInvalidSurrogatePairs) {
    Babel.packages.traverse.default(ast, {
      StringLiteral: (path) => {
        const node = path.node;
        if ('extra' in node) {
          const extra = node.extra;
          if ('rawValue' in extra) {
            extra.rawValue = replaceInvalidSurrogatePairs(extra.rawValue);
          }
        }

        if ('value' in node) {
          node.value = replaceInvalidSurrogatePairs(node.value);
        }
      }
    });
  }

  if (options?.base64EncodeStringLiterals) {
    base64EncodeStringLiterals(ast);
  }

  // Store all scopes in a dictionary, and add a scope UID to each AST node.
  //
  // We don't try to get scope information when there are errors in the AST
  // (this only happens when errorRecovery is true), because (1) scope
  // information would be invalid anyway, and (2) babel-traverse would crash
  // with an exception during scope computation.
  const scopes = {};
  if (options?.computeScopes && !('errors' in ast && ast.errors.length > 0)) {
    Babel.packages.traverse.default(ast, {
      enter(path) {
        const scope = path.scope;
        if ('uid' in scope && typeof scope.uid === 'number') {
          scopes[scope.uid] = scope;
          path.node.scopeUid = scope.uid;
        }
      }
    });

    for (const scope of Object.values(scopes)) {
      for (const [name, binding] of Object.entries(scope.bindings)) {
        for (const referencePath of binding.referencePaths) {
          referencePath.node.referencedSymbol = {
            name: name,
            defScopeUid: scope.uid,
          };
        }

        const def_node = binding.path.node;
        if (def_node.definedSymbols === undefined) {
          def_node.definedSymbols = [];
        }
        def_node.definedSymbols.push({
          name,
          defScopeUid: scope.uid,
        });
      }
    }
  }

  // We don't serialize to JSON even though it's possible. The reason is that
  // the AST of TSCompiler (the other choice) cannot be directly serialized due
  // to the existence of parent pointers. Therefore, it would not be a fair
  // comparison if we serialize here for Babel.
  return {ast: JSON.stringify(ast), scopes: Object.values(scopes)};
}
exports.parseInternal = parseInternal;

// =============================================================================

/**
 * Generates JavaScript code from an AST.
 * @param {object!} ast
 * @param {object!} options
 * @return {string}
 */
function generateInternal(ast, options) {
  if (options.base64DecodeStringLiterals) {
    base64DecodeStringLiterals(ast);
  }

  const {code} = Babel.packages.generator.default(ast, options);
  return code;
}
exports.generateInternal = generateInternal;

/**
 * @param {!Object} error
 * @return {?{line: number, column: number}}
 */
function maybeGetPosition(error) {
  if (!(error?.loc instanceof Object)) {
    return null;
  }

  if (typeof error?.loc?.line != 'number' ||
      typeof error?.loc?.column != 'number') {
    return null;
  }

  return {line: error.loc.line, column: error.loc.column};
}

/**
 * @param {!Object} error
 * @return {{
 *    name: string,
 *    message: string,
 *    loc: ?{line: number, column: number}
 * }}
 */
function unknownToBabelError(error) {
  return {
    name: error?.name || '{error}',
    message: error?.message || '',
    loc: typeof error === 'object' ? maybeGetPosition(error) : null,
  };
}

/**
 * Parses JavaScript source into an AST.
 * @param {string} sourceCode
 * @param {string?} optionsSerialized
 * @return {{ast: string, response: string}} A JSON-serialized AST and a
 *     JSON-serialized `BabelParseResponse`.
 */
exports.parse = function(sourceCode, optionsSerialized) {
  let options = undefined;
  if (optionsSerialized) {
    options = JSON.parse(optionsSerialized);
  }

  try {
    let {ast, scopes} = parseInternal(sourceCode, options);

    const scopesPb = {
      scopes: {},
    };

    for (const scope of scopes) {
      if (scope === null) continue;
      const uid = scope.uid;

      const scopePb = {
        uid: uid,
        bindings: {},
      };

      if (scope.parent) {
        scopePb.parentUid = scope.parent.uid;
      }

      for (const [name, binding] of Object.entries(scope.bindings)) {
        const bindingKindPb = (() => {
          switch (binding.kind) {
            case 'var':
              return 'KIND_VAR';
            case 'let':
              return 'KIND_LET';
            case 'const':
              return 'KIND_CONST';
            case 'module':
              return 'KIND_MODULE';
            case 'hoisted':
              return 'KIND_HOISTED';
            case 'param':
              return 'KIND_PARAM';
            case 'local':
              return 'KIND_LOCAL';
            case 'unknown':
              return 'KIND_UNKNOWN';
            default:
              return 'KIND_UNKNOWN';
          }
        })();

        const bindingPb = {
          kind: bindingKindPb,
          name: name,
        };

        scopePb.bindings[name] = bindingPb;
      }

      scopesPb.scopes[uid] = scopePb;
    }

    const response = {
      errors: [],
      scopes: scopesPb,
    };

    return {
      ast: ast,
      response: JSON.stringify(response),
    };

  } catch (error) {
    const response = {};

    const babelError = unknownToBabelError(error);
    if (babelError) {
      response.errors = [babelError];
    }

    return {ast: '', response: JSON.stringify(response)};
  }
};

/**
 * Generates JavaScript code from an AST.
 * @param {string} astString
 * @param {string?} optionsSerialized
 * @return {{source: string, response: string}} The generated code and a
 *     JSON-serialized `BabelGenerateResponse`.
 */
exports.generate = function(astString, optionsSerialized) {
  try {
    let ast = JSON.parse(astString);

    let options = {};
    if (optionsSerialized) {
      options = JSON.parse(optionsSerialized);
    }

    const source = generateInternal(ast, options);
    return {source, response: '{}'};

  } catch (error) {
    const response = {};

    const errorPb = unknownToBabelError(error);
    if (errorPb) {
      response.error = errorPb;
    }

    return {source: '', response: JSON.stringify(response)};
  }
};

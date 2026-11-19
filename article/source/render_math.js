/* Typeset article mathematics at build time, retaining accessible MathML. */
'use strict';
const fs = require('fs');
const katex = require('../../assets/vendor/katex/katex.min.js');

function renderMath(source) {
  return source.replace(/\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g, (_, inline, display) =>
    katex.renderToString(inline === undefined ? display : inline, {
      displayMode: display !== undefined,
      throwOnError: true,
      strict: 'error',
      trust: false,
      output: 'htmlAndMathml'
    }));
}

module.exports = {renderMath};
if (require.main === module) process.stdout.write(renderMath(fs.readFileSync(0, 'utf8')));

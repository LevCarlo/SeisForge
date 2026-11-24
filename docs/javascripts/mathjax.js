// MathJax for pymdownx.arithmatex (generic mode): $...$ inline, $$...$$ display.
// Every display equation is numbered (tags: "all"); \label{} + \eqref{} link
// to it within the page.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "all",
  },
  chtml: {
    // Match the x-height of the body font so inline maths sits with the text.
    matchFontHeight: true,
    mtextInheritFont: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Re-typeset after instant page loads (Material and Zensical both expose document$).
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});

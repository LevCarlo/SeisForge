// MathJax for pymdownx.arithmatex (generic mode): $...$ inline, $$...$$ display.
//
// Markdown pages: arithmatex turns $...$ into \(...\) inside elements of
// class "arithmatex", and only those elements are typeset.
// Notebook pages (mkdocs-jupyter): the converted notebook keeps raw $...$ in
// its Markdown cells (class jp-RenderedMarkdown) and in SymPy's LaTeX output
// (class jp-RenderedLatex). The MathJax-2 configuration that mkdocs-jupyter
// injects does nothing under MathJax 3, so those classes are typeset here,
// with $ delimiters. Everything else stays ignored, so a literal dollar sign
// in ordinary prose is never treated as math. The ignore pattern names the
// top-level Material containers (not ".*|"): an element without a class then
// inherits the state of its nearest classed ancestor, so the <p> elements
// inside a notebook Markdown cell are typeset while page prose is not.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: "md-header|md-container|md-dialog|md-banner",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown|jp-RenderedLatex",
  },
};

// Re-typeset after Material's instant page loads.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});

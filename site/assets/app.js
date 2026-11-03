// Theme toggle: OS preference by default, explicit choice remembered per browser.
(function () {
  var root = document.documentElement, btn = document.getElementById("theme");
  if (!btn) return;
  function current() {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  btn.addEventListener("click", function () {
    var next = current() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("llmf-theme", next); } catch (_) {}
  });
  // Open the transcript/details targeted by a #hash link.
  if (location.hash) {
    var el = document.querySelector(location.hash);
    if (el && el.tagName === "DETAILS") el.open = true;
  }
})();

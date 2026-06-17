document.addEventListener("DOMContentLoaded", () => {
  document
    .querySelectorAll('a[href="https://github.com/PlatinumCD/sculptor-mlir"]')
    .forEach((link) => {
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
    });
});

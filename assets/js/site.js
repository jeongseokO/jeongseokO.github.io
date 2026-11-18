// Theme toggle and BibTeX copy. The page works without this file.
(function () {
  var root = document.documentElement;

  var toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    var media = window.matchMedia('(prefers-color-scheme: dark)');
    var current = function () {
      return root.getAttribute('data-theme') || (media.matches ? 'dark' : 'light');
    };
    var sync = function () { toggle.setAttribute('aria-pressed', current() === 'dark' ? 'true' : 'false'); };
    toggle.hidden = false;
    sync();
    toggle.addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* storage unavailable: keep for this page only */ }
      sync();
    });
    media.addEventListener && media.addEventListener('change', sync);
  }

  if (navigator.clipboard) {
    document.querySelectorAll('.copy-btn').forEach(function (btn) {
      var label = btn.textContent;
      btn.hidden = false;
      btn.addEventListener('click', function () {
        var code = btn.parentNode.querySelector('pre');
        navigator.clipboard.writeText(code.textContent).then(function () {
          btn.textContent = btn.getAttribute('data-copied');
          setTimeout(function () { btn.textContent = label; }, 1600);
        });
      });
    });
  }
})();

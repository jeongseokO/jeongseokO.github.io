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

  // Email links copy the address instead of opening a mail app. Without the
  // Clipboard API (or if copying fails) the mailto: link works as before.
  if (navigator.clipboard) {
    var status = document.getElementById('copy-status');
    document.querySelectorAll('.js-copy-email').forEach(function (link) {
      var label = link.textContent;
      var timer;
      link.addEventListener('click', function (event) {
        event.preventDefault();
        navigator.clipboard.writeText(link.getAttribute('data-email')).then(function () {
          link.textContent = link.getAttribute('data-copied');
          link.classList.add('is-copied');
          if (status) status.textContent = status.getAttribute('data-message');
          clearTimeout(timer);
          timer = setTimeout(function () {
            link.textContent = label;
            link.classList.remove('is-copied');
            if (status) status.textContent = '';
          }, 1600);
        }, function () { window.location.href = link.href; });
      });
    });
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

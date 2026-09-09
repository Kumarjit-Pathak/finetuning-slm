/* ============================================================
   SLM Fine-Tuning Tutorial — Core Interactivity
   Loaded on every page. No emoji, no mascot.
   ============================================================ */

(function () {
  'use strict';

  // ------------------------------------------------------------
  // 1. Dark mode (persist via localStorage; respect OS default)
  // ------------------------------------------------------------
  const THEME_KEY = 'slm-theme';

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    updateThemeToggleLabel(theme);
  }

  function updateThemeToggleLabel(theme) {
    document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
      btn.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
      btn.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
    });
  }

  window.slmToggleTheme = function () {
    applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  };

  function initTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) {
      applyTheme(saved);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      applyTheme('dark');
    } else {
      applyTheme('light');
    }
  }

  // ------------------------------------------------------------
  // 2. Tabs (technique picker, use-case picker, etc.)
  //    <div class="tabs" role="tablist"> with buttons + panels
  //    identified by data-tab and data-panel attributes.
  // ------------------------------------------------------------
  function initTabs(root) {
    (root || document).querySelectorAll('[data-tab-group]').forEach(group => {
      const buttons = group.querySelectorAll('[data-tab]');
      const panels = group.querySelectorAll('[data-panel]');

      buttons.forEach(btn => {
        btn.addEventListener('click', () => {
          const target = btn.getAttribute('data-tab');
          buttons.forEach(b => {
            const active = b.getAttribute('data-tab') === target;
            b.classList.toggle('active', active);
            b.setAttribute('aria-selected', active ? 'true' : 'false');
          });
          panels.forEach(p => {
            p.classList.toggle('active', p.getAttribute('data-panel') === target);
          });
          // Deep-link support
          if (btn.dataset.updateHash === 'true') {
            history.replaceState(null, '', '#' + target);
          }
        });
      });
    });
  }

  function openTabFromHash() {
    const hash = window.location.hash.replace('#', '');
    if (!hash) return;
    const btn = document.querySelector('[data-tab="' + hash + '"]');
    if (btn) btn.click();
  }

  // ------------------------------------------------------------
  // 3. Copy-to-clipboard for <pre> code blocks with .code-toolbar
  // ------------------------------------------------------------
  function initCopyButtons() {
    document.querySelectorAll('.code-toolbar').forEach(wrap => {
      if (wrap.querySelector('.copy-btn')) return;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {
        const codeEl = wrap.querySelector('pre code, pre');
        if (!codeEl) return;
        try {
          await navigator.clipboard.writeText(codeEl.textContent.trim());
          btn.textContent = 'Copied';
          btn.classList.add('copied');
          setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1200);
        } catch (_) {
          btn.textContent = 'Press Cmd+C';
        }
      });
      wrap.appendChild(btn);
    });
  }

  // ------------------------------------------------------------
  // 4. Sortable tables (click <th data-sort="number|text">)
  // ------------------------------------------------------------
  function initSortableTables() {
    document.querySelectorAll('table[data-sortable]').forEach(table => {
      const ths = table.querySelectorAll('thead th');
      ths.forEach((th, idx) => {
        if (!th.dataset.sort) return;
        th.style.cursor = 'pointer';
        let asc = true;
        th.addEventListener('click', () => {
          const tbody = table.querySelector('tbody');
          const rows = Array.from(tbody.querySelectorAll('tr'));
          const kind = th.dataset.sort;
          rows.sort((a, b) => {
            const av = (a.cells[idx].dataset.value ?? a.cells[idx].textContent).trim();
            const bv = (b.cells[idx].dataset.value ?? b.cells[idx].textContent).trim();
            if (kind === 'number') {
              return asc ? parseFloat(av) - parseFloat(bv) : parseFloat(bv) - parseFloat(av);
            }
            return asc ? av.localeCompare(bv) : bv.localeCompare(av);
          });
          rows.forEach(r => tbody.appendChild(r));
          ths.forEach(t => t.removeAttribute('aria-sort'));
          th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
          asc = !asc;
        });
      });
    });
  }

  // ------------------------------------------------------------
  // 5. Progress tracker (mark pages as read)
  //    Stored as a Set in localStorage under 'slm-progress'.
  // ------------------------------------------------------------
  const PROGRESS_KEY = 'slm-progress';

  function getProgress() {
    try { return new Set(JSON.parse(localStorage.getItem(PROGRESS_KEY) || '[]')); }
    catch (_) { return new Set(); }
  }

  function saveProgress(set) {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(Array.from(set)));
  }

  window.slmMarkRead = function (pageId) {
    const set = getProgress();
    set.add(pageId);
    saveProgress(set);
    updateProgressBadges();
  };

  window.slmToggleRead = function (pageId) {
    const set = getProgress();
    if (set.has(pageId)) set.delete(pageId);
    else set.add(pageId);
    saveProgress(set);
    updateProgressBadges();
  };

  function updateProgressBadges() {
    const set = getProgress();
    document.querySelectorAll('[data-progress-item]').forEach(el => {
      const id = el.getAttribute('data-progress-item');
      el.classList.toggle('is-read', set.has(id));
      const checkbox = el.querySelector('input[type="checkbox"]');
      if (checkbox) checkbox.checked = set.has(id);
    });
    document.querySelectorAll('[data-progress-count]').forEach(el => {
      const total = parseInt(el.getAttribute('data-progress-count'), 10) || 20;
      el.textContent = set.size + ' / ' + total;
    });
    document.querySelectorAll('[data-progress-bar]').forEach(el => {
      const total = parseInt(el.getAttribute('data-progress-bar'), 10) || 20;
      const pct = Math.min(100, Math.round((set.size / total) * 100));
      el.style.width = pct + '%';
    });
  }

  // ------------------------------------------------------------
  // 6. Highlight active nav link based on filename
  // ------------------------------------------------------------
  function highlightActiveNav() {
    const path = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-link, .side-nav a').forEach(a => {
      const href = a.getAttribute('href');
      if (!href) return;
      if (href === path || (path === '' && href === 'index.html')) {
        a.classList.add('active');
      }
    });
  }

  // ------------------------------------------------------------
  // 7. Mermaid init (only if present)
  // ------------------------------------------------------------
  function initMermaid() {
    if (typeof window.mermaid === 'undefined') return;
    const dark = currentTheme() === 'dark';
    window.mermaid.initialize({
      startOnLoad: true,
      theme: dark ? 'dark' : 'default',
      fontFamily: 'Inter, system-ui, sans-serif',
      securityLevel: 'loose'
    });
  }

  // ------------------------------------------------------------
  // 8. KaTeX auto-render (only if loaded)
  // ------------------------------------------------------------
  function initKatex() {
    if (typeof window.renderMathInElement !== 'function') return;
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true  },
        { left: '$',  right: '$',  display: false }
      ],
      throwOnError: false
    });
  }

  // ------------------------------------------------------------
  // Bootstrapping
  // ------------------------------------------------------------
  initTheme();

  document.addEventListener('DOMContentLoaded', () => {
    highlightActiveNav();
    initTabs();
    initCopyButtons();
    initSortableTables();
    updateProgressBadges();
    openTabFromHash();
    initMermaid();
    initKatex();

    // Hash-linked tabs re-open on navigation
    window.addEventListener('hashchange', openTabFromHash);
  });
})();

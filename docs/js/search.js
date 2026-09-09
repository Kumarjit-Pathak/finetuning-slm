/* ============================================================
   SLM Fine-Tuning Tutorial — Client-side search (stub)
   Populated when Round 6 (deployment) builds the search index.
   ============================================================ */

(function () {
  'use strict';

  // Search index will be inlined as `window.SLM_SEARCH_INDEX` in
  // Round 6 (deployment). Until then, this file just wires up
  // the search UI so the input/keyboard shortcut work.
  const INDEX = window.SLM_SEARCH_INDEX || [];

  function search(query) {
    if (!query) return [];
    const q = query.toLowerCase();
    return INDEX
      .map(item => {
        const hay = (item.title + ' ' + (item.tags || []).join(' ') + ' ' + (item.body || '')).toLowerCase();
        const idx = hay.indexOf(q);
        return idx === -1 ? null : { ...item, score: 1 / (idx + 1) };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score)
      .slice(0, 12);
  }

  function renderResults(results, container) {
    container.innerHTML = '';
    if (results.length === 0) {
      container.innerHTML = '<div class="p-3 text-sm text-slate-500">No matches.</div>';
      return;
    }
    results.forEach(r => {
      const a = document.createElement('a');
      a.href = r.url;
      a.className = 'block p-3 border-b border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800';
      a.innerHTML =
        '<div class="text-sm font-semibold">' + escapeHtml(r.title) + '</div>' +
        '<div class="text-xs text-slate-500 mt-0.5">' + escapeHtml(r.snippet || '') + '</div>';
      container.appendChild(a);
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, m => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[m]);
  }

  document.addEventListener('DOMContentLoaded', () => {
    const input = document.querySelector('[data-search-input]');
    const results = document.querySelector('[data-search-results]');
    if (!input || !results) return;

    input.addEventListener('input', () => {
      renderResults(search(input.value.trim()), results);
    });

    // Keyboard shortcut: "/" focuses search
    document.addEventListener('keydown', e => {
      if (e.key === '/' && document.activeElement !== input && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        input.focus();
      }
    });
  });
})();

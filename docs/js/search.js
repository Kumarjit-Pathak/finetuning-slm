/* ============================================================
   SLM Fine-Tuning Tutorial — Client-side search
   Self-injecting overlay + relevance-scored search across every
   page. Include this file after main.js on any page that should
   support search. The search index lives in search-index.js.
   ============================================================ */

(function () {
  'use strict';

  var INDEX_URL = 'js/search-index.js';

  // ------------------------------------------------------------
  // 1. Inject overlay markup once on load
  // ------------------------------------------------------------
  function injectOverlay() {
    if (document.getElementById('slm-search-overlay')) return;

    var overlay = document.createElement('div');
    overlay.id = 'slm-search-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Search the tutorial');
    overlay.innerHTML = [
      '<div class="slm-search-backdrop" data-slm-close></div>',
      '<div class="slm-search-panel">',
      '  <div class="slm-search-header">',
      '    <input type="search" data-search-input placeholder="Search techniques, use cases, safety..." autocomplete="off" spellcheck="false" aria-label="Search query">',
      '    <kbd class="slm-search-esc" data-slm-close>Esc</kbd>',
      '  </div>',
      '  <div class="slm-search-results" data-search-results>',
      '    <div class="slm-search-empty">Start typing to search. Press <kbd>/</kbd> anywhere to open this dialog.</div>',
      '  </div>',
      '  <div class="slm-search-footer">',
      '    <span><kbd>&#8593;</kbd><kbd>&#8595;</kbd> navigate</span>',
      '    <span><kbd>Enter</kbd> open</span>',
      '    <span><kbd>Esc</kbd> close</span>',
      '  </div>',
      '</div>'
    ].join('');
    document.body.appendChild(overlay);

    overlay.addEventListener('click', function (e) {
      if (e.target.hasAttribute('data-slm-close')) closeSearch();
    });
  }

  // ------------------------------------------------------------
  // 2. Overlay open/close
  // ------------------------------------------------------------
  function openSearch() {
    var overlay = document.getElementById('slm-search-overlay');
    if (!overlay) return;
    overlay.classList.add('is-open');
    var input = overlay.querySelector('[data-search-input]');
    if (input) { input.value = ''; input.focus(); }
    var results = overlay.querySelector('[data-search-results]');
    if (results) results.innerHTML = '<div class="slm-search-empty">Start typing to search. Press <kbd>/</kbd> anywhere to open this dialog.</div>';
  }

  function closeSearch() {
    var overlay = document.getElementById('slm-search-overlay');
    if (overlay) overlay.classList.remove('is-open');
  }

  // ------------------------------------------------------------
  // 3. Search algorithm
  // ------------------------------------------------------------
  function search(query) {
    var INDEX = window.SLM_SEARCH_INDEX || [];
    if (!query) return [];
    var q = query.toLowerCase().trim();
    var terms = q.split(/\s+/).filter(Boolean);
    if (terms.length === 0) return [];

    return INDEX
      .map(function (item) {
        var title = (item.title || '').toLowerCase();
        var body  = ((item.body || '') + ' ' + (item.snippet || '') + ' ' + (item.tags || []).join(' ')).toLowerCase();

        var score = 0;
        for (var i = 0; i < terms.length; i++) {
          var t = terms[i];
          if (title.indexOf(t) !== -1) score += 10;
          if (title.split(/\W+/).indexOf(t) !== -1) score += 20;
          if (body.indexOf(t) !== -1) score += 3;
        }
        return score > 0 ? Object.assign({}, item, { _score: score }) : null;
      })
      .filter(Boolean)
      .sort(function (a, b) { return b._score - a._score; })
      .slice(0, 20);
  }

  function escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function highlightMatch(text, terms) {
    var out = escapeHtml(text);
    terms.forEach(function (t) {
      if (!t) return;
      var re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      out = out.replace(re, '<mark>$1</mark>');
    });
    return out;
  }

  function renderResults(results, container, terms) {
    if (results.length === 0) {
      container.innerHTML = '<div class="slm-search-empty">No matches for that query.</div>';
      return;
    }
    var html = results.map(function (r) {
      return [
        '<a class="slm-search-hit" href="' + escapeHtml(r.url) + '" data-slm-hit tabindex="0">',
        '  <div class="slm-search-hit-title">' + highlightMatch(r.title, terms) + '</div>',
        '  <div class="slm-search-hit-url">' + escapeHtml(r.url) + '</div>',
        '  ' + (r.snippet ? '<div class="slm-search-hit-snippet">' + highlightMatch(r.snippet, terms) + '</div>' : ''),
        '</a>'
      ].join('');
    }).join('');
    container.innerHTML = html;
  }

  // ------------------------------------------------------------
  // 4. Keyboard shortcuts
  // ------------------------------------------------------------
  function bindKeyboard() {
    document.addEventListener('keydown', function (e) {
      var overlay = document.getElementById('slm-search-overlay');
      var isOpen = overlay && overlay.classList.contains('is-open');
      var target = e.target;
      var isInput = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);

      if (e.key === '/' && !isInput && !e.metaKey && !e.ctrlKey && !isOpen) {
        e.preventDefault();
        openSearch();
        return;
      }
      if (isOpen && e.key === 'Escape') {
        e.preventDefault();
        closeSearch();
        return;
      }
      if (isOpen && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
        var hits = Array.from(overlay.querySelectorAll('[data-slm-hit]'));
        if (hits.length === 0) return;
        var idx = hits.indexOf(document.activeElement);
        e.preventDefault();
        var next = idx + (e.key === 'ArrowDown' ? 1 : -1);
        if (next < 0) next = hits.length - 1;
        if (next >= hits.length) next = 0;
        hits[next].focus();
      }
    });
  }

  // ------------------------------------------------------------
  // 5. Load search-index.js dynamically
  // ------------------------------------------------------------
  function loadIndex(cb) {
    if (window.SLM_SEARCH_INDEX) { cb(); return; }
    var s = document.createElement('script');
    s.src = INDEX_URL;
    s.async = true;
    s.onload = cb;
    s.onerror = function () { console.warn('[slm] search index failed to load'); cb(); };
    document.head.appendChild(s);
  }

  // ------------------------------------------------------------
  // 6. Nav "Search" button injection
  // ------------------------------------------------------------
  function injectSearchButton() {
    var toggles = document.querySelectorAll('[data-theme-toggle]');
    toggles.forEach(function (t) {
      if (t.parentNode.querySelector('[data-search-open]')) return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-ghost text-xs';
      btn.setAttribute('data-search-open', '');
      btn.setAttribute('aria-label', 'Open search');
      btn.innerHTML = 'Search <kbd class="slm-search-kbd">/</kbd>';
      btn.addEventListener('click', openSearch);
      t.parentNode.insertBefore(btn, t);
    });
  }

  // ------------------------------------------------------------
  // Bootstrap
  // ------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    injectOverlay();
    injectSearchButton();
    bindKeyboard();
    loadIndex(function () {
      var input = document.querySelector('[data-search-input]');
      var results = document.querySelector('[data-search-results]');
      if (!input || !results) return;
      input.addEventListener('input', function () {
        var q = input.value.trim();
        var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
        renderResults(search(q), results, terms);
      });
      input.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown') {
          var hits = document.querySelectorAll('[data-slm-hit]');
          if (hits.length > 0) { e.preventDefault(); hits[0].focus(); }
        }
      });
    });
  });

  window.slmOpenSearch = openSearch;
})();

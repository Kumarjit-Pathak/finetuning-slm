// SLM Fine-Tuning Tutorial — Interactivity

// Dark mode toggle
function toggleDarkMode() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}

// Load saved theme
document.addEventListener('DOMContentLoaded', () => {
  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
});

// Table sorting
function sortTable(table, column, asc = true) {
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));

  rows.sort((a, b) => {
    const aVal = a.cells[column].textContent.trim();
    const bVal = b.cells[column].textContent.trim();
    const aNum = parseFloat(aVal);
    const bNum = parseFloat(bVal);

    if (!isNaN(aNum) && !isNaN(bNum)) {
      return asc ? aNum - bNum : bNum - aNum;
    }
    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  });

  rows.forEach(row => tbody.appendChild(row));
}

// Initialize sortable tables
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.comparison-table').forEach(table => {
    const headers = table.querySelectorAll('th');
    headers.forEach((th, i) => {
      let asc = true;
      th.addEventListener('click', () => {
        sortTable(table, i, asc);
        asc = !asc;
      });
    });
  });
});

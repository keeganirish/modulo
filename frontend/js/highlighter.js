export function bindHighlighter(root) {
  let pinned = [];
  const ids = element => (element?.dataset.sourceIds || '').split(/\s+/).filter(Boolean);
  const paint = active => root.querySelectorAll('[data-source-ids]').forEach(element => element.classList.toggle('linked-active', ids(element).some(id => active.includes(id))));
  const target = event => event.target.closest('[data-source-ids]');
  root.addEventListener('pointerover', event => { if (!pinned.length) paint(ids(target(event))); });
  root.addEventListener('pointerout', () => { if (!pinned.length) paint([]); });
  root.addEventListener('focusin', event => { if (!pinned.length) paint(ids(target(event))); });
  root.addEventListener('focusout', () => { if (!pinned.length) paint([]); });
  const toggle = element => { const next = ids(element); pinned = pinned.join(' ') === next.join(' ') ? [] : next; paint(pinned); };
  root.addEventListener('click', event => { if (target(event)) toggle(target(event)); });
  root.addEventListener('keydown', event => {
    if (event.key === 'Escape') { pinned = []; paint([]); }
    if ((event.key === 'Enter' || event.key === ' ') && target(event)?.tagName.toLowerCase() === 'rect') { event.preventDefault(); toggle(target(event)); }
  });
  return () => { pinned = []; paint([]); };
}

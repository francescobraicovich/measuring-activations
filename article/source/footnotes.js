// Footnotes remain ordinary linked endnotes when JavaScript is unavailable.
const notes = new Map();
let activeNote;
let closeTimer;

function positionNote() {
  if (!activeNote) return;
  const { trigger, panel } = activeNote;
  const bounds = trigger.getBoundingClientRect();
  const column = trigger.closest('.prose').getBoundingClientRect();
  const margin = 12;
  const gutter = 18;
  const rightSpace = innerWidth - margin - column.right - gutter;
  const leftSpace = column.left - gutter - margin;
  const width = panel.getBoundingClientRect().width;
  let left = (innerWidth - width) / 2;
  const inMargin = Math.max(rightSpace, leftSpace) >= width;
  if (rightSpace >= width) {
    left = column.right + gutter;
  } else if (leftSpace >= width) {
    left = column.left - gutter - width;
  }
  const height = panel.getBoundingClientRect().height;
  let top = bounds.top - 8;
  if (!inMargin) {
    top = bounds.bottom + 12;
    if (top + height > innerHeight - margin) top = bounds.top - height - 12;
  }
  panel.style.left = `${left}px`;
  panel.style.top = `${Math.max(margin, Math.min(top, innerHeight - height - margin))}px`;
}

function closeNote() {
  clearTimeout(closeTimer);
  if (!activeNote) return;
  activeNote.panel.hidden = true;
  activeNote.trigger.setAttribute('aria-expanded', 'false');
  activeNote = undefined;
}

function openNote(trigger, panel, pinned = false) {
  clearTimeout(closeTimer);
  const wasPinned = activeNote?.trigger === trigger && activeNote.pinned;
  closeNote();
  activeNote = { trigger, panel, pinned: pinned || wasPinned };
  panel.hidden = false;
  trigger.setAttribute('aria-expanded', 'true');
  positionNote();
}

function scheduleClose() {
  clearTimeout(closeTimer);
  closeTimer = setTimeout(() => {
    if (activeNote && !activeNote.pinned && !activeNote.panel.contains(document.activeElement)) closeNote();
  }, 180);
}

document.querySelectorAll('.footnote-ref').forEach(reference => {
  const anchor = reference.querySelector('a');
  const target = document.getElementById(anchor.hash.slice(1));
  if (!target) return;
  let panel = notes.get(target.id);
  if (!panel) {
    panel = document.createElement('aside');
    panel.className = 'sidenote';
    panel.id = `note-${target.id}`;
    panel.hidden = true;
    panel.setAttribute('role', 'note');
    panel.setAttribute('aria-label', `Footnote ${notes.size + 1}`);
    const heading = document.createElement('div');
    heading.className = 'note-heading';
    heading.textContent = `[${notes.size + 1}] `;
    const dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.className = 'plain-button note-dismiss';
    dismiss.textContent = 'close';
    dismiss.addEventListener('click', () => {
      const trigger = activeNote.trigger;
      closeNote();
      trigger.focus();
    });
    heading.append(dismiss);
    panel.append(heading);
    const content = target.cloneNode(true);
    content.querySelectorAll('.footnote-backref').forEach(link => link.remove());
    panel.append(...content.childNodes);
    panel.addEventListener('pointerenter', () => clearTimeout(closeTimer));
    panel.addEventListener('pointerleave', scheduleClose);
    panel.addEventListener('focusout', scheduleClose);
    document.body.append(panel);
    notes.set(target.id, panel);
  }

  const trigger = document.createElement('span');
  trigger.className = 'note-trigger';
  trigger.tabIndex = 0;
  trigger.setAttribute('role', 'button');
  trigger.setAttribute('aria-expanded', 'false');
  trigger.setAttribute('aria-controls', panel.id);
  // Highlighted phrases and their reference form one wrapping target.
  const highlighted = reference.previousSibling;
  if (highlighted?.nodeName === 'MARK') {
    reference.before(trigger);
    trigger.append(highlighted);
  } else {
    reference.before(trigger);
    trigger.setAttribute('aria-label', `Open footnote ${anchor.textContent}`);
  }
  const number = document.createElement('sup');
  number.textContent = anchor.textContent.replace(/[\[\]]/g, '');
  trigger.id = anchor.id;
  trigger.append(number);
  reference.remove();

  const toggle = () => {
    if (activeNote?.trigger === trigger && activeNote.pinned) closeNote();
    else openNote(trigger, panel, true);
  };
  trigger.addEventListener('pointerenter', event => {
    if (event.pointerType === 'mouse') openNote(trigger, panel);
  });
  trigger.addEventListener('pointerleave', scheduleClose);
  trigger.addEventListener('click', toggle);
  trigger.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggle();
    }
  });
  trigger.addEventListener('focusout', scheduleClose);
});
if (notes.size) document.body.classList.add('notes-enhanced');

document.addEventListener('pointerdown', event => {
  if (activeNote && !activeNote.panel.contains(event.target) && !activeNote.trigger.contains(event.target)) closeNote();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && activeNote) {
    const trigger = activeNote.trigger;
    const restoreFocus = activeNote.panel.contains(document.activeElement);
    closeNote();
    if (restoreFocus) trigger.focus();
  }
});
addEventListener('resize', positionNote);
addEventListener('scroll', positionNote, { passive: true });

const emailDialog = document.querySelector('#email-dialog');
if (emailDialog) {
  document.querySelector('[data-email-open]').addEventListener('click', event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    closeNote();
    emailDialog.showModal();
  });
  emailDialog.addEventListener('click', event => {
    if (event.target !== emailDialog) return;
    const bounds = emailDialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) emailDialog.close();
  });
}

const additionalResults = document.querySelector('.additional-results');
additionalResults?.addEventListener('toggle', () => {
  if (additionalResults.open) requestAnimationFrame(() => Plotly.Plots.resize('depth-chart'));
});

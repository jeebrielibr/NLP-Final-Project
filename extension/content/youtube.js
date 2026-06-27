/**
 * YouTube Content Script — observes comments, extracts text, classifies via background,
 * and applies hide/highlight actions based on user settings.
 */

// --- State ---
const processedComments = new Set(); // comment IDs already classified
const pendingQueue = new Map();       // id → text, waiting to be classified
const classificationCache = new Map();// id → { label, score }
let debounceTimer = null;
let observer = null;
let sessionJudolCount = 0;
let sessionTotalCount = 0;

// --- Settings (loaded from storage) ---
let settings = { enabled: true, mode: 'highlight', threshold: 0.70 };

/**
 * Load settings from chrome.storage.sync.
 */
async function loadSettings() {
  const stored = await chrome.storage.sync.get(['enabled', 'mode', 'threshold']);
  settings = {
    enabled: stored.enabled ?? true,
    mode: stored.mode ?? 'highlight',
    threshold: stored.threshold ?? 0.70,
  };
}

/**
 * Listen for settings changes from popup.
 */
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'sync') {
    if (changes.enabled) settings.enabled = changes.enabled.newValue;
    if (changes.mode) {
      settings.mode = changes.mode.newValue;
      reapplyAllActions();
    }
    if (changes.threshold) {
      settings.threshold = changes.threshold.newValue;
      reapplyAllActions();
    }
  }
});

// --- DOM Selectors for YouTube comments ---
const COMMENT_SELECTOR = 'ytd-comment-thread-renderer';
const COMMENT_TEXT_SELECTOR = '#content-text';
const COMMENT_ID_ATTR = 'data-comment-id';

/**
 * Extract comment ID from a comment element.
 * YouTube assigns unique IDs we can use; fall back to index-based hash.
 */
function getCommentId(el) {
  const id = el.getAttribute(COMMENT_ID_ATTR);
  if (id) return id;
  // Fallback: generate stable ID from text content
  const textEl = el.querySelector(COMMENT_TEXT_SELECTOR);
  const text = textEl ? textEl.textContent.trim() : '';
  return 'hash_' + simpleHash(text);
}

function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash).toString(36);
}

/**
 * Extract visible text from a comment element and preprocess it.
 */
function extractCommentText(el) {
  const textEl = el.querySelector(COMMENT_TEXT_SELECTOR);
  if (!textEl) return '';
  const raw = textEl.textContent.trim();
  return preprocessText(raw);
}

// --- Classification pipeline ---

/** Snapshot of the batch currently being processed (for re-queue on error). */
let currentBatch = {};

/**
 * Listen for individual classification results sent by the service worker
 * via chrome.tabs.sendMessage (avoids long-lived sendResponse channels).
 */
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'CLASSIFY_RESULT') {
    const { id, result } = message;

    classificationCache.set(id, result);
    processedComments.add(id);
    sessionTotalCount++;

    if (!result.error && result.label === 'judol' && result.score >= settings.threshold) {
      sessionJudolCount++;
    }

    applyAction(id, result);

    // Remove from currentBatch so it won't be re-queued on failure
    delete currentBatch[id];

    // Report stats to background
    chrome.runtime.sendMessage({
      type: 'UPDATE_STATS',
      data: { judolCount: sessionJudolCount, totalCount: sessionTotalCount },
    });
  }

  if (message.type === 'CLASSIFY_BATCH_DONE') {
    // Any remaining items in currentBatch failed (tab disconnect, etc.)
    for (const [id, text] of Object.entries(currentBatch)) {
      pendingQueue.set(id, text);
    }
    currentBatch = {};

    // If there are re-queued items, schedule a retry
    if (pendingQueue.size > 0) {
      scheduleProcessing();
    }
  }
});

/**
 * Process a batch of queued comments.
 * Sends the batch to the service worker and receives results
 * individually via chrome.tabs.sendMessage (see listener above).
 */
async function processQueue() {
  if (pendingQueue.size === 0) return;

  // Snapshot current queue as the batch being processed
  currentBatch = {};
  for (const [id, text] of pendingQueue) {
    currentBatch[id] = text;
  }
  pendingQueue.clear();

  try {
    chrome.runtime.sendMessage({
      type: 'CLASSIFY_BATCH',
      data: currentBatch,
    });
  } catch (err) {
    console.error('Failed to send batch to service worker:', err);
    // Re-queue everything
    for (const [id, text] of Object.entries(currentBatch)) {
      pendingQueue.set(id, text);
    }
    currentBatch = {};
  }
}

/**
 * Debounced processor — waits for DOM mutation bursts to settle.
 */
function scheduleProcessing() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    processQueue();
    debounceTimer = null;
  }, 1500);
}

// --- Action application (hide / highlight) ---

/**
 * Apply hide or highlight to a classified comment.
 */
function applyAction(commentId, result) {
  if (!settings.enabled) return;

  const el = findCommentElement(commentId);
  if (!el) return;

  const isJudol = !result.error && result.label === 'judol' && result.score >= settings.threshold;

  // Remove previous action classes
  el.classList.remove('judol-highlight', 'judol-hide');

  if (isJudol) {
    if (settings.mode === 'hide') {
      el.classList.add('judol-hide');
    } else {
      el.classList.add('judol-highlight');
      // Add confidence badge
      addBadge(el, result.score);
    }
  } else {
    removeBadge(el);
  }
}

/**
 * Re-apply actions to all previously classified comments
 * (used when user toggles mode or threshold).
 */
function reapplyAllActions() {
  for (const [id, result] of classificationCache) {
    applyAction(id, result);
  }
}

/**
 * Find a comment DOM element by its ID.
 */
function findCommentElement(commentId) {
  // Try data attribute first
  const byAttr = document.querySelector(`[${COMMENT_ID_ATTR}="${commentId}"]`);
  if (byAttr) return byAttr;

  // Fallback: iterate comments and match by hash
  const comments = document.querySelectorAll(COMMENT_SELECTOR);
  for (const el of comments) {
    const elId = getCommentId(el);
    if (elId === commentId) return el;
  }
  return null;
}

// --- Badge (confidence % overlay) ---

function addBadge(el, score) {
  removeBadge(el); // Remove existing first
  const badge = document.createElement('span');
  badge.className = 'judol-badge';
  badge.textContent = `⚠ Judol ${Math.round(score * 100)}%`;
  const textEl = el.querySelector(COMMENT_TEXT_SELECTOR);
  if (textEl) {
    textEl.parentElement.insertBefore(badge, textEl);
  }
}

function removeBadge(el) {
  const existing = el.querySelector('.judol-badge');
  if (existing) existing.remove();
}

// --- MutationObserver setup ---

/**
 * Scan all currently visible comments and queue new ones.
 */
function scanComments() {
  if (!settings.enabled) return;

  const comments = document.querySelectorAll(COMMENT_SELECTOR);
  for (const el of comments) {
    const id = getCommentId(el);
    if (processedComments.has(id)) continue;
    if (pendingQueue.has(id)) continue;

    const text = extractCommentText(el);
    if (!text) continue;

    // Check cache first (from previous page visits)
    if (classificationCache.has(id)) {
      processedComments.add(id);
      sessionTotalCount++;
      if (classificationCache.get(id).label === 'judol') sessionJudolCount++;
      applyAction(id, classificationCache.get(id));
      continue;
    }

    pendingQueue.set(id, text);
  }

  if (pendingQueue.size > 0) {
    scheduleProcessing();
  }
}

/**
 * Start observing YouTube comment section for new/changed comments.
 */
function startObserver() {
  const commentsSection = document.querySelector('#comments') ||
                          document.querySelector('ytd-comments') ||
                          document.querySelector('#comment-section-renderer');

  if (!commentsSection) {
    // Comments section not loaded yet — retry
    setTimeout(startObserver, 2000);
    return;
  }

  observer = new MutationObserver((mutations) => {
    let hasNewComments = false;
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.matches?.(COMMENT_SELECTOR) || node.querySelector?.(COMMENT_SELECTOR)) {
            hasNewComments = true;
          }
        }
      }
    }
    if (hasNewComments) {
      scanComments();
    }
  });

  observer.observe(commentsSection, { childList: true, subtree: true });
  scanComments(); // Initial scan
}

// --- YouTube SPA navigation handling ---
// YouTube uses SPA navigation — content scripts persist but the DOM changes.
// Watch for URL changes to restart the observer.

let lastUrl = location.href;

function onNavigationChange() {
  if (location.href === lastUrl) return;
  lastUrl = location.href;

  // Reset state for new video page
  processedComments.clear();
  pendingQueue.clear();
  sessionJudolCount = 0;
  sessionTotalCount = 0;
  classificationCache.clear();

  // Remove all badges and action classes
  document.querySelectorAll('.judol-highlight, .judol-hide').forEach(el => {
    el.classList.remove('judol-highlight', 'judol-hide');
  });
  document.querySelectorAll('.judol-badge').forEach(el => el.remove());

  // Restart observer for new comment section
  if (observer) observer.disconnect();
  observer = null;
  startObserver();

  chrome.runtime.sendMessage({
    type: 'UPDATE_STATS',
    data: { judolCount: 0, totalCount: 0 },
  });
}

// Listen for YouTube SPA navigation via popstate and pushState
window.addEventListener('popstate', onNavigationChange);

// Also patch history.pushState/replaceState to catch in-page navigations
const origPushState = history.pushState;
history.pushState = function () {
  origPushState.apply(this, arguments);
  onNavigationChange();
};
const origReplaceState = history.replaceState;
history.replaceState = function () {
  origReplaceState.apply(this, arguments);
  onNavigationChange();
};

// --- Initialization ---
async function init() {
  await loadSettings();
  startObserver();
}

init();

/**
 * Service Worker — handles Space API calls and message routing.
 * Uses custom Space API (FastAPI) to classify comments with IndoBERT Focal model.
 */

const API_URL = 'https://brielibr-indobert-judol-api.hf.space/predict';
const API_TIMEOUT_MS = 45000;
const BATCH_SIZE = 5;

let apiAvailable = true;

async function classifySingle(text) {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text }),
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const result = await response.json();
    apiAvailable = true;
    return {
      label: result.label,
      score: result.score,
    };
  } catch (err) {
    console.error('API Error:', err);
    apiAvailable = false;
    return { label: 'bukan_judol', score: 0, error: true };
  }
}

/**
 * Classify a batch and send each result back to the calling tab
 * individually via chrome.tabs.sendMessage. This avoids holding a
 * long-lived sendResponse channel that Chrome may close prematurely.
 */
async function classifyBatchAndReply(commentMap, tabId) {
  const entries = Object.entries(commentMap);

  for (let i = 0; i < entries.length; i += BATCH_SIZE) {
    const chunk = entries.slice(i, i + BATCH_SIZE);
    for (const [id, text] of chunk) {
      const result = await classifySingle(text);

      try {
        chrome.tabs.sendMessage(tabId, {
          type: 'CLASSIFY_RESULT',
          id: id,
          result: result,
        });
      } catch (e) {
        // Tab may have navigated away — skip silently
        console.warn('Could not send result to tab:', e);
      }

      if (i + BATCH_SIZE < entries.length) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }
  }

  // Signal batch completion so the content script can re-queue failures
  try {
    chrome.tabs.sendMessage(tabId, { type: 'CLASSIFY_BATCH_DONE' });
  } catch (e) {
    console.warn('Could not send batch-done to tab:', e);
  }
}

async function checkApiHealth() {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: 'test' }),
      signal: AbortSignal.timeout(5000),
    });
    apiAvailable = response.ok;
    return apiAvailable;
  } catch {
    apiAvailable = false;
    return false;
  }
}

// --- Message handlers from content script and popup ---

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'CLASSIFY_BATCH': {
      // Fire off batch processing in background — no sendResponse needed.
      // Results are delivered via chrome.tabs.sendMessage one by one.
      const tabId = sender.tab?.id;
      if (tabId) {
        classifyBatchAndReply(message.data, tabId);
      }
      sendResponse({ ack: true });
      break;
    }

    case 'CLASSIFY_SINGLE':
      classifySingle(message.text)
        .then(result => sendResponse(result))
        .catch(err => sendResponse({ label: 'bukan_judol', score: 0, error: true }));
      return true;

    case 'CHECK_API':
      checkApiHealth()
        .then(available => sendResponse({ available }))
        .catch(() => sendResponse({ available: false }));
      return true;

    case 'GET_API_STATUS':
      sendResponse({ available: apiAvailable });
      break;

    case 'UPDATE_STATS': {
      const { judolCount, totalCount } = message.data;
      chrome.storage.local.set({ sessionStats: { judolCount, totalCount } });
      if (judolCount > 0) {
        chrome.action.setBadgeText({ text: String(judolCount) });
        chrome.action.setBadgeBackgroundColor({ color: '#e53935' });
      } else {
        chrome.action.setBadgeText({ text: '' });
      }
      sendResponse({ ok: true });
      break;
    }

    default:
      console.warn('Unknown message type:', message.type);
  }
});

// Initialize default settings on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({
    enabled: true,
    mode: 'highlight',
    threshold: 0.70,
  });
  chrome.storage.local.set({ sessionStats: { judolCount: 0, totalCount: 0 } });
  chrome.action.setBadgeText({ text: '' });
});

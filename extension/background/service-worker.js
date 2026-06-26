/**
 * Service Worker — handles HF Inference API calls and message routing.
 * Uses HuggingFace Inference API to classify comments with IndoBERT Focal model.
 */

const HF_API_URL = 'https://api-inference.huggingface.co/models/';
// Replace with your actual HF model repo path after deployment
const HF_MODEL_ID = '<YOUR_HF_USERNAME>/indobert-judol-focal';
const API_TIMEOUT_MS = 10000;
const BATCH_SIZE = 5;

let apiAvailable = true;

/**
 * Classify a single comment text via HF Inference API.
 * Returns { label: 'judol'|'bukan_judol', score: number }
 */
async function classifySingle(text) {
  try {
    const response = await fetch(HF_API_URL + HF_MODEL_ID, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: text }),
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });

    if (!response.ok) {
      const errBody = await response.text();
      console.error('HF API error:', response.status, errBody);
      apiAvailable = false;
      return { label: 'bukan_judol', score: 0, error: true };
    }

    const result = await response.json();

    // HF Inference API returns array of label-score pairs
    // e.g. [{ label: 'LABEL_1', score: 0.97 }, { label: 'LABEL_0', score: 0.03 }]
    if (Array.isArray(result) && result.length > 0) {
      // Find the highest-score prediction
      const topPred = result.reduce((a, b) => a.score > b.score ? a : b);
      const label = topPred.label === 'LABEL_1' ? 'judol' : 'bukan_judol';
      apiAvailable = true;
      return { label, score: topPred.score };
    }

    // Some models return a single object
    if (result.label && result.score) {
      const label = result.label === 'LABEL_1' ? 'judol' : 'bukan_judol';
      apiAvailable = true;
      return { label, score: result.score };
    }

    console.error('Unexpected API response format:', result);
    return { label: 'bukan_judol', score: 0, error: true };
  } catch (err) {
    if (err.name === 'AbortError') {
      console.error('HF API request timed out');
    } else {
      console.error('HF API request failed:', err);
    }
    apiAvailable = false;
    return { label: 'bukan_judol', score: 0, error: true };
  }
}

/**
 * Classify a batch of comments. Groups texts into BATCH_SIZE chunks
 * and sends them sequentially to avoid rate-limiting.
 * Returns Map<commentId, { label, score }>
 */
async function classifyBatch(commentMap) {
  const entries = Object.entries(commentMap);
  const results = {};

  for (let i = 0; i < entries.length; i += BATCH_SIZE) {
    const chunk = entries.slice(i, i + BATCH_SIZE);
    // Process chunk sequentially within the batch
    for (const [id, text] of chunk) {
      results[id] = await classifySingle(text);
      // Small delay between requests to respect rate limits
      if (i + BATCH_SIZE < entries.length) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }
  }

  return results;
}

/**
 * Check if the HF API is reachable (lightweight probe).
 */
async function checkApiHealth() {
  try {
    const response = await fetch(HF_API_URL + HF_MODEL_ID, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: 'test' }),
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
    case 'CLASSIFY_BATCH':
      classifyBatch(message.data).then(sendResponse);
      return true; // Indicates async response

    case 'CLASSIFY_SINGLE':
      classifySingle(message.text).then(sendResponse);
      return true;

    case 'CHECK_API':
      checkApiHealth().then(sendResponse);
      return true;

    case 'GET_API_STATUS':
      sendResponse({ available: apiAvailable });
      break;

    case 'UPDATE_STATS': {
      const { judolCount, totalCount } = message.data;
      // Store session stats
      chrome.storage.local.set({ sessionStats: { judolCount, totalCount } });
      // Update badge
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

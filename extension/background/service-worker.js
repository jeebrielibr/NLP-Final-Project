/**
 * Service Worker — handles Space API calls and message routing.
 * Uses custom Space API (FastAPI) to classify comments with IndoBERT Focal model.
 */

// Ganti URL Inference API dengan URL Space Anda
const API_URL = 'https://brielibr-indobert-judol-api.hf.space/predict';
const API_TIMEOUT_MS = 45000; // 45 detik timeout untuk permintaan API
const BATCH_SIZE = 5;

let apiAvailable = true;

async function classifySingle(text) {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // TIDAK ADA HEADER AUTHORIZATION LAGI!
      body: JSON.stringify({ text: text }), 
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    // Format respons sudah disesuaikan oleh FastAPI kita
    const result = await response.json(); 
    
    apiAvailable = true;
    return { 
        label: result.label,       // Langsung 'judol' atau 'bukan_judol'
        score: result.score 
    };

  } catch (err) {
    console.error('API Error:', err);
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
 * Check if the API is reachable (lightweight probe).
 */
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

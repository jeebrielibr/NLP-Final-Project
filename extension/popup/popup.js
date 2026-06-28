/**
 * Popup logic — manages settings UI, persists via chrome.storage,
 * and reports API status and session stats.
 */

// --- DOM refs ---
const enabledCheckbox = document.getElementById('enabled');
const modeHighlight = document.querySelector('input[name="mode"][value="highlight"]');
const modeHide = document.querySelector('input[name="mode"][value="hide"]');
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('threshold-value');
const apiStatusDot = document.getElementById('api-status-dot');
const apiStatusText = document.getElementById('api-status-text');
const judolCountEl = document.getElementById('judol-count');
const totalCountEl = document.getElementById('total-count');

// --- Load current settings ---
async function loadUIState() {
  const syncData = await chrome.storage.sync.get(['enabled', 'mode', 'threshold']);
  enabledCheckbox.checked = syncData.enabled ?? true;
  (syncData.mode === 'hide' ? modeHide : modeHighlight).checked = true;
  thresholdSlider.value = syncData.threshold ?? 0.70;
  thresholdValue.textContent = thresholdSlider.value;

  const localData = await chrome.storage.local.get('sessionStats');
  const stats = localData.sessionStats ?? { judolCount: 0, totalCount: 0 };
  judolCountEl.textContent = stats.judolCount;
  totalCountEl.textContent = stats.totalCount;
}

// --- Event handlers ---
enabledCheckbox.addEventListener('change', () => {
  chrome.storage.sync.set({ enabled: enabledCheckbox.checked });
});

modeHighlight.addEventListener('change', () => {
  if (modeHighlight.checked) {
    chrome.storage.sync.set({ mode: 'highlight' });
  }
});

modeHide.addEventListener('change', () => {
  if (modeHide.checked) {
    chrome.storage.sync.set({ mode: 'hide' });
  }
});

thresholdSlider.addEventListener('input', () => {
  thresholdValue.textContent = thresholdSlider.value;
  chrome.storage.sync.set({ threshold: parseFloat(thresholdSlider.value) });
});

// --- API status check ---
async function checkApiStatus() {
  apiStatusDot.className = 'status-dot';
  apiStatusText.textContent = 'Memeriksa API...';

  try {
    const result = await chrome.runtime.sendMessage({ type: 'CHECK_API' });
    if (result) {
      apiStatusDot.classList.add('available');
      apiStatusText.textContent = 'API tersedia';
    } else {
      apiStatusDot.classList.add('unavailable');
      apiStatusText.textContent = 'API tidak tersedia';
    }
  } catch {
    apiStatusDot.classList.add('unavailable');
    apiStatusText.textContent = 'API tidak tersedia';
  }
}

// --- Stats refresh (poll every 2s while popup is open) ---
function refreshStats() {
  chrome.storage.local.get('sessionStats').then(data => {
    const stats = data.sessionStats ?? { judolCount: 0, totalCount: 0 };
    judolCountEl.textContent = stats.judolCount;
    totalCountEl.textContent = stats.totalCount;
  });
}

setInterval(refreshStats, 2000);

// --- Init ---
loadUIState();
checkApiStatus();
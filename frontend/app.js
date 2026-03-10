/* Focus.ai – frontend logic */
(function () {
  'use strict';

  const API_VERIFY = '/api/v1/verify';
  const CIRCUMFERENCE = 2 * Math.PI * 50; // r=50 → ≈ 314.16

  // DOM refs
  const dropZone      = document.getElementById('drop-zone');
  const fileInput     = document.getElementById('file-input');
  const previewImg    = document.getElementById('preview-img');
  const fileInfo      = document.getElementById('file-info');
  const fileNameEl    = document.getElementById('file-name');
  const clearBtn      = document.getElementById('clear-btn');
  const verifyBtn     = document.getElementById('verify-btn');
  const uploadSection = document.getElementById('upload-section');
  const resultsSection= document.getElementById('results-section');
  const loadingOverlay= document.getElementById('loading-overlay');
  const newAnalysisBtn= document.getElementById('new-analysis-btn');
  const errorBanner   = document.getElementById('error-banner');
  const errorMsg      = document.getElementById('error-message');
  const errorClose    = document.getElementById('error-close-btn');

  let selectedFile = null;

  /* ── file selection ─────────────────────────────────────────── */
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  clearBtn.addEventListener('click', clearFile);
  newAnalysisBtn.addEventListener('click', resetToUpload);
  verifyBtn.addEventListener('click', runVerification);
  errorClose.addEventListener('click', hideError);

  function setFile(file) {
    selectedFile = file;
    fileNameEl.textContent = file.name;
    fileInfo.classList.remove('hidden');
    verifyBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewImg.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewImg.classList.add('hidden');
    fileInfo.classList.add('hidden');
    verifyBtn.disabled = true;
  }

  function resetToUpload() {
    clearFile();
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
  }

  /* ── API call ────────────────────────────────────────────────── */
  async function runVerification() {
    if (!selectedFile) return;
    showLoading(true);

    try {
      const form = new FormData();
      form.append('file', selectedFile);

      const resp = await fetch(API_VERIFY, { method: 'POST', body: form });

      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `Server error ${resp.status}`);
      }

      const data = await resp.json();
      showLoading(false);
      renderResults(data);
    } catch (err) {
      showLoading(false);
      showError(err.message || 'Unknown error');
    }
  }

  /* ── render results ──────────────────────────────────────────── */
  function renderResults(data) {
    uploadSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    const scorePct = Math.round(data.authenticity_score * 100);

    // Score arc
    const arc = document.getElementById('score-arc');
    const filled = (scorePct / 100) * CIRCUMFERENCE;
    arc.style.strokeDasharray = `${filled} ${CIRCUMFERENCE}`;
    arc.style.stroke = tierColour(data.tier);

    document.getElementById('score-value').textContent = scorePct;

    // Tier badge
    const tierBadge = document.getElementById('tier-badge');
    tierBadge.textContent = data.tier.replace(/_/g, ' ');
    tierBadge.className = 'tier-badge tier-' + data.tier;

    document.getElementById('confidence-label').textContent =
      'Confidence: ' + data.confidence;
    document.getElementById('recommendation-text').textContent =
      data.recommendation;
    document.getElementById('summary-text').textContent = data.summary;

    // Module bars
    const modules = data.module_scores;
    Object.entries(modules).forEach(([key, val]) => {
      const pct = Math.round(val * 100);
      const bar = document.getElementById('bar-' + key);
      const label = document.getElementById('pct-' + key);
      if (bar) {
        bar.style.width = pct + '%';
        bar.style.background = scoreColour(val);
      }
      if (label) label.textContent = pct + '%';
    });

    // Signals
    const list = document.getElementById('signals-list');
    list.innerHTML = '';
    (data.all_signals || []).forEach(sig => {
      const li = document.createElement('li');
      li.textContent = sig;
      li.className = signalClass(sig);
      list.appendChild(li);
    });
  }

  /* ── helpers ─────────────────────────────────────────────────── */
  function tierColour(tier) {
    const map = {
      AUTHENTIC: '#3fb950',
      LIKELY_AUTHENTIC: '#7ee787',
      UNCERTAIN: '#d29922',
      LIKELY_SYNTHETIC: '#e3772c',
      SYNTHETIC: '#f85149',
    };
    return map[tier] || '#58a6ff';
  }

  function scoreColour(val) {
    if (val >= 0.75) return '#3fb950';
    if (val >= 0.55) return '#7ee787';
    if (val >= 0.40) return '#d29922';
    if (val >= 0.25) return '#e3772c';
    return '#f85149';
  }

  function signalClass(sig) {
    const lower = sig.toLowerCase();
    if (lower.includes('consistent') || lower.includes('natural') ||
        lower.includes('authentic') || lower.includes('present') ||
        lower.includes('detected:') || lower.includes('camera')) {
      return 'signal-positive';
    }
    if (lower.includes('possible') || lower.includes('moderate') ||
        lower.includes('mildly') || lower.includes('above-average')) {
      return 'signal-warning';
    }
    if (lower.includes('synthetic') || lower.includes('ai-generated') ||
        lower.includes('no exif') || lower.includes('suspicious') ||
        lower.includes('reject') || lower.includes('highly uniform')) {
      return 'signal-danger';
    }
    return '';
  }

  function showLoading(show) {
    loadingOverlay.classList.toggle('hidden', !show);
  }

  function showError(msg) {
    errorMsg.textContent = msg;
    errorBanner.classList.remove('hidden');
    setTimeout(hideError, 6000);
  }

  function hideError() {
    errorBanner.classList.add('hidden');
  }
})();

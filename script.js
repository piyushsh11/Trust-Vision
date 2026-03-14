const api = 'http://localhost:8000';

const datasetSelect = document.getElementById('datasetSelect');
const modelSelect = document.getElementById('modelSelect');
const datasetInfo = document.getElementById('datasetInfo');
const attackTypeSelect = document.getElementById('attackType');
const epsilonSlider = document.getElementById('epsilon');
const epsLabel = document.getElementById('epsLabel');
const uploadInput = document.getElementById('upload');
const uploadBtn = document.getElementById('uploadBtn');
const preview = document.getElementById('preview');
const attackBtn = document.getElementById('attackBtn');
const defenseBtn = document.getElementById('defenseBtn');
const chipButtons = document.querySelectorAll('.chip');

const predLabel = document.getElementById('predLabel');
const predConf = document.getElementById('predConf');
const predModel = document.getElementById('predModel');
const certRadius = document.getElementById('certRadius');
const origAttack = document.getElementById('origAttack');
const advAttack = document.getElementById('advAttack');
const heatAttack = document.getElementById('heatAttack');
const attackDetails = document.getElementById('attackDetails');

const defenseImg = document.getElementById('defenseImg');
const robustLabel = document.getElementById('robustLabel');
const robustConf = document.getElementById('robustConf');
const robustModel = document.getElementById('robustModel');
const accClean = document.getElementById('accClean');
const accAdv = document.getElementById('accAdv');
const accRobust = document.getElementById('accRobust');
const latency = document.getElementById('latency');

const tabs = document.querySelectorAll('.tab');
const panes = document.querySelectorAll('.tab-pane');
const metricsChartCanvas = document.getElementById('metricsChart');
let chart;

const datasetMeta = {
  set1: 'Industrial + retinal demo pack (new Samples A–D).',
  medmnist: 'MedMNIST demo samples (RetinaMNIST thumbnails).',
  items: 'Items pack (new uploads Samples E–H) tuned for SDG3/SDG9.'
};

const svgPlaceholder = (label, bg) =>
  `data:image/svg+xml;utf8,${encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='640' height='400'><rect width='100%' height='100%' fill='${bg}'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='#111827' font-size='28' font-family='Inter,sans-serif'>${label}</text></svg>`
  )}`;

const samples = {
  set1: {
    stop: 'images/SAMPLE A.png',
    part: 'images/SAMPLE B.png',
    face: 'images/SAMPLE C.png',
    bonus: 'images/SAMPLE D.png'
  },
  medmnist: {
    stop: 'images/medmnist_1.png',
    part: 'images/medmnist_2.png',
    face: 'images/medmnist_3.png',
    extra: 'images/medmnist_4.png',
    bonus: 'images/medmnist_5.png'
  },
  items: {
    stop: 'images/SAMPLE E.png',
    part: 'images/SAMPLE F.png',
    face: 'images/SAMPLE G.png',
    bonus: 'images/SAMPLE H.png'
  }
};

let currentImage = null; // data URL
let currentDataset = datasetSelect ? datasetSelect.value || 'set1' : 'set1';
let currentModel = modelSelect ? modelSelect.value || 'classifier' : 'classifier';
let currentAttack = null;
let currentAttackType = attackTypeSelect ? attackTypeSelect.value : 'fgsm';
let currentEpsilon = epsilonSlider ? Number(epsilonSlider.value) / 255 : 8 / 255;
let lastOriginalLabel = '';
let lastAdvLabel = '';

function setDatasetInfo(key) {
  datasetInfo.textContent = datasetMeta[key] || '';
}

function updateAttackButtons() {
  const disabled = currentModel === 'yolo';
  attackBtn.disabled = disabled;
  defenseBtn.disabled = disabled;
  attackBtn.textContent = disabled ? 'Generate attack (classifier only)' : 'Generate adversarial attack';
  defenseBtn.textContent = disabled ? 'Apply defense (classifier only)' : 'Apply defense';
}

function setPreview(dataUrl) {
  currentImage = dataUrl;
  preview.style.backgroundImage = 'none';
  preview.innerHTML = `<img src="${dataUrl}" alt="preview" style="width:100%;height:100%;object-fit:contain;border-radius:12px;">`;
}

async function fetchToBase64(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

async function selectSample(id) {
  chipButtons.forEach((b) => b.classList.toggle('active', b.dataset.id === id));
  const path = samples[currentDataset][id];
  if (!path) return;
  try {
    const dataUrl = await fetchToBase64(path);
    setPreview(dataUrl);
    await classify(dataUrl);
  } catch (err) {
    console.error('Failed to load sample', err);
    preview.textContent = 'Failed to load sample image';
  }
}

async function classify(dataUrl) {
  try {
    const res = await fetch(`${api}/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, dataset: currentDataset, model: currentModel })
    });
    const out = await res.json();
    if (out.error) {
      predLabel.textContent = out.error;
      predConf.textContent = '—';
      predModel.textContent = '—';
      if (certRadius) certRadius.textContent = '—';
      return;
    }
    const topDetection = Array.isArray(out.detections) && out.detections.length > 0 ? out.detections[0] : null;
    predLabel.textContent = topDetection ? `${topDetection.label} (${out.detections.length} objs)` : out.label;
    predConf.textContent = `${(out.confidence * 100).toFixed(1)}%`;
    predModel.textContent = out.model;
    if (certRadius) certRadius.textContent = out.certified_radius_l2 != null ? `${(out.certified_radius_l2 || 0).toFixed(4)}` : '—';
  } catch (e) {
    predLabel.textContent = 'Backend not running';
    console.error(e);
  }
}

async function generateAttack() {
  if (!currentImage) return;
  try {
    const res = await fetch(`${api}/attack`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: currentImage, dataset: currentDataset, epsilon: currentEpsilon, attack: currentAttackType })
    });
    const out = await res.json();
    currentAttack = out.adv_image;
    lastOriginalLabel = out.original_prediction.label;
    lastAdvLabel = out.adv_prediction.label;
    origAttack.style.backgroundImage = `url('${out.original}')`;
    advAttack.style.backgroundImage = `url('${out.adv_image}')`;
    heatAttack.style.backgroundImage = `url('${out.heatmap}')`;
    attackDetails.textContent = `ε = ${(out.epsilon * 255).toFixed(2)}/255 · ${(out.attack || currentAttackType).toUpperCase()}`;
    predLabel.textContent = out.original_prediction.label;
    predConf.textContent = `${(out.original_prediction.confidence * 100).toFixed(1)}%`;
    updateExplanation(`Attack used ${(out.attack || currentAttackType).toUpperCase()} at ε ${(out.epsilon * 255).toFixed(2)}/255. Original → Adv: ${lastOriginalLabel} → ${lastAdvLabel}. Higher ε increases risk.`);
    goToTab('results');
  } catch (e) {
    attackDetails.textContent = 'Attack failed (is the backend running on http://localhost:8000?)';
    updateExplanation('Attack failed to run; check backend connectivity.');
  }
}

async function applyDefense() {
  if (!currentAttack) return;
  try {
    const res = await fetch(`${api}/defend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ adv_image: currentAttack, dataset: currentDataset })
    });
    const out = await res.json();
    defenseImg.style.backgroundImage = `url('${currentAttack}')`;
    robustLabel.textContent = out.robust_prediction.label;
    robustConf.textContent = `${(out.robust_prediction.confidence * 100).toFixed(1)}%`;
    robustModel.textContent = out.model;
    accClean.textContent = `${(out.clean_accuracy * 100).toFixed(1)}%`;
    accAdv.textContent = `${(out.adv_accuracy * 100).toFixed(1)}%`;
    accRobust.textContent = `${(out.robust_accuracy * 100).toFixed(1)}%`;
    latency.textContent = `${out.latency_ms.toFixed(1)} ms`;
    updateChart(out.clean_accuracy, out.adv_accuracy, out.robust_accuracy);
    const safe = out.robust_prediction.label === lastOriginalLabel && out.robust_prediction.confidence > 0.5;
    const verdict = safe ? 'Safer for this sample (defense recovered the correct label).' : 'Still vulnerable on this sample (defense did not fully recover).';
    updateExplanation(`${verdict} Baseline label: ${lastOriginalLabel || '—'}, Adversarial label: ${lastAdvLabel || '—'}, Defended: ${out.robust_prediction.label} @ ${(out.robust_prediction.confidence * 100).toFixed(1)}%.`);
  } catch (e) {
    robustLabel.textContent = 'Defense failed (backend?)';
    updateExplanation('Defense call failed; backend may be down.');
  }
}

attackTypeSelect.addEventListener('change', () => {
  currentAttackType = attackTypeSelect.value;
});

datasetSelect?.addEventListener('change', async () => {
  currentDataset = datasetSelect.value;
  setDatasetInfo(currentDataset);
  await selectSample('stop');
});

modelSelect?.addEventListener('change', async () => {
  currentModel = modelSelect.value;
  updateAttackButtons();
  await selectSample('stop');
});

epsilonSlider.addEventListener('input', (e) => {
  const val = Number(e.target.value);
  currentEpsilon = val / 255;
  epsLabel.textContent = `${val}/255`;
});

uploadBtn.addEventListener('click', () => uploadInput.click());

uploadInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    setPreview(reader.result);
    await classify(reader.result);
  };
  reader.readAsDataURL(file);
});

chipButtons.forEach((btn) => btn.addEventListener('click', () => selectSample(btn.dataset.id)));
attackBtn.addEventListener('click', generateAttack);
defenseBtn.addEventListener('click', applyDefense);

// init
if (datasetSelect) datasetSelect.value = currentDataset;
if (modelSelect) modelSelect.value = currentModel;
setDatasetInfo(currentDataset);
selectSample('stop');
updateChart(0.9, 0.4, 0.8);
updateAttackButtons();

// tabs
tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((t) => t.classList.remove('active'));
    panes.forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// chart
function updateChart(clean = 0, adv = 0, robust = 0) {
  if (!metricsChartCanvas || typeof Chart === 'undefined') return;
  const data = [clean * 100, adv * 100, robust * 100];
  if (chart) {
    chart.data.datasets[0].data = data;
    chart.update();
    return;
  }
  chart = new Chart(metricsChartCanvas, {
    type: 'bar',
    data: {
      labels: ['Clean', 'Adversarial', 'Robust'],
      datasets: [
        {
          label: 'Accuracy (%)',
          data,
          backgroundColor: ['#3b82f6', '#ef4444', '#10b981'],
          borderWidth: 1
        }
      ]
    },
    options: {
      scales: { y: { beginAtZero: true, max: 100 } },
      plugins: { legend: { display: false } }
    }
  });
}

function updateExplanation(text) {
  const el = document.getElementById('explainDynamic');
  if (el) el.textContent = text;
}

function goToTab(tabId) {
  tabs.forEach((t) => {
    const active = t.dataset.tab === tabId;
    t.classList.toggle('active', active);
  });
  panes.forEach((p) => p.classList.toggle('active', p.id === tabId));
  const pane = document.getElementById(tabId);
  if (pane) pane.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

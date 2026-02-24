/* ──────────────────────────────────────────────────────────────
   System Failure Early Warning Engine – Enterprise Dashboard
   Original 7 upgrades + 10 enterprise features
   ────────────────────────────────────────────────────────────── */

const API = '';
const POLL_MS = 2000;

// ── Server selector ─────────────────────────────────────────
function getServerId() {
    return document.getElementById('serverSelect').value;
}

// ── Sensitivity slider ──────────────────────────────────────
const slider = document.getElementById('sensitivitySlider');
const sensVal = document.getElementById('sensitivityValue');
slider.addEventListener('input', () => { sensVal.textContent = slider.value; });
slider.addEventListener('change', async () => {
    const sid = getServerId();
    await fetch(`${API}/settings/sensitivity?value=${slider.value}&server_id=${sid}`, { method: 'POST' });
});

// ── Inject failure button ───────────────────────────────────
let injecting = false;
async function injectFailure() {
    const btn = document.getElementById('injectBtn');
    if (!injecting) {
        await fetch(`${API}/simulate/failure?count=15`, { method: 'POST' });
        btn.textContent = '⏹ Stop Injection';
        btn.classList.remove('btn-danger-glow');
        btn.style.background = '#ff1744';
        injecting = true;
    } else {
        await fetch(`${API}/simulate/stop`, { method: 'POST' });
        btn.textContent = '💥 Inject Failure';
        btn.classList.add('btn-danger-glow');
        btn.style.background = '';
        injecting = false;
    }
}

// ── Retrain model button ────────────────────────────────────
async function retrainModel() {
    const btn = event.target;
    btn.textContent = '⏳ Training...';
    btn.disabled = true;
    try {
        const res = await fetch(`${API}/model/retrain?server_id=${getServerId()}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            btn.textContent = '✅ Done!';
            setTimeout(() => { btn.textContent = '🔄 Retrain'; btn.disabled = false; }, 3000);
        } else {
            btn.textContent = '❌ Error';
            setTimeout(() => { btn.textContent = '🔄 Retrain'; btn.disabled = false; }, 3000);
        }
    } catch (e) {
        btn.textContent = '🔄 Retrain';
        btn.disabled = false;
    }
}

// ── Charts ──────────────────────────────────────────────────
const chartOpts = {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 400 },
    scales: {
        x: { display: true, grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#666', maxTicksLimit: 8, font: { size: 9 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888', font: { size: 10 } } },
    },
    plugins: { legend: { labels: { color: '#aaa', font: { size: 11 } } } }
};

function makeChart(id, datasets, yOpts = {}) {
    return new Chart(document.getElementById(id), {
        type: 'line',
        data: { labels: [], datasets },
        options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, ...yOpts } } }
    });
}

const cpuMemChart = makeChart('cpuMemChart', [
    { label: 'CPU %', data: [], borderColor: '#00e5ff', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
    { label: 'Memory %', data: [], borderColor: '#b388ff', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
]);

const diskRespChart = makeChart('diskRespChart', [
    { label: 'Disk I/O', data: [], borderColor: '#448aff', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
    { label: 'Response Time ms', data: [], borderColor: '#ffab00', borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0 },
]);

const netChart = makeChart('netChart', [
    { label: 'Network KB/s', data: [], borderColor: '#00e676', borderWidth: 2, fill: { target: 'origin', above: 'rgba(0,230,118,0.08)' }, tension: 0.3, pointRadius: 0 },
]);

const healthChart = new Chart(document.getElementById('healthChart'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Health Score', data: [], borderColor: '#00e676', borderWidth: 2.5, fill: { target: 'origin', above: 'rgba(0,230,118,0.08)' }, tension: 0.3, pointRadius: 0 },
            { label: 'Forecast', data: [], borderColor: '#ffab00', borderWidth: 2, borderDash: [6, 4], fill: false, tension: 0.3, pointRadius: 0 },
        ]
    },
    options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, min: 0, max: 100 } } }
});

const shapChart = new Chart(document.getElementById('shapChart'), {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Contribution', data: [], backgroundColor: [] }] },
    options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888', font: { size: 10 } } },
            y: { grid: { display: false }, ticks: { color: '#ccc', font: { size: 10 } } },
        },
        plugins: { legend: { display: false } }
    }
});

// ── Gauge drawing ───────────────────────────────────────────
function drawGauge(canvas, value) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h - 5, r = Math.min(w, h) - 15;
    ctx.clearRect(0, 0, w, h);

    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0, false);
    ctx.lineWidth = 14; ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.stroke();

    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, '#ff1744'); grad.addColorStop(0.5, '#ffab00'); grad.addColorStop(1, '#00e676');
    const angle = Math.PI + (value / 100) * Math.PI;
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, angle, false);
    ctx.lineWidth = 14; ctx.lineCap = 'round'; ctx.strokeStyle = grad; ctx.stroke();

    const nx = cx + (r - 20) * Math.cos(angle);
    const ny = cy + (r - 20) * Math.sin(angle);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(nx, ny);
    ctx.lineWidth = 2.5; ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff'; ctx.fill();
}

// ── Polling functions ───────────────────────────────────────
async function pollHealth() {
    try {
        const sid = getServerId();
        const res = await fetch(`${API}/health?server_id=${sid}`);
        const d = await res.json();

        document.getElementById('healthValue').textContent = d.health_score;
        drawGauge(document.getElementById('gaugeCanvas'), d.health_score);
        document.getElementById('anomalyScore').textContent = d.anomaly_score?.toFixed(4) || '0.0000';

        // Risk badge
        const badge = document.getElementById('riskBadge');
        const rl = (d.risk_level || 'GREEN').toUpperCase();
        badge.className = 'risk-badge ' + rl.toLowerCase();
        badge.textContent = '● ' + rl;

        // Failure probability
        const fp = d.failure_prob || 0;
        document.getElementById('failureProb').textContent = (fp * 100).toFixed(1) + '%';
        document.getElementById('failureProb').style.color = fp > 0.5 ? '#ff1744' : fp > 0.2 ? '#ffab00' : '#00e676';
        document.getElementById('failureMarker').style.left = Math.min(100, fp * 100) + '%';

        // Confidence
        const conf = d.confidence || 0;
        const confEl = document.getElementById('confidenceValue');
        confEl.textContent = conf.toFixed(2);
        confEl.style.color = conf > 0.7 ? '#ff1744' : conf > 0.4 ? '#ffab00' : '#00e676';

        // Model status pill
        const pill = document.getElementById('modelStatusPill');
        const ms = d.model_status || 'active';
        if (ms === 'fallback_mode') {
            pill.className = 'status-pill fallback';
            pill.textContent = '⚠ Fallback';
        } else {
            pill.className = 'status-pill active';
            pill.textContent = '● ML Active';
        }

        // Root cause
        document.getElementById('rootCauseText').textContent = d.root_cause || 'System nominal';
    } catch (e) { console.error('pollHealth', e); }
}

async function pollMetrics() {
    try {
        const sid = getServerId();
        const res = await fetch(`${API}/metrics/recent?limit=60&server_id=${sid}`);
        const data = await res.json();
        if (!data.length) return;

        const rows = data.reverse();
        const labels = rows.map((_, i) => i);

        const latest = rows[rows.length - 1];
        document.getElementById('kpiCpu').textContent = latest.cpu.toFixed(1) + '%';
        document.getElementById('kpiMem').textContent = latest.memory.toFixed(1) + '%';
        document.getElementById('kpiDisk').textContent = latest.disk_io.toFixed(1) + ' MB/s';
        document.getElementById('kpiResp').textContent = latest.response_time.toFixed(0) + ' ms';
        document.getElementById('kpiNet').textContent = latest.network.toFixed(1) + ' KB/s';

        cpuMemChart.data.labels = labels;
        cpuMemChart.data.datasets[0].data = rows.map(r => r.cpu);
        cpuMemChart.data.datasets[1].data = rows.map(r => r.memory);
        cpuMemChart.update('none');

        diskRespChart.data.labels = labels;
        diskRespChart.data.datasets[0].data = rows.map(r => r.disk_io);
        diskRespChart.data.datasets[1].data = rows.map(r => r.response_time);
        diskRespChart.update('none');

        netChart.data.labels = labels;
        netChart.data.datasets[0].data = rows.map(r => r.network);
        netChart.update('none');
    } catch (e) { console.error('pollMetrics', e); }
}

async function pollHealthHistory() {
    try {
        const sid = getServerId();
        const res = await fetch(`${API}/health/history?limit=60&server_id=${sid}`);
        const data = await res.json();
        if (!data.length) return;

        const rows = data.reverse();
        const labels = rows.map((_, i) => i);

        const tl = document.getElementById('timeline');
        tl.innerHTML = rows.map(r =>
            `<span class="timeline-dot ${r.anomaly_flag ? 'anomaly' : 'normal'}"></span>`
        ).join('');

        healthChart.data.labels = labels;
        healthChart.data.datasets[0].data = rows.map(r => r.health_score);

        try {
            const fRes = await fetch(`${API}/health/forecast?server_id=${sid}`);
            const forecast = await fRes.json();
            if (forecast.forecast && forecast.forecast.length > 0) {
                const forecastLabels = [...labels];
                const forecastData = new Array(rows.length).fill(null);
                forecastData[rows.length - 1] = rows[rows.length - 1].health_score;
                for (let i = 0; i < forecast.forecast.length; i++) {
                    forecastLabels.push(labels.length + i);
                    forecastData.push(forecast.forecast[i]);
                }
                healthChart.data.labels = forecastLabels;
                healthChart.data.datasets[0].data = [...rows.map(r => r.health_score), ...new Array(forecast.forecast.length).fill(null)];
                healthChart.data.datasets[1].data = forecastData;
            }
        } catch (e) { /* forecast optional */ }

        healthChart.update('none');
    } catch (e) { console.error('pollHistory', e); }
}

async function pollAlerts() {
    try {
        const sid = getServerId();
        const res = await fetch(`${API}/alerts?limit=30&server_id=${sid}`);
        const data = await res.json();
        const tbody = document.querySelector('#alertTable tbody');
        tbody.innerHTML = data.map(a => {
            const sevColor = a.severity === 'CRITICAL' ? '#ff1744' : '#ffab00';
            const time = a.timestamp ? new Date(a.timestamp).toLocaleTimeString() : '—';
            return `<tr><td>${time}</td><td style="color:${sevColor};font-weight:700">${a.severity}</td><td>${a.message}</td></tr>`;
        }).join('');
    } catch (e) { console.error('pollAlerts', e); }
}

async function pollShap() {
    try {
        const res = await fetch(`${API}/explain`);
        const data = await res.json();
        const contribs = data.contributions || [];
        if (!contribs.length) return;

        shapChart.data.labels = contribs.map(c => c.name.replace(/_/g, ' '));
        shapChart.data.datasets[0].data = contribs.map(c => c.contribution);
        shapChart.data.datasets[0].backgroundColor = contribs.map(c =>
            c.contribution > 0 ? 'rgba(255,23,68,0.7)' : 'rgba(0,230,118,0.7)'
        );
        shapChart.update('none');
    } catch (e) { console.error('pollShap', e); }
}

async function pollServers() {
    try {
        const res = await fetch(`${API}/servers`);
        const data = await res.json();
        const sel = document.getElementById('serverSelect');
        const current = sel.value;
        sel.innerHTML = (data.servers || ['local']).map(s =>
            `<option value="${s}" ${s === current ? 'selected' : ''}>${s === 'local' ? '📍 Local Server' : '🖥 ' + s}</option>`
        ).join('');
    } catch (e) { /* optional */ }
}

// ── Enterprise polling ──────────────────────────────────────
async function pollDrift() {
    try {
        const res = await fetch(`${API}/model/drift-status`);
        const d = await res.json();

        const pill = document.getElementById('driftPill');
        if (d.drift_detected) {
            pill.style.display = 'inline-flex';
            pill.className = 'status-pill drift';
            pill.textContent = '⚠ DRIFT';
        } else {
            pill.style.display = 'none';
        }

        document.getElementById('entDriftStatus').textContent = d.drift_detected ? '⚠ DETECTED' : '✅ Normal';
        document.getElementById('entDriftStatus').style.color = d.drift_detected ? '#ff1744' : '#00e676';
        document.getElementById('entAnomalyRate').textContent = (d.anomaly_rate * 100).toFixed(1) + '%';
        document.getElementById('entAnomalyRate').style.color = d.anomaly_rate > 0.06 ? '#ff1744' : '#00e676';
    } catch (e) { /* optional */ }
}

async function pollEnterprise() {
    try {
        // Performance stats (may require API key — fails gracefully)
        const perfRes = await fetch(`${API}/system/performance`);
        if (perfRes.ok) {
            const p = await perfRes.json();
            document.getElementById('entLatency').textContent = p.avg_inference_latency_ms.toFixed(1) + ' ms';
            document.getElementById('entReqMin').textContent = p.requests_per_minute;
            document.getElementById('entMemory').textContent = p.app_memory_mb.toFixed(0) + ' MB';
        }
    } catch (e) { /* optional */ }

    try {
        // Model info (may require API key)
        const infoRes = await fetch(`${API}/model/info`);
        if (infoRes.ok) {
            const m = await infoRes.json();
            document.getElementById('entModelStatus').textContent = m.model_status === 'active' ? '✅ Active' : '⚠ Fallback';
            document.getElementById('entModelStatus').style.color = m.model_status === 'active' ? '#00e676' : '#ffab00';
            document.getElementById('entModelVer').textContent = m.model_version || 'v1.0';
        }
    } catch (e) { /* optional */ }
}

// ── Load per-server sensitivity ─────────────────────────────
async function loadSensitivity() {
    try {
        const res = await fetch(`${API}/settings?server_id=${getServerId()}`);
        const d = await res.json();
        slider.value = d.sensitivity;
        sensVal.textContent = d.sensitivity;
    } catch (e) { /* optional */ }
}

// ── Init ────────────────────────────────────────────────────
async function pollAll() {
    await Promise.all([pollHealth(), pollMetrics(), pollHealthHistory(), pollAlerts(), pollShap(), pollDrift()]);
}

loadSensitivity();
pollServers();
pollAll();
pollEnterprise();
setInterval(pollAll, POLL_MS);
setInterval(pollEnterprise, 10000);
setInterval(pollServers, 15000);

// Reload sensitivity when server changes
document.getElementById('serverSelect').addEventListener('change', () => {
    loadSensitivity();
    pollAll();
});

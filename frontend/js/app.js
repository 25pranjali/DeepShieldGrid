// ===========================================================================
// GridSentry frontend logic.
// Talks to the Flask backend running at API_BASE. If the backend isn't
// running, every function below shows a clear "can't connect" message
// instead of leaving blank/undefined values on the page.
// ===========================================================================

const API_BASE = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? `http://${window.location.hostname}:5000`
  : "http://127.0.0.1:5000";

let currentRole = null;
let currentIncidentId = null;
let dashboardPollTimer = null;

// Keys used to persist the session in localStorage so it survives page
// reloads (a full refresh, Live Server's auto-reload on file change, etc).
// This is a demo token, not a real auth scheme -- don't treat it as secure.
const TOKEN_KEY = "gridsentry_token";
const ROLE_KEY = "gridsentry_role";

// ---------------------------------------------------------------------------
// Session restore — runs once when app.js loads (i.e. on every page load).
// If we find a saved token, skip the login screen and go straight to the
// dashboard instead of forcing the user to sign in again.
// ---------------------------------------------------------------------------
function restoreSession() {
  const token = localStorage.getItem(TOKEN_KEY);
  const role = localStorage.getItem(ROLE_KEY);
  if (token && role) {
    enterApp(role);
  }
}

// ---------------------------------------------------------------------------
// Shared "show the app UI" logic, used by both a fresh login and a
// restored session on page load.
// ---------------------------------------------------------------------------
function enterApp(role) {
  currentRole = role;
  document.getElementById("login").style.display = "none";
  document.getElementById("app").classList.add("visible");
  document.getElementById("rolePill").textContent =
    "Role: " + (currentRole === "admin" ? "Admin" : "User");

  // What-If Simulator is admin-only
  document.getElementById("navWhatIf").hidden = currentRole !== "admin";

  syncDomainDropdown();

  if (dashboardPollTimer) clearInterval(dashboardPollTimer);
  try {
    updateDashboard();
    dashboardPollTimer = setInterval(updateDashboard, 2000);
  } catch (e) {
    console.warn("Dashboard update warning:", e);
  }
}

// ---------------------------------------------------------------------------
// Login / logout
// ---------------------------------------------------------------------------
async function loginUser() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const errorBox = document.getElementById("loginError");
  errorBox.classList.remove("visible");

  if (!username || !password) {
    errorBox.textContent = "Enter a username and password.";
    errorBox.classList.add("visible");
    return;
  }

  let data;
  try {
    const res = await fetch(`${API_BASE}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    data = await res.json();

    if (!res.ok || !data.success) {
      errorBox.textContent = data.error || "Invalid username or password.";
      errorBox.classList.add("visible");
      return;
    }
  } catch (err) {
    console.error("Login fetch error:", err);
    errorBox.textContent = "Unable to connect to the GridSentry backend. Is server.py running on port 5000?";
    errorBox.classList.add("visible");
    return;
  }

  // Persist the session so a reload (manual refresh, Live Server
  // auto-reload, etc.) doesn't bounce the user back to the sign-in screen.
  localStorage.setItem(TOKEN_KEY, data.token);
  localStorage.setItem(ROLE_KEY, data.role);

  enterApp(data.role);
}

function logout() {
  currentRole = null;
  currentIncidentId = null;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ROLE_KEY);
  if (dashboardPollTimer) clearInterval(dashboardPollTimer);
  document.getElementById("app").classList.remove("visible");
  document.getElementById("login").style.display = "flex";
}

// ---------------------------------------------------------------------------
// Page navigation
// ---------------------------------------------------------------------------
function showPage(id, btn) {
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  document.getElementById("page-" + id).classList.add("active");
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  btn.classList.add("active");
}

function goToShapForCurrentIncident() {
  if (currentIncidentId == null) return;
  loadShapExplanation(currentIncidentId);
  const btn = Array.from(document.querySelectorAll(".nav-item")).find(
    (n) => n.textContent.includes("Explanation")
  );
  showPage("shap", btn);
}

function downloadIncidentReport() {
  if (currentIncidentId == null) {
    alert("Select an incident before downloading its report.");
    return;
  }

  window.open(`${API_BASE}/api/incidents/${currentIncidentId}/report`, "_blank");
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
async function updateDashboard() {
  await Promise.all([loadLatestReading(), loadIncidents(), loadSimulationStatus(), loadIncidentDropdowns()]);
}

async function handleRefreshClick() {
  const btn = document.getElementById("btnRefresh");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "↻ Refreshing…";
  }
  try {
    await updateDashboard();
    if (btn) {
      btn.textContent = "✓ Refreshed";
      setTimeout(() => {
        btn.textContent = "Refresh now";
        btn.disabled = false;
      }, 700);
    }
  } catch (err) {
    if (btn) {
      btn.textContent = "Refresh now";
      btn.disabled = false;
    }
  }
}

let currentLogFilter = "all";
let cachedIncidentsList = [];

function updateDomainHeaderInfo(domainKey) {
  const subTitleEl = document.getElementById("dashSubTitle");
  const scopeNoteEl = document.getElementById("dashScopeNote");
  if (!subTitleEl || !scopeNoteEl) return;

  const infoMap = {
    "FDI_TSA": {
      sub: "Simulated real-time PMU feed (IEEE C37.118), replayed from dataset",
      scope: "Classes: FDI · Normal · TSA"
    },
    "IEC61850": {
      sub: "Substation Process Bus GOOSE/SV feed (IEC 61850), replayed from dataset",
      scope: "Classes: Normal · Replay · Fault · Injection · Masquerade"
    },
    "IEC104": {
      sub: "SCADA Telecontrol Protocol feed (IEC 60870-5-104), replayed from dataset",
      scope: "Classes: Normal · Command Injection · Telemetry Spoofing"
    },
    "MSU_ORNL": {
      sub: "Power Transmission Protection feed (MSU / ORNL), replayed from dataset",
      scope: "Classes: Natural · Line Fault · Trip Maintenance"
    },
    "UPLOADED": {
      sub: "Custom simulation feed from uploaded dataset",
      scope: "Classes: Dynamic Model Classification"
    }
  };

  const info = infoMap[domainKey] || infoMap["FDI_TSA"];
  subTitleEl.textContent = info.sub;
  scopeNoteEl.textContent = info.scope;
}

async function syncDomainDropdown() {
  try {
    const res = await fetch(`${API_BASE}/api/simulation/domains`);
    if (!res.ok) return;
    const data = await res.json();
    const sel = document.getElementById("simDomainSelect");
    if (!sel) return;

    const currentVal = data.active_domain || sel.value;
    const optionsHtml = (data.available_domains || []).map((d) => `
      <option value="${d.id}" ${d.id === currentVal ? "selected" : ""}>${d.name}</option>
    `).join("");
    sel.innerHTML = optionsHtml;
    sel.value = currentVal;
    updateDomainHeaderInfo(currentVal);
  } catch (e) {
    console.warn("syncDomainDropdown error:", e);
  }
}

async function switchSimulationDomain(domainId) {
  try {
    const res = await fetch(`${API_BASE}/api/simulation/domain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain: domainId }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.latest_reading) {
        renderTelemetryCards(data.latest_reading);
      }
      updateDomainHeaderInfo(domainId);
      await updateDashboard();
    }
  } catch (err) {
    console.error("Failed to switch domain:", err);
  }
}

function fmt(value, decimals = 4) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toFixed(decimals);
}

function renderTelemetryCards(data) {
  const gridEl = document.getElementById("telemetryGrid");
  if (!gridEl || !data) return;

  let items = [];
  if (Array.isArray(data.display_features) && data.display_features.length > 0) {
    items = data.display_features;
  } else if (data.features && typeof data.features === "object" && Object.keys(data.features).length > 0) {
    items = Object.entries(data.features).slice(0, 4).map(([k, v]) => ({
      name: k,
      value: typeof v === "number" ? v.toFixed(4) : String(v),
      unit: ""
    }));
  }

  if (items.length === 0) return;

  const colCount = Math.min(Math.max(items.length, 1), 4);
  gridEl.style.display = "grid";
  gridEl.style.gridTemplateColumns = `repeat(${colCount}, 1fr)`;
  gridEl.style.gap = "14px";
  gridEl.style.marginBottom = "22px";

  gridEl.innerHTML = items.map((item, idx) => {
    const valText = (item.value !== null && item.value !== undefined && item.value !== "") ? item.value : "—";
    const unitHtml = item.unit ? `<span class="unit"> ${item.unit}</span>` : "";
    return `
      <div class="card" id="cardFeat_${idx}">
        <h3 id="featTitle${idx}" title="${item.name}">${item.name}</h3>
        <div class="readout" id="readFeat_${idx}">${valText}${unitHtml}</div>
        <div class="delta" style="font-size:11.5px; color:var(--text-dim); margin-top:4px;">Live Sensor Telemetry</div>
      </div>
    `;
  }).join("");
}

async function loadLatestReading() {
  const errorBox = document.getElementById("dashboardError");
  try {
    const res = await fetch(`${API_BASE}/api/readings/latest`);
    const data = await res.json();

    if (!res.ok) {
      return;
    }
    errorBox.innerHTML = "";

    // Dynamic Display Features (Adapts automatically to active dataset/domain)
    renderTelemetryCards(data);

    document.getElementById("readingTimestamp").textContent = data.timestamp || "—";
    document.getElementById("predClass").textContent = data.prediction || "—";
    document.getElementById("predConfidence").textContent =
      data.confidence != null ? data.confidence + "%" : "—";
    document.getElementById("predGroundTruth").textContent = data.ground_truth_label || "—";
    
    const predDomainEl = document.getElementById("predDomain");
    if (predDomainEl) {
      predDomainEl.textContent = data.domain || data.source || "PMU Synchrophasor (IEEE C37.118)";
    }

    const riskScoreEl = document.getElementById("riskScoreNum");
    const riskLevelEl = document.getElementById("riskLevelText");
    riskScoreEl.textContent = data.risk_score != null ? data.risk_score : "—";
    riskLevelEl.textContent = data.risk_level || "—";
    riskLevelEl.className = "risk-level " + (data.risk_level ? data.risk_level.toLowerCase() : "normal");

    const banner = document.getElementById("statusBanner");
    const isAttack = data.prediction && data.prediction !== "Normal" && data.prediction !== "Natural";
    banner.className = "status-banner " + (isAttack ? "attack" : "normal");
    document.getElementById("bannerTitle").textContent = isAttack
      ? `${data.prediction} detected on active telemetry feed`
      : (data.prediction === "Normal" || data.prediction === "Natural" ? "System operating normally" : "Waiting for data…");
    document.getElementById("bannerSub").textContent = isAttack
      ? `Confidence ${data.confidence}% · Risk: ${data.risk_level} (${data.risk_score}/100)`
      : `Nominal baseline telemetry · Domain: ${data.domain || 'PMU Synchrophasor'}`;
  } catch (err) {
    errorBox.innerHTML = `<div class="error-msg">Unable to connect to GridSentry backend. Make sure server.py is running on ${API_BASE}.</div>`;
  }
}

function setLogFilter(filterName) {
  currentLogFilter = filterName;
  document.querySelectorAll(".pill-btn").forEach(btn => btn.classList.remove("active"));
  const btnId = "filter" + filterName.charAt(0).toUpperCase() + filterName.slice(1);
  const activeBtn = document.getElementById(btnId);
  if (activeBtn) activeBtn.classList.add("active");
  renderIncidentsTable(cachedIncidentsList);
}

function renderIncidentsTable(incidents) {
  const tbody = document.getElementById("incidentsTableBody");
  if (!tbody) return;

  if (!Array.isArray(incidents) || incidents.length === 0) {
    tbody.innerHTML =
      '<tr class="empty-row"><td colspan="5">No telemetry records yet</td></tr>';
    return;
  }

  let filtered = incidents;

  // Attacks Only
  if (currentLogFilter === "attacks") {
    filtered = incidents.filter(
      i =>
        i.attack_type !== "Normal" &&
        i.attack_type !== "Natural"
    );
  }

  // Normal Baseline
  else if (currentLogFilter === "normal") {
    filtered = incidents.filter(
      i =>
        i.attack_type === "Normal" ||
        i.attack_type === "Natural"
    );
  }

  if (filtered.length === 0) {
    tbody.innerHTML =
      `<tr class="empty-row">
        <td colspan="5">No ${currentLogFilter} records in current feed</td>
       </tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((inc) => {
      const isNormal =
        inc.attack_type === "Normal" ||
        inc.attack_type === "Natural";

      let riskLevel = inc.risk_level || "—";

      // Keep Normal only for normal telemetry
      if (isNormal) {
        riskLevel = "Normal";
      }

      // CSS class for Low / Medium / Critical
      let badgeClass = "normal";

      if (!isNormal) {
        badgeClass = riskLevel.toLowerCase();
      }

      const classDisplay = isNormal
        ? `<span style="color:var(--green); font-weight:600;">
             ✓ Normal
           </span>`
        : `<span style="color:var(--text); font-weight:600;">
             ${inc.attack_type}
           </span>`;

      return `
        <tr class="clickable" onclick="openIncident(${inc.id})">
          <td class="mono">
            ${(inc.timestamp || "")
              .slice(0, 19)
              .replace("T", " ")}
          </td>

          <td>${inc.source || "—"}</td>

          <td>${classDisplay}</td>

          <td>
            ${inc.confidence != null
              ? inc.confidence + "%"
              : "—"}
          </td>

          <td>
            <span class="badge ${badgeClass}">
              ${riskLevel}
            </span>
          </td>
        </tr>
      `;
    })
    .join("");
}

async function loadIncidents() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    if (!res.ok) return;
    const incidents = await res.json();
    cachedIncidentsList = incidents;
    renderIncidentsTable(cachedIncidentsList);
  } catch (err) {
    // Errors here are surfaced by loadLatestReading's connection check already.
  }
}

async function loadSimulationStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/simulation/status`);
    const data = await res.json();
    const dot = document.getElementById("simDot");
    const text = document.getElementById("simStatusText");
    if (data.running) {
      dot.classList.add("on");
      text.textContent = `Simulation running — row ${data.current_index} / ${data.dataset_length}`;
    } else {
      dot.classList.remove("on");
      text.textContent = "Simulation stopped";
    }
  } catch (err) {
    // handled elsewhere
  }
}

async function startSimulation() {
  try {
    await fetch(`${API_BASE}/api/simulation/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interval_seconds: 2 }),
    });
    updateDashboard();
  } catch (err) {
    document.getElementById(
      "dashboardError"
    ).innerHTML = `<div class="error-msg">Unable to connect to GridSentry backend.</div>`;
  }
}

async function stopSimulation() {
  try {
    await fetch(`${API_BASE}/api/simulation/stop`, { method: "POST" });
    updateDashboard();
  } catch (err) {
    // ignore, dashboard poll will surface connection errors
  }
}

async function uploadSimulationDataset(event) {
  const file = event.target.files[0];
  if (!file) return;

  const errorBox = document.getElementById("dashboardError");
  errorBox.innerHTML = `<div style="background:var(--panel-2); border-left:3px solid var(--cyan); padding:10px 14px; margin-bottom:14px; border-radius:4px; font-size:13px; color:var(--text);">Analyzing uploaded dataset <strong>${file.name}</strong> and matching compatible models...</div>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/datasets/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (res.status === 422 || data.status === "rejected") {
      errorBox.innerHTML = `<div class="error-msg" style="margin-bottom:14px;"><strong>Upload Incompatible:</strong> ${data.message}</div>`;
      event.target.value = "";
      return;
    }

    if (!res.ok || data.status === "error") {
      errorBox.innerHTML = `<div class="error-msg" style="margin-bottom:14px;"><strong>Upload Error:</strong> ${data.error || data.message || "Failed to process dataset."}</div>`;
      event.target.value = "";
      return;
    }

    const isAttack = data.prediction && data.prediction !== "Normal" && data.prediction !== "Natural";
    const statusClass = isAttack ? "attack" : "normal";
    errorBox.innerHTML = `
      <div style="background:var(--panel); border:1px solid ${isAttack ? 'var(--red)' : 'var(--cyan)'}; padding:14px; border-radius:6px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <strong style="color:var(--text); font-size:14px;">Dataset Analysis & Attack Detection Complete</strong>
          <span class="badge ${statusClass}">${data.prediction}</span>
        </div>
        <div style="font-size:12.5px; color:var(--text-dim); line-height:1.6;">
          • Matched Model: <strong style="color:var(--text);">${data.selected_model}</strong> (${data.domain})<br>
          • Model Confidence: <strong style="color:var(--text);">${data.confidence}%</strong> · Risk Level: <strong style="color:var(--text);">${data.risk_level}</strong> (${data.risk_score}/100)<br>
          ${data.is_unknown_attack ? '<span style="color:var(--red);">• Zero-Day / Unknown Anomaly detected via Autoencoder reconstruction error!</span><br>' : ''}
          • Features evaluated: <code>${(data.features_used || []).join(", ")}</code>
        </div>
        ${data.incident_id ? `<div style="margin-top:10px;"><button class="btn-secondary" style="padding:6px 12px; font-size:12px;" onclick="openIncident(${data.incident_id})">View Incident Details & SHAP →</button></div>` : ''}
      </div>
    `;

    event.target.value = "";
    // Immediately show the uploaded dataset values
    if (data.display_features) {
      renderTelemetryCards(data);
    }

    // Refresh dashboard using the newly uploaded dataset
    await syncDomainDropdown();
    await updateDashboard();
    await loadIncidents();
    await loadIncidentDropdowns();

    // Make sure the uploaded dataset is selected
    const domainSelect = document.getElementById("simDomainSelect");
    if (domainSelect) {
      domainSelect.value = "UPLOADED";
    }
  } catch (err) {
    errorBox.innerHTML = `<div class="error-msg" style="margin-bottom:14px;">Unable to upload dataset. Ensure backend server is running.</div>`;
    event.target.value = "";
  }
}

// ---------------------------------------------------------------------------
// Incident dropdown selectors (Attack Detection + SHAP pages)
// ---------------------------------------------------------------------------
async function loadIncidentDropdowns() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    const incidents = await res.json();
    if (!Array.isArray(incidents)) return;

    const selDet = document.getElementById("detectionIncidentSelect");
    const selShap = document.getElementById("shapIncidentSelect");
    if (!selDet || !selShap) return;

    const makeOptions = (list) => {
      let html = '<option value="" disabled selected>Select incident…</option>';
      list.forEach((inc) => {
        html += `<option value="${inc.id}">#${inc.id} – ${inc.attack_type} (${inc.risk_level})</option>`;
      });
      return html;
    };

    selDet.innerHTML = makeOptions(incidents);
    selShap.innerHTML = makeOptions(incidents);
  } catch (err) {
    // dropdowns stay as-is
  }
}

function onSelectDetectionIncident(val) {
  if (!val) return;
  openIncident(Number(val));
}

function onSelectShapIncident(val) {
  if (!val) return;
  loadShapExplanation(Number(val));
}

// ---------------------------------------------------------------------------
// Attack Detection page
// ---------------------------------------------------------------------------
async function openIncident(id) {
  currentIncidentId = id;
  await loadIncidentDetails(id);
  const btn = Array.from(document.querySelectorAll(".nav-item")).find((n) =>
    n.textContent.includes("Attack Detection")
  );
  showPage("detection", btn);
}

async function loadIncidentDetails(id) {
  try {
    const res = await fetch(`${API_BASE}/api/incidents/${id}`);
    const inc = await res.json();

    if (!res.ok) {
      document.getElementById("detectionEmpty").textContent = inc.error || "Incident not found.";
      document.getElementById("detectionEmpty").style.display = "block";
      document.getElementById("detectionContent").style.display = "none";
      return;
    }

    document.getElementById("detectionEmpty").style.display = "none";
    document.getElementById("detectionContent").style.display = "block";
    document.getElementById("detectionSubtitle").textContent = `Incident #${inc.id}`;

    document.getElementById("detIncidentId").textContent = inc.id;
    document.getElementById("detTimestamp").textContent = inc.timestamp || "—";
    document.getElementById("detSource").textContent = inc.source || "—";
    document.getElementById("detManipulated").textContent = inc.manipulated_fields || "Not available";
    document.getElementById("detAttackType").textContent = inc.attack_type;
    document.getElementById("detConfidence").textContent = inc.confidence + "%";
    document.getElementById("detRiskLevel").textContent = inc.risk_level;

    document.getElementById("detGroundTruth").textContent = inc.ground_truth_label || "Not available";
    document.getElementById("detPrediction").textContent = inc.attack_type;
    document.getElementById("detStatus").textContent = inc.prediction_status;

    const gtBox = document.getElementById("gtBox");
    const predBox = document.getElementById("predBox");
    gtBox.className = "compare-box";
    predBox.className = "compare-box";
    if (inc.prediction_status === "Correct") {
      gtBox.classList.add("match");
      predBox.classList.add("match");
    } else if (inc.prediction_status === "Incorrect") {
      gtBox.classList.add("mismatch");
      predBox.classList.add("mismatch");
    }
  } catch (err) {
    document.getElementById("detectionEmpty").textContent =
      "Unable to connect to GridSentry backend.";
    document.getElementById("detectionEmpty").style.display = "block";
    document.getElementById("detectionContent").style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// SHAP page
// ---------------------------------------------------------------------------
async function loadShapExplanation(id) {
  try {
    const res = await fetch(`${API_BASE}/api/incidents/${id}/shap`);
    const data = await res.json();

    if (!res.ok) {
      document.getElementById("shapEmpty").textContent = data.error || "No SHAP data available.";
      document.getElementById("shapEmpty").style.display = "block";
      document.getElementById("shapContent").style.display = "none";
      return;
    }

    const values = data.shap_values || [];
    if (values.length === 0) {
      document.getElementById("shapEmpty").textContent =
        "No SHAP values were stored for this incident (SHAP may not be installed on the backend).";
      document.getElementById("shapEmpty").style.display = "block";
      document.getElementById("shapContent").style.display = "none";
      return;
    }

    document.getElementById("shapEmpty").style.display = "none";
    document.getElementById("shapContent").style.display = "block";
    document.getElementById("shapSubtitle").textContent =
      `SHAP feature contributions for Incident #${data.incident_id} (${data.attack_type})`;

    const rankedValues = [...values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const maxAbs = Math.max(...rankedValues.map((v) => Math.abs(v.value)), 0.0001);
    const strongest = rankedValues[0];
    const summaryHtml = `
      <div class="shap-summary-card">
        <div class="shap-metric-label">Strongest driver</div>
        <div class="shap-metric-value ${strongest.value >= 0 ? "pos" : "neg"}">${strongest.feature}</div>
      </div>
      <div class="shap-summary-card">
        <div class="shap-metric-label">Contribution</div>
        <div class="shap-metric-value ${strongest.value >= 0 ? "pos" : "neg"}">${strongest.value >= 0 ? "+" : ""}${strongest.value.toFixed(3)}</div>
      </div>`;
    document.getElementById("shapSummary").innerHTML = summaryHtml;

    const rowsHtml = rankedValues
      .map((v) => {
        const pct = Math.max(8, (Math.abs(v.value) / maxAbs) * 100);
        const cls = v.value >= 0 ? "pos" : "neg";
        const color = v.value >= 0 ? "var(--red)" : "var(--cyan)";
        const sign = v.value >= 0 ? "+" : "";
        return `
          <div class="shap-row">
            <div class="name">${v.feature}</div>
            <div class="shap-bar-track">
              <div class="shap-bar-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <div class="shap-val" style="color:${color}">${sign}${v.value.toFixed(3)}</div>
          </div>`;
      })
      .join("");
    document.getElementById("shapRows").innerHTML = rowsHtml;
  } catch (err) {
    document.getElementById("shapEmpty").textContent = "Unable to connect to GridSentry backend.";
    document.getElementById("shapEmpty").style.display = "block";
    document.getElementById("shapContent").style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// What-If Simulator
// ---------------------------------------------------------------------------
async function runWhatIf() {
  const errorBox = document.getElementById("wiError");
  errorBox.style.display = "none";

  const inputs = {
    "Actual frequency value": document.getElementById("wiFrequency").value,
    "Fraction of second": document.getElementById("wiFraction").value,
    "Time synchronized": document.getElementById("wiTimeSync").value,
    "interarrival time": document.getElementById("wiInterarrival").value,
    "time difference": document.getElementById("wiTimeDiff").value,
  };

  for (const [key, val] of Object.entries(inputs)) {
    if (val === "" || val === null || Number.isNaN(Number(val))) {
      errorBox.textContent = `Enter a valid number for "${key}".`;
      errorBox.style.display = "block";
      return;
    }
  }

  const payload = {};
  for (const [key, val] of Object.entries(inputs)) payload[key] = Number(val);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errorBox.textContent = data.error || "Prediction failed.";
      errorBox.style.display = "block";
      return;
    }

    document.getElementById("wiClass").textContent = data.prediction;
    document.getElementById("wiConfidence").textContent = data.confidence + "%";
    document.getElementById("wiRiskScore").textContent = data.risk_score;
    const levelEl = document.getElementById("wiRiskLevel");
    levelEl.textContent = data.risk_level;
    levelEl.className = "risk-level " + data.risk_level.toLowerCase();

    document.getElementById("wiResult").classList.add("visible");

    if (data.incident_id) {
      await loadIncidents();
    }
  } catch (err) {
    errorBox.textContent = "Unable to connect to GridSentry backend.";
    errorBox.style.display = "block";
  }
}

// Try to restore an existing session as soon as this script runs (it's
// loaded at the end of <body>, so the DOM is already parsed at this point).
restoreSession();
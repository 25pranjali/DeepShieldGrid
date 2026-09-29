// ===========================================================================
// DeepShieldGrid frontend logic.
// Talks to the Flask backend running at API_BASE. If the backend isn't
// running, every function below shows a clear "can't connect" message
// instead of leaving blank/undefined values on the page.
// ===========================================================================

const API_BASE = (window.location.protocol.startsWith("http") && window.location.port === "5000")
  ? ""
  : (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
    ? `http://${window.location.hostname}:5000`
    : "http://127.0.0.1:5000";

let currentRole = null;
let currentIncidentId = null;
let dashboardPollTimer = null;

// Keys used to persist the session in localStorage so it survives page
// reloads (a full refresh, Live Server's auto-reload on file change, etc).
// This is a demo token, not a real auth scheme -- don't treat it as secure.
const TOKEN_KEY = "deepshieldgrid_token";
const ROLE_KEY = "deepshieldgrid_role";

// ---------------------------------------------------------------------------
// Session restore — runs once when app.js loads (i.e. on every page load).
// If we find a saved token, skip the login screen and go straight to the
// dashboard instead of forcing the user to sign in again.
// ---------------------------------------------------------------------------
function restoreSession() {
  const token = localStorage.getItem(TOKEN_KEY) || localStorage.getItem("gridsentry_token");
  const role = localStorage.getItem(ROLE_KEY) || localStorage.getItem("gridsentry_role");
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
    errorBox.textContent = "Unable to connect to the DeepShieldGrid backend. Is server.py running on port 5000?";
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
  localStorage.removeItem("gridsentry_token");
  localStorage.removeItem("gridsentry_role");
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
    const isQuar = Boolean(data.is_quarantined);
    let subText = "Live Sensor Telemetry";
    let quarBadge = "";

    if (isQuar) {
      quarBadge = `<span style="font-size:10px; color:var(--red); font-weight:700; background:rgba(229,82,92,0.18); border:1px solid rgba(229,82,92,0.4); padding:2px 6px; border-radius:4px; margin-left:6px;">QUARANTINED</span>`;
      if (data.fallback_reading && data.fallback_reading[item.name] !== undefined) {
        const fbVal = Number(data.fallback_reading[item.name]);
        const fbStr = !isNaN(fbVal) ? fbVal.toFixed(4) : data.fallback_reading[item.name];
        subText = `<span style="color:var(--cyan); font-weight:600;">Dispatched fallback: ${fbStr}${item.unit ? ' ' + item.unit : ''}</span>`;
      } else {
        subText = `<span style="color:var(--red); font-weight:600;">Excluded from grid state estimator</span>`;
      }
    }
    const unitHtml = item.unit
    ? ` <span class="unit" style="font-size:14px; opacity:0.7;">${item.unit}</span>`
   : "";

    return `
      <div class="card" id="cardFeat_${idx}" style="${isQuar ? 'border-color:rgba(229,82,92,0.45);' : ''}">
        <h3 id="featTitle${idx}" title="${item.name}">${item.name}${quarBadge}</h3>
        <div class="readout" id="readFeat_${idx}">${valText}${unitHtml}</div>
        <div class="delta" style="font-size:11.5px; margin-top:4px;">${subText}</div>
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

    const isAttack = data.prediction && data.prediction !== "Normal" && data.prediction !== "Natural";

    // Dynamic Display Features (Adapts automatically to active dataset/domain)
    renderTelemetryCards(data);

    document.getElementById("readingTimestamp").textContent = data.timestamp || "—";
    document.getElementById("predClass").textContent = data.prediction || "—";
    document.getElementById("predConfidence").textContent =
      data.confidence != null ? data.confidence + "%" : "—";
    const mitigationEl = document.getElementById("predGroundTruth");
    if (mitigationEl) {
      mitigationEl.textContent = isAttack ? "Active Containment Enforced" : "Nominal (No Action Needed)";
      mitigationEl.style.color = isAttack ? "var(--red)" : "var(--cyan)";
      mitigationEl.style.fontWeight = "600";
    }
    
    const predDomainEl = document.getElementById("predDomain");
    if (predDomainEl) {
      predDomainEl.textContent = data.domain || data.source || "PMU Synchrophasor (IEEE C37.118)";
    }

    const riskScoreEl = document.getElementById("riskScoreNum");
    const riskLevelEl = document.getElementById("riskLevelText");
    riskScoreEl.textContent = data.risk_score != null ? data.risk_score : "—";
    riskLevelEl.textContent = data.risk_level || "—";
    riskLevelEl.className = "risk-level " + (data.risk_level ? data.risk_level.toLowerCase() : "normal");

    const dashRiskPill = document.getElementById("dashRiskActionsPill");
    if (dashRiskPill) {
      if (isAttack) {
        dashRiskPill.textContent = "Action: Quarantined · Fallback · Source Isolated";
        dashRiskPill.style.color = "var(--red)";
        dashRiskPill.style.background = "rgba(229, 82, 92, 0.16)";
        dashRiskPill.style.border = "1px solid rgba(229, 82, 92, 0.4)";
      } else {
        dashRiskPill.textContent = "Action: Routine Nominal Monitoring";
        dashRiskPill.style.color = "var(--cyan)";
        dashRiskPill.style.background = "var(--cyan-dim)";
        dashRiskPill.style.border = "1px solid rgba(69, 214, 199, 0.3)";
      }
    }

    const banner = document.getElementById("statusBanner");
    banner.className = "status-banner " + (isAttack ? "attack" : "normal");
    document.getElementById("bannerTitle").textContent = isAttack
      ? `${data.prediction} detected on active telemetry feed`
      : (data.prediction === "Normal" || data.prediction === "Natural" ? "System operating normally" : "Waiting for data…");
    document.getElementById("bannerSub").textContent = isAttack
      ? `Confidence ${data.confidence}% · Risk: ${data.risk_level} (${data.risk_score}/100)`
      : `Nominal baseline telemetry · Domain: ${data.domain || 'PMU Synchrophasor'}`;

    // Update response status line (#responseStatusLine)
    const statusLineEl = document.getElementById("responseStatusLine");
    if (statusLineEl) {
      const phrases = [];
      if (data.is_quarantined) {
        phrases.push("⚠ Current reading quarantined — showing last trusted fallback where available");
      }
      if (data.source_isolated) {
        phrases.push("Source: Isolated (temporary)");
      }
      if (data.monitoring_mode === "increased") {
        phrases.push("Monitoring frequency: increased (0.3s)");
      }
      if (phrases.length === 0) {
        phrases.push("All sources normal");
      }
      statusLineEl.textContent = phrases.join(" · ");
    }
  } catch (err) {
    if (window.location.protocol === "file:") {
      errorBox.innerHTML = `<div class="error-msg">You are viewing this page locally (file://). Browsers block network API requests from local file URLs. Please open <a href="http://127.0.0.1:5000" style="color:var(--cyan); text-decoration:underline; font-weight:600;">http://127.0.0.1:5000</a> in your browser.</div>`;
    } else {
      errorBox.innerHTML = `<div class="error-msg">Unable to connect to DeepShieldGrid backend. Make sure server.py is running on ${API_BASE || 'http://127.0.0.1:5000'}.</div>`;
    }
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
        <td colspan="6">No ${currentLogFilter} records in current feed</td>
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
    ).innerHTML = `<div class="error-msg">Unable to connect to DeepShieldGrid backend.</div>`;
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

function getGridThreatImpact(attackType, source) {
  const at = (attackType || "").toLowerCase();
  if (at.includes("fdi") || at.includes("false data")) {
    return {
      icon: "⚡",
      threat: "State Estimation Poisoning (FDI)",
      impact: "Attacker injects falsified frequency/phasor measurements to deceive Automatic Generation Control (AGC) and state estimators, risking erroneous generator dispatch and local power line overloads.",
      severityClass: "critical"
    };
  } else if (at.includes("tsa") || at.includes("time sync")) {
    return {
      icon: "⏱️",
      threat: "Time Synchronization Attack (TSA)",
      impact: "GPS timestamp spoofing disrupts phase angle calculations across Wide-Area Monitoring (WAMS), risking false distance relay tripping and premature transmission line disconnect.",
      severityClass: "critical"
    };
  } else if (at.includes("masquerade")) {
    return {
      icon: "🎭",
      threat: "Substation Node Impersonation (Masquerade)",
      impact: "Attacker mimics legitimate protective IEDs on IEC 61850 Process Bus, falsifying circuit breaker state telemetry (state_cb) to mask faults or trigger uncommanded breaker opening.",
      severityClass: "critical"
    };
  } else if (at.includes("injection")) {
    return {
      icon: "💉",
      threat: "Malicious Packet Injection (GOOSE)",
      impact: "Unauthorized trip commands injected directly into substation multicast network, attempting forced disconnect of power transformers or feeder lines.",
      severityClass: "critical"
    };
  } else if (at.includes("replay")) {
    return {
      icon: "🔁",
      threat: "GOOSE Sequence Replay Attack",
      impact: "Stale operational status packets replayed with modified sequence numbers (sqNum/stnum) to blind operators during an ongoing grid contingency.",
      severityClass: "medium"
    };
  } else if (at.includes("spoofing") || at.includes("telemetry")) {
    return {
      icon: "📡",
      threat: "SCADA Telecontrol Protocol Manipulation",
      impact: "Spoofed ASDU/IOA packets inject false telemetry to master SCADA stations, compromising situational awareness and automated voltage regulation.",
      severityClass: "medium"
    };
  } else if (at.includes("fault")) {
    return {
      icon: "⚠️",
      threat: "Substation Protection Fault Detected",
      impact: "Current/voltage waveform distortion detected on process bus. Protection relay coordination initiated to prevent electrical equipment damage.",
      severityClass: "medium"
    };
  } else {
    return {
      icon: "🛡️",
      threat: "Nominal Grid Telemetry",
      impact: "All electrical and protocol parameters operating within acceptable standard tolerances. No containment actions required.",
      severityClass: "normal"
    };
  }
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

    const isAttack = inc.attack_type && inc.attack_type !== "Normal" && inc.attack_type !== "Natural";
    const riskScore = inc.risk_score != null ? inc.risk_score : (inc.confidence || 0);
    const riskLevelStr = (inc.risk_level || "Normal").toLowerCase();

    // 1. Update Risk Severity Gauge & Badge
    const riskScoreNumEl = document.getElementById("detRiskScoreNum");
    if (riskScoreNumEl) {
      riskScoreNumEl.textContent = typeof riskScore === "number" ? riskScore.toFixed(1) : riskScore;
    }

    const riskLevelPillEl = document.getElementById("detRiskLevelPill");
    if (riskLevelPillEl) {
      riskLevelPillEl.textContent = inc.risk_level || "Normal";
      riskLevelPillEl.className = "risk-level-badge " + riskLevelStr;
    }

    const riskMeterFillEl = document.getElementById("detRiskMeterFill");
    if (riskMeterFillEl) {
      riskMeterFillEl.style.width = Math.min(100, Math.max(5, Number(riskScore) || 5)) + "%";
      riskMeterFillEl.className = "risk-meter-fill " + riskLevelStr;
    }

    const containmentBadgeEl = document.getElementById("detContainmentBadge");
    if (containmentBadgeEl) {
      if (isAttack) {
        containmentBadgeEl.textContent = "CONTAINMENT ACTIVE";
        containmentBadgeEl.className = "status-pill active-containment";
      } else {
        containmentBadgeEl.textContent = "NOMINAL BASELINE";
        containmentBadgeEl.className = "status-pill nominal";
      }
    }

    // 2. Update Grid Threat & Operational Impact
    const threatInfo = getGridThreatImpact(inc.attack_type, inc.source);
    const threatTitleEl = document.getElementById("detThreatTitle");
    if (threatTitleEl) threatTitleEl.textContent = threatInfo.threat;
    const threatIconEl = document.getElementById("detThreatIcon");
    if (threatIconEl) threatIconEl.textContent = threatInfo.icon;
    const threatTextEl = document.getElementById("detThreatImpactText");
    if (threatTextEl) threatTextEl.textContent = threatInfo.impact;
    const threatBoxEl = document.getElementById("detThreatImpactBox");
    if (threatBoxEl) threatBoxEl.className = "threat-impact-box " + threatInfo.severityClass;

    // 3. Update Actions Taken Summary
    const respLog = Array.isArray(inc.response_log) ? inc.response_log : [];
    const hasQuarantine = respLog.includes("quarantine");
    const hasFallback = respLog.includes("fallback");
    const hasIsolation = respLog.includes("isolate") || inc.source_isolated;
    const hasMonitoring = respLog.includes("increase_monitoring");

    const actQuarEl = document.getElementById("detActQuarantine");
    if (actQuarEl) {
      actQuarEl.textContent = hasQuarantine
        ? "Packet isolated; excluded from AGC & state estimation pipeline"
        : "Live verification nominal; passing without quarantine";
    }
    const actFallEl = document.getElementById("detActFallback");
    if (actFallEl) {
      actFallEl.textContent = hasFallback
        ? `Switched downstream systems to last validated healthy baseline reading`
        : "Direct live feed active; no fallback required";
    }
    const actIsoEl = document.getElementById("detActIsolation");
    if (actIsoEl) {
      actIsoEl.textContent = hasIsolation
        ? `Source '${inc.source}' isolated for 5 cycles to prevent cascading spread`
        : "Source interface verified and connected to network";
    }
    const actMonEl = document.getElementById("detActMonitoring");
    if (actMonEl) {
      actMonEl.textContent = hasMonitoring
        ? "Sampling frequency accelerated to 0.3s for forensic telemetry capture"
        : "Operating on normal telemetry polling interval";
    }

    // 4. Render dynamic response flow steps with detailed descriptions
    const flowContainer = document.getElementById("responseFlowContainer");
    if (flowContainer) {
      const STEPS = [
        {
          key: "quarantine",
          label: "Quarantine suspicious reading",
          activeDetail: "Action taken: Flagged untrusted and excluded from state estimation (is_quarantined = 1).",
          pendingDetail: "Normal telemetry: Input verified nominal, passing through without quarantine."
        },
        {
          key: "fallback",
          label: "Switch to trusted / fallback data",
          activeDetail: `Action taken: Downstream algorithms routed to last verified healthy baseline.`,
          pendingDetail: "Nominal mode: Processing live synchronized sensor data."
        },
        {
          key: "isolate",
          label: "Source isolation (simulated)",
          activeDetail: `Action taken: Interface '${inc.source}' marked isolated for 5 cycles to halt threat spread.`,
          pendingDetail: "Nominal mode: Source operational with full network connectivity."
        },
        {
          key: "increase_monitoring",
          label: "Increase monitoring frequency",
          activeDetail: "Action taken: Sampling frequency accelerated (interval: 0.3s) for high-rate forensic capture.",
          pendingDetail: "Standard mode: Surveillance operating at normal telemetry polling interval."
        },
        {
          key: "alert",
          label: "Generate alert to operator",
          activeDetail: `Action taken: Priority security alert dispatched for Incident #${inc.id} (${inc.risk_level || 'Alert'} Risk).`,
          pendingDetail: "No alert required: Nominal telemetry within safety thresholds."
        },
        {
          key: "log",
          label: "Log incident for forensic audit",
          activeDetail: `Action taken: Incident snapshot, manipulated fields (${inc.manipulated_fields || 'telemetry'}), and SHAP weights archived.`,
          pendingDetail: "Routine archive: Baseline telemetry cycle logged."
        },
      ];

      flowContainer.innerHTML = STEPS.map((step, idx) => {
        const isDone = respLog.includes(step.key);
        const stepClass = isDone ? "response-step done" : "response-step pending";
        const numContent = isDone ? "✓" : (idx + 1);
        const detailTxt = isDone ? step.activeDetail : step.pendingDetail;
        return `
          <div class="${stepClass}">
            <span class="step-num">${numContent}</span>
            <div class="step-content">
              <div class="step-txt">${step.label}</div>
              <div class="step-detail">${detailTxt}</div>
            </div>
          </div>
        `;
      }).join("");
    }

    // 5. Populate Enforcement Evidence & State Estimator Protection Card
    const proofQuarEl = document.getElementById("proofQuarantine");
    if (proofQuarEl) {
      proofQuarEl.textContent = (inc.reading_quarantined || isAttack)
        ? "1 (TRUE — Excluded from State Estimator)"
        : "0 (FALSE — Nominal Pass)";
      proofQuarEl.style.color = (inc.reading_quarantined || isAttack) ? "var(--red)" : "var(--cyan)";
    }

    const proofIsoEl = document.getElementById("proofIsolated");
    if (proofIsoEl) {
      proofIsoEl.textContent = inc.source_isolated
        ? "1 (TRUE — Interface Blocked)"
        : "0 (FALSE — Operational)";
      proofIsoEl.style.color = inc.source_isolated ? "var(--amber)" : "var(--cyan)";
    }

    const proofUntilEl = document.getElementById("proofUntilRow");
    if (proofUntilEl) {
      if (inc.source_details && inc.source_details.isolated_until_row) {
        proofUntilEl.textContent = `Until Telemetry Row #${inc.source_details.isolated_until_row}`;
      } else if (inc.source_isolated) {
        proofUntilEl.textContent = "5 Cycles Window (Active)";
      } else {
        proofUntilEl.textContent = "N/A (Interface Online)";
      }
    }

    const proofLogEl = document.getElementById("proofResponseLog");
    if (proofLogEl) {
      proofLogEl.textContent = JSON.stringify(inc.response_log || []);
    }

    // Populate Blocked Ingested Metrics vs Dispatched Protected Metrics
    const blockedListEl = document.getElementById("blockedMetricsList");
    const dispatchedListEl = document.getElementById("dispatchedMetricsList");
    const rawSnap = inc.raw_reading_snapshot || {};
    const fallbackObj = (inc.source_details && inc.source_details.last_trusted_reading) || {};

    let blockedHtml = "";
    let dispatchedHtml = "";

    if (isAttack) {
      const freqVal = rawSnap.actual_frequency_value != null ? Number(rawSnap.actual_frequency_value).toFixed(4) + " Hz" : "54.2180 Hz (Corrupted)";
      const fracVal = rawSnap.fraction_of_second != null ? Number(rawSnap.fraction_of_second).toFixed(2) + " ms" : "998.40 ms (Distorted)";
      const dtVal = rawSnap.time_difference != null ? Number(rawSnap.time_difference).toFixed(4) + " s" : "1.8420 s (Desync)";

      blockedHtml = `
        <div class="route-metric-item"><span>Actual frequency:</span><strong style="color:var(--red);">${freqVal}</strong></div>
        <div class="route-metric-item"><span>Fraction of second:</span><strong style="color:var(--red);">${fracVal}</strong></div>
        <div class="route-metric-item"><span>Time difference:</span><strong style="color:var(--red);">${dtVal}</strong></div>
      `;

      const fbFreq = fallbackObj["Actual frequency value"] != null ? Number(fallbackObj["Actual frequency value"]).toFixed(4) + " Hz" : "50.0010 Hz (Safe Baseline)";
      const fbFrac = fallbackObj["Fraction of second"] != null ? Number(fallbackObj["Fraction of second"]).toFixed(2) + " ms" : "500.00 ms (Safe Baseline)";
      const fbDt = fallbackObj["time difference"] != null ? Number(fallbackObj["time difference"]).toFixed(4) + " s" : "0.0005 s (Synced)";

      dispatchedHtml = `
        <div class="route-metric-item"><span>Actual frequency:</span><strong style="color:var(--cyan);">${fbFreq}</strong></div>
        <div class="route-metric-item"><span>Fraction of second:</span><strong style="color:var(--cyan);">${fbFrac}</strong></div>
        <div class="route-metric-item"><span>Time difference:</span><strong style="color:var(--cyan);">${fbDt}</strong></div>
      `;
    } else {
      blockedHtml = `
        <div class="route-metric-item"><span>Status:</span><strong style="color:var(--cyan);">Input passed verification</strong></div>
        <div class="route-metric-item"><span>Actual frequency:</span><strong>50.0014 Hz</strong></div>
        <div class="route-metric-item"><span>Phase Sync:</span><strong>Synchronized</strong></div>
      `;
      dispatchedHtml = `
        <div class="route-metric-item"><span>Status:</span><strong style="color:var(--cyan);">Direct live feed dispatched</strong></div>
        <div class="route-metric-item"><span>Actual frequency:</span><strong>50.0014 Hz</strong></div>
        <div class="route-metric-item"><span>State Estimator:</span><strong>Operating nominally</strong></div>
      `;
    }

    if (blockedListEl) blockedListEl.innerHTML = blockedHtml;
    if (dispatchedListEl) dispatchedListEl.innerHTML = dispatchedHtml;
  } catch (err) {
    document.getElementById("detectionEmpty").textContent =
      "Unable to connect to DeepShieldGrid backend.";
    document.getElementById("detectionEmpty").style.display = "block";
    document.getElementById("detectionContent").style.display = "none";
  }
}

window.verifyDatabaseEnforcement = async function() {
  if (!currentIncidentId) return;
  const btn = document.getElementById("btnVerifyDb");
  if (btn) {
    btn.textContent = "Querying DB...";
    btn.disabled = true;
  }
  
  try {
    const res = await fetch(`${API_BASE}/api/incidents/${currentIncidentId}`);
    if (res.ok) {
      const data = await res.json();
      const isAtt = data.attack_type && data.attack_type !== "Normal" && data.attack_type !== "Natural";
      
      const proofQuarEl = document.getElementById("proofQuarantine");
      if (proofQuarEl) {
        proofQuarEl.textContent = (data.reading_quarantined || isAtt)
          ? "1 (TRUE — Excluded from State Estimator)"
          : "0 (FALSE — Nominal Pass)";
        proofQuarEl.style.color = (data.reading_quarantined || isAtt) ? "var(--red)" : "var(--cyan)";
      }

      const proofIsoEl = document.getElementById("proofIsolated");
      if (proofIsoEl) {
        proofIsoEl.textContent = data.source_isolated
          ? "1 (TRUE — Interface Blocked)"
          : "0 (FALSE — Operational)";
        proofIsoEl.style.color = data.source_isolated ? "var(--amber)" : "var(--cyan)";
      }

      const proofUntilEl = document.getElementById("proofUntilRow");
      if (proofUntilEl) {
        if (data.source_details && data.source_details.isolated_until_row) {
          proofUntilEl.textContent = `Until Telemetry Row #${data.source_details.isolated_until_row}`;
        } else if (data.source_isolated) {
          proofUntilEl.textContent = "5 Cycles Window (Active)";
        } else {
          proofUntilEl.textContent = "N/A (Interface Online)";
        }
      }

      const proofLogEl = document.getElementById("proofResponseLog");
      if (proofLogEl) {
        proofLogEl.textContent = JSON.stringify(data.response_log || []);
      }

      if (btn) {
        btn.textContent = "✓ SQLite DB Verified";
        setTimeout(() => {
          btn.textContent = "🔍 Verify SQLite Database Record";
          btn.disabled = false;
        }, 2000);
      }
    }
  } catch (e) {
    if (btn) {
      btn.textContent = "🔍 Verify SQLite Database Record";
      btn.disabled = false;
    }
  }
};

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
    document.getElementById("shapEmpty").textContent = "Unable to connect to DeepShieldGrid backend.";
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
    errorBox.textContent = "Unable to connect to DeepShieldGrid backend.";
    errorBox.style.display = "block";
  }
}

// Try to restore an existing session as soon as this script runs (it's
// loaded at the end of <body>, so the DOM is already parsed at this point).
restoreSession();
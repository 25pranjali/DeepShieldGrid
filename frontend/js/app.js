// ===========================================================================
// GridSentry frontend logic.
// Talks to the Flask backend running at API_BASE. If the backend isn't
// running, every function below shows a clear "can't connect" message
// instead of leaving blank/undefined values on the page.
// ===========================================================================

const API_BASE = "http://127.0.0.1:5000";

let currentRole = null;
let currentIncidentId = null;
let dashboardPollTimer = null;

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

  try {
    const res = await fetch(`${API_BASE}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();

    if (!res.ok || !data.success) {
      errorBox.textContent = data.error || "Invalid username or password.";
      errorBox.classList.add("visible");
      return;
    }

    currentRole = data.role;
    document.getElementById("login").style.display = "none";
    document.getElementById("app").classList.add("visible");
    document.getElementById("rolePill").textContent =
      "Role: " + (currentRole === "admin" ? "Admin" : "User");

    // What-If Simulator is admin-only
    document.getElementById("navWhatIf").hidden = currentRole !== "admin";

    updateDashboard();
    dashboardPollTimer = setInterval(updateDashboard, 4000);
  } catch (err) {
    errorBox.textContent = "Unable to connect to the GridSentry backend. Is server.py running?";
    errorBox.classList.add("visible");
  }
}

function logout() {
  currentRole = null;
  currentIncidentId = null;
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

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
async function updateDashboard() {
  await Promise.all([loadLatestReading(), loadIncidents(), loadSimulationStatus()]);
}

function fmt(value, decimals = 4) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toFixed(decimals);
}

async function loadLatestReading() {
  const errorBox = document.getElementById("dashboardError");
  try {
    const res = await fetch(`${API_BASE}/api/readings/latest`);
    const data = await res.json();

    if (!res.ok) {
      // No readings yet is expected before the first simulation run - not a hard error.
      return;
    }
    errorBox.innerHTML = "";

    const f = data.features || {};
    document.getElementById("readFrequency").textContent = fmt(f["Actual frequency value"], 4) + " Hz";
    document.getElementById("readFraction").textContent = fmt(f["Fraction of second"], 2);
    document.getElementById("readInterarrival").textContent = fmt(f["interarrival time"], 5) + " s";
    document.getElementById("readTimeDiff").textContent = fmt(f["time difference"], 4) + " s";

    document.getElementById("readingTimestamp").textContent = data.timestamp || "—";
    document.getElementById("predClass").textContent = data.prediction || "—";
    document.getElementById("predConfidence").textContent =
      data.confidence != null ? data.confidence + "%" : "—";
    document.getElementById("predGroundTruth").textContent = data.ground_truth_label || "—";
    document.getElementById("predTimeSync").textContent =
      f["Time synchronized"] != null ? f["Time synchronized"] : "—";

    const riskScoreEl = document.getElementById("riskScoreNum");
    const riskLevelEl = document.getElementById("riskLevelText");
    riskScoreEl.textContent = data.risk_score != null ? data.risk_score : "—";
    riskLevelEl.textContent = data.risk_level || "—";
    riskLevelEl.className = "risk-level " + (data.risk_level ? data.risk_level.toLowerCase() : "");

    const banner = document.getElementById("statusBanner");
    const isAttack = data.prediction && data.prediction !== "Normal";
    banner.className = "status-banner " + (isAttack ? "attack" : "normal");
    document.getElementById("bannerTitle").textContent = isAttack
      ? `${data.prediction} detected on the current PMU reading`
      : (data.prediction === "Normal" ? "System operating normally" : "Waiting for data…");
    document.getElementById("bannerSub").textContent = isAttack
      ? `Confidence ${data.confidence}% · Risk: ${data.risk_level}`
      : "Simulated feed running from the historical dataset";
  } catch (err) {
    errorBox.innerHTML = `<div class="error-msg">Unable to connect to GridSentry backend. Make sure server.py is running on ${API_BASE}.</div>`;
  }
}

async function loadIncidents() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    const incidents = await res.json();

    const tbody = document.getElementById("incidentsTableBody");
    if (!Array.isArray(incidents) || incidents.length === 0) {
      tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No incidents yet</td></tr>';
      return;
    }

    tbody.innerHTML = incidents
      .map(
        (inc) => `
        <tr class="clickable" onclick="openIncident(${inc.id})">
          <td class="mono">${(inc.timestamp || "").slice(0, 19).replace("T", " ")}</td>
          <td>${inc.source || "—"}</td>
          <td>${inc.attack_type}</td>
          <td>${inc.confidence != null ? inc.confidence + "%" : "—"}</td>
          <td><span class="badge attack">${inc.risk_level || "—"}</span></td>
        </tr>`
      )
      .join("");
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

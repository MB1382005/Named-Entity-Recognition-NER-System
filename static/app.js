const sourceInput = document.getElementById("sourceInput");
const extractBtn = document.getElementById("extractBtn");
const clearBtn = document.getElementById("clearBtn");
const annotatedProse = document.getElementById("annotatedProse");
const entityList = document.getElementById("entityList");
const errorStrip = document.getElementById("errorStrip");

let lastResult = null;
let activeFilter = "ALL";

// Top tabs
document.querySelectorAll(".tab-link").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-link").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
  });
});

// Preset snippets
document.querySelectorAll(".preset-btn[data-sample]").forEach((btn) => {
  btn.addEventListener("click", () => {
    sourceInput.value = btn.dataset.sample;
    runExtraction();
  });
});

clearBtn.addEventListener("click", () => {
  sourceInput.value = "";
  lastResult = null;
  annotatedProse.innerHTML = '<span class="prose-empty">Detected entity spans will appear highlighted inline here.</span>';
  entityList.innerHTML = '<div class="list-empty">No entities extracted yet.</div>';
  document.getElementById("totalCount").textContent = "0 spans";
  ["ALL", "PER", "ORG", "LOC", "MISC"].forEach((k) => {
    document.getElementById(`cnt-${k}`).textContent = "0";
  });
  errorStrip.classList.add("hidden");
});

// Filter pills
document.querySelectorAll(".filter-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    activeFilter = pill.dataset.filter;
    if (lastResult) renderEntityList(lastResult.entities);
  });
});

extractBtn.addEventListener("click", runExtraction);
sourceInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    runExtraction();
  }
});

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function runExtraction() {
  const text = sourceInput.value.trim();
  if (!text) return;

  extractBtn.disabled = true;
  errorStrip.classList.add("hidden");

  try {
    const res = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Extraction failed.");
    lastResult = data;
    renderAnnotated(data);
    renderCounts(data);
    renderEntityList(data.entities);
  } catch (err) {
    errorStrip.textContent = err.message;
    errorStrip.classList.remove("hidden");
  } finally {
    extractBtn.disabled = false;
  }
}

function renderAnnotated(data) {
  const text = data.text;
  let html = "";
  let cursor = 0;

  for (const ent of data.entities) {
    html += escapeHtml(text.slice(cursor, ent.start));
    html += `<span class="ent-mark ent-${ent.label}" title="${ent.label} (${ent.score}%)">${escapeHtml(ent.text)}<sub>${ent.label}</sub></span>`;
    cursor = ent.end;
  }
  html += escapeHtml(text.slice(cursor));
  annotatedProse.innerHTML = html;
}

function renderCounts(data) {
  document.getElementById("totalCount").textContent = `${data.total} span${data.total === 1 ? "" : "s"}`;
  document.getElementById("cnt-ALL").textContent = data.total;
  for (const k of ["PER", "ORG", "LOC", "MISC"]) {
    document.getElementById(`cnt-${k}`).textContent = data.counts[k] || 0;
  }
}

function renderEntityList(entities) {
  const filtered = activeFilter === "ALL" ? entities : entities.filter((e) => e.label === activeFilter);
  if (!filtered.length) {
    entityList.innerHTML = '<div class="list-empty">No matching entities in this view.</div>';
    return;
  }
  entityList.innerHTML = filtered
    .map(
      (e) => `
      <div class="entity-item">
        <span class="ei-text">${escapeHtml(e.text)}</span>
        <div class="ei-right">
          <span class="ei-score">${e.score}%</span>
          <span class="leg leg-${e.label}">${e.label}</span>
        </div>
      </div>
    `
    )
    .join("");
}

async function initMeta() {
  try {
    const res = await fetch("/api/meta");
    const data = await res.json();
    document.getElementById("modelTag").textContent = data.model_name || "Not Loaded";

    const benchBody = document.getElementById("benchBody");
    benchBody.innerHTML = (data.benchmark || [])
      .map(
        (r) => `
        <tr>
          <td>${r["Model"]}</td>
          <td style="font-family:'JetBrains Mono',monospace;">${(Number(r["Precision"]) * 100).toFixed(2)}%</td>
          <td style="font-family:'JetBrains Mono',monospace;">${(Number(r["Recall"]) * 100).toFixed(2)}%</td>
          <td style="font-family:'JetBrains Mono',monospace;">${(Number(r["F1-Score"]) * 100).toFixed(2)}%</td>
        </tr>
      `
      )
      .join("");

    if (data.report) {
      document.getElementById("reportPre").textContent = data.report;
      document.getElementById("reportCard").classList.remove("hidden");
    }

    const schemaGrid = document.getElementById("schemaGrid");
    schemaGrid.innerHTML = Object.entries(data.schema || {})
      .map(
        ([k, v]) => `
        <div class="schema-card">
          <span class="leg leg-${k}">B-${k} / I-${k}</span>
          <h3>${v.name}</h3>
          <p>${v.desc}</p>
        </div>
      `
      )
      .join("");
  } catch (e) {
    console.error("Metadata error:", e);
  }
}

initMeta();

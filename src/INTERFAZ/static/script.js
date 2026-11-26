const $ = (id) => document.getElementById(id);
let info = null, mode = "nueva";
const pct = (v) => (100 * v).toFixed(1) + " %";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
// Nombres reales de vacas y máquinas (Data/mapa_id_vacas.csv); sin el mapa se muestra el identificador.
const cowName = (id) => info?.etiquetas?.vacas?.[id] ?? id;
const machineName = (id) => info?.etiquetas?.maquinas?.[id] ?? id;
const byName = (name) => (a, b) => name(a).localeCompare(name(b), "es", {numeric: true});
const fmtDate = (s) => { const [d, t] = s.split(" "); const [y, m, dd] = d.split("-"); return `${dd}/${m}/${y} ${t.slice(0, 5)}`; };

async function api(path, body) {
  const response = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Error en la petición");
  return data;
}

function thi(t, rh) { return (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26); }
function updateThi() {
  const t = parseFloat($("temperatura").value), rh = parseFloat($("humedad").value);
  $("thi").textContent = isNaN(t) || isNaN(rh) ? "" : `THI de la sesión: ${thi(t, rh).toFixed(1)}`;
}

function setMode(next) {
  mode = next;
  document.querySelectorAll(".modes button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("nueva").hidden = mode !== "nueva";
  $("existente").hidden = mode !== "existente";
  $("mode-hint").textContent = mode === "nueva"
    ? "Capturas lo que se sabe al entrar al robot; el resto sale del historial de la vaca."
    : "Predice un ordeño que ya está en los datos y muestra si el evento ocurrió.";
}

async function loadCow() {
  const cow = await api(`/api/vacas/${encodeURIComponent($("vaca").value)}`);
  $("ultimo").textContent = `Último ordeño: ${fmtDate(cow.ultimo)}`;
  const [date, time] = cow.sugerido.split(" ");
  $("fecha").value = date; $("hora").value = time.slice(0, 5);
  $("maquina").value = cow.maquina; $("numero").value = cow.numero_ordeno;
  $("temperatura").value = cow.temperatura; $("humedad").value = cow.humedad;
  $("horas").value = cow.horas_thi_alto_24h;
  $("ordeno").innerHTML = cow.ordenos.map((s) => `<option value="${s}">${fmtDate(s)}</option>`).join("");
  updateThi();
}

// Los factores son las variables que más aumentan el riesgo (aporte positivo), hasta 3, con su
// explicación sin tecnicismos y qué revisar.
function factorList(factors) {
  if (!factors.length) return `<p class="hint factors-note">Ninguna variable aumenta el riesgo frente a una vaca promedio.</p>`;
  const note = factors.length < 3
    ? `<p class="hint factors-note">Solo ${factors.length === 1 ? "1 variable aumenta" : `${factors.length} variables aumentan`} el riesgo.</p>` : "";
  return `<ol class="factors">${factors.map((f, i) => `
    <li><span class="name">${i + 1}. ${esc(f.nombre)}</span>${f.cuando ? ` <span class="value">(${esc(f.cuando)})</span>` : ""}
      <div class="value">${esc(f.valor_legible)}</div>
      <div class="up">▲ aumenta el riesgo</div>
      <details><summary>¿Qué significa y qué hacer?</summary>
        ${f.significado ? `<p>${esc(f.significado)}</p><p class="action"><b>Qué revisar:</b> ${esc(f.accion)}</p>` : ""}
        <p class="tech">Variable ${esc(f.variable)} · valor ${esc(f.valor)} · aporte +${f.aporte.toFixed(3)} log-odds</p>
      </details>
    </li>`).join("")}</ol>${note}`;
}

function renderResults(data) {
  const events = info.eventos, models = Object.keys(data.resultados);
  const header = `<tr><th>Evento</th>${models.map((m) => `<th>${esc(m)}</th>`).join("")}${data.real ? "<th>Real</th>" : ""}</tr>`;
  const rows = events.map((e) => `<tr><td>${esc(e.nombre)}</td>${models.map((m) => {
    const r = data.resultados[m][e.id];
    return `<td class="num">${pct(r.probabilidad)} ${r.alerta ? '<span class="badge alert">alerta</span>' : ""}</td>`;
  }).join("")}${data.real ? `<td>${data.real[e.id] ? "ocurrió" : "no ocurrió"}</td>` : ""}</tr>`).join("");

  const detail = models.map((m) => `<div class="model"><h3>${esc(m)}</h3><div class="cards">${events.map((e) => {
    const r = data.resultados[m][e.id];
    return `<div class="card">
      <div class="event">${esc(e.nombre)}</div>
      <div class="prob">${pct(r.probabilidad)}</div>
      <div class="meter"><span style="width:${(100 * r.probabilidad).toFixed(1)}%"></span><i style="left:${(100 * r.umbral).toFixed(1)}%"></i></div>
      <span class="badge ${r.alerta ? "alert" : "ok"}">${r.alerta ? "Alerta" : "Sin alerta"} · umbral ${pct(r.umbral)}</span>
      ${data.real ? `<div class="real">Real: ${data.real[e.id] ? "ocurrió" : "no ocurrió"}</div>` : ""}
      ${factorList(r.factores)}
    </div>`;
  }).join("")}</div></div>`).join("");

  const notes = (data.avisos || []).map((a) => `<p class="notice">${esc(a)}</p>`).join("");
  $("tab-resultados").innerHTML = `<h2>Vaca ${esc(cowName(data.vaca))} · ${fmtDate(data.inicio)}</h2>${notes}
    <div class="table-wrap"><table>${header}${rows}</table></div>${detail}`;
  $("tab-variables").innerHTML = `<p class="hint">Valores con los que se predijo la sesión, en sus unidades. Casi todas
    describen ordeños anteriores de la vaca o del hato; en la pestaña <b>Glosario</b> está qué significa cada una.</p>
    <div class="table-wrap"><table><tr><th>Variable</th><th>Valor</th></tr>${data.variables.map((v) =>
      `<tr><td>${esc(v.nombre)}<div class="tech">${esc(v.variable)}</div></td><td class="num">${esc(v.valor)}</td></tr>`).join("")}</table></div>`;
}

function performanceTable(name, performance) {
  const metrics = Object.keys(Object.values(performance)[0]);
  return `<div class="model"><h3>${esc(name)}</h3><div class="table-wrap"><table>
    <tr><th>Evento</th>${metrics.map((k) => `<th>${k}</th>`).join("")}</tr>
    ${info.eventos.map((e) => `<tr><td>${esc(e.nombre)}</td>${metrics.map((k) => {
      const v = performance[e.id][k];
      return `<td class="num">${v === null ? "—" : v.toFixed(3)}</td>`;
    }).join("")}</tr>`).join("")}
  </table></div></div>`;
}

function renderPerformance() {
  const baseline = info.linea_base
    ? performanceTable("Línea base sin modelo: persistencia", info.linea_base) +
      `<p class="hint">Alerta si el ordeño anterior de la vaca tuvo el evento (src/EVALUATION/lineas_base.py). No da probabilidad, así que no tiene Brier.</p>`
    : "";
  $("tab-desempeno").innerHTML = `<p class="hint">Métricas en test (el 15 % más reciente de las sesiones). F1, recall y precisión usan el umbral
    elegido en validación para detectar al menos 80 % de los eventos; Brier mide la calibración de la probabilidad (más bajo es mejor).</p>` +
    info.modelos.map((m) => performanceTable(m.nombre, m.desempeno)).join("") + baseline;
}

function renderGlossary() {
  $("tab-glosario").innerHTML = `<p class="hint">Qué significa cada variable que puede aparecer como factor y qué revisar
    cuando aparece. Son sugerencias: los factores son asociaciones que encontró cada modelo, no causas, y no sustituyen
    la revisión del veterinario o del técnico.</p>
    <dl class="glossary">${info.glosario.map((g) => `<div>
      <dt>${esc(g.nombre)} <span class="tech">${esc(g.variable)}</span></dt>
      <dd>${esc(g.significado)}</dd>
      <dd class="action"><b>Qué revisar:</b> ${esc(g.accion)}</dd>
    </div>`).join("")}</dl>`;
}

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("error").textContent = "";
  const chosen = [...document.querySelectorAll("#modelos input:checked")].map((i) => i.value);
  if (!chosen.length) { $("error").textContent = "Elige al menos un modelo."; return; }
  const body = {modo: mode, vaca: $("vaca").value, modelos: chosen};
  if (mode === "nueva") {
    Object.assign(body, {
      inicio: `${$("fecha").value} ${$("hora").value}`, maquina: $("maquina").value,
      temperatura: parseFloat($("temperatura").value), humedad: parseFloat($("humedad").value),
      horas_thi_alto_24h: parseFloat($("horas").value), numero_ordeno: parseInt($("numero").value),
    });
    if ([body.temperatura, body.humedad, body.horas_thi_alto_24h, body.numero_ordeno].some(isNaN) || !$("fecha").value || !$("hora").value) {
      $("error").textContent = "Completa todos los campos."; return;
    }
  } else {
    body.inicio = $("ordeno").value;
  }
  const button = $("predecir");
  button.disabled = true; button.textContent = "Corriendo los modelos...";
  showTab("resultados");
  $("tab-resultados").innerHTML = `<p class="loading">Corriendo los modelos...</p>`;
  try {
    renderResults(await api("/api/predecir", body));
  } catch (error) {
    $("error").textContent = error.message;
    $("tab-resultados").innerHTML = `<p class="empty">${esc(error.message)}</p>`;
  } finally {
    button.disabled = false; button.textContent = "Predecir";
  }
});

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  ["resultados", "variables", "desempeno", "glosario"].forEach((t) => { $(`tab-${t}`).hidden = t !== name; });
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
["temperatura", "humedad"].forEach((id) => $(id).addEventListener("input", updateThi));
$("vaca").addEventListener("change", () => loadCow().catch((e) => { $("error").textContent = e.message; }));

(async () => {
  setMode("nueva");
  try {
    info = await api("/api/info");
    const cows = Object.keys(info.etiquetas.vacas).length ? [...info.vacas].sort(byName((v) => cowName(v.id))) : info.vacas;
    $("vaca").innerHTML = cows.map((v) => `<option value="${esc(v.id)}">${esc(cowName(v.id))} (último ${fmtDate(v.ultimo).slice(0, 10)})</option>`).join("");
    $("fecha").max = info.fecha_maxima;
    $("fecha-hint").textContent = `Fecha máxima: ${fmtDate(info.fecha_maxima + " 00:00").slice(0, 10)}, el día siguiente al fin del dataset.`;
    $("maquina").innerHTML = [...info.maquinas].sort(byName(machineName))
      .map((m) => `<option value="${esc(m)}">${esc(machineName(m))}</option>`).join("");
    $("modelos").innerHTML = info.modelos.map((m) => `<label><input type="checkbox" value="${esc(m.nombre)}" checked> ${esc(m.nombre)}</label>`).join("");
    renderPerformance();
    renderGlossary();
    await loadCow();
    $("predecir").disabled = false; $("predecir").textContent = "Predecir";
  } catch (error) {
    $("error").textContent = "No se pudo conectar con el servidor. Córrelo con: python3 src/INTERFAZ/app.py";
  }
})();

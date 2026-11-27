const $ = (id) => document.getElementById(id);
let info = null, mode = "nueva", lastResult = null;
const pct = (v) => (100 * v).toFixed(1) + " %";
const pct0 = (v) => Math.round(100 * v) + " %";
// Sin decimales, salvo que dos valores que se comparan se redondeen igual (33.7 % contra 33.8 %).
const pctVs = (v, other) => pct0(v) === pct0(other) ? pct(v) : pct0(v);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
// Nombres reales de vacas y máquinas (Data/mapa_id_vacas.csv); sin el mapa se muestra el identificador.
const cowName = (id) => info?.etiquetas?.vacas?.[id] ?? id;
const machineName = (id) => info?.etiquetas?.maquinas?.[id] ?? id;
const byName = (name) => (a, b) => name(a).localeCompare(name(b), "es", {numeric: true});
// Duraciones en minutos (de tiempos.py) a «5 min 17 s» o «56 s».
function fmtDur(minutes) {
  const total = Math.round(Math.abs(minutes) * 60), m = Math.floor(total / 60), s = total % 60;
  return m ? (s ? `${m} min ${s} s` : `${m} min`) : `${s} s`;
}
const fmtExtra = (minutes) => (minutes < 0 ? "−" : "+") + fmtDur(minutes);
const fmtDate = (s) => { const [d, t] = s.split(" "); const [y, m, dd] = d.split("-"); return `${dd}/${m}/${y} ${t.slice(0, 5)}`; };

// Íconos de static/iconos.svg: acompañan al texto, nunca lo reemplazan (por eso aria-hidden).
const ico = (id, cls = "") => `<svg class="ico ${cls}" aria-hidden="true"><use href="iconos.svg#${id}"/></svg>`;
const EVENT_ICONS = {
  mechanical_connection_failure_event: "pezonera", health_anomaly_event: "salud",
  behavior_interruption_event: "patada", alguna_anomalia: "alerta",
};
const eventName = (id) => info.eventos.find((e) => e.id === id).nombre;
const eventLabel = (id) => `${ico(EVENT_ICONS[id])}${esc(eventName(id))}`;
// Primero alguna anomalía (la pregunta principal) y después los tres eventos.
const eventsFirstAny = () => [...info.eventos].sort((a, b) => (b.id === "alguna_anomalia") - (a.id === "alguna_anomalia"));

const emptyState = (text) => `<div class="empty">
  <svg class="empty-art" viewBox="0 0 240 96" aria-hidden="true">
    <use href="iconos.svg#vaca" x="20" y="10" width="76" height="76" class="art-cow"/>
    <path d="M108 48h34m-7-7 7 7-7 7" class="art-arrow"/>
    <use href="iconos.svg#robot" x="148" y="10" width="76" height="76" class="art-robot"/>
  </svg><p>${text}</p></div>`;

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

// Ubre vista desde arriba (cabeza hacia arriba) con el cuarto de la variable resaltado.
const QUARTERS = {di: "delantero izquierdo", dd: "delantero derecho", ti: "trasero izquierdo", td: "trasero derecho"};
function udder(variable) {
  const quarter = variable.match(/^blood_diff_(di|dd|ti|td)$/)?.[1];
  if (!quarter) return "";
  const cell = (q, x, y) => `<rect x="${x}" y="${y}" width="40" height="30" rx="10" class="${q === quarter ? "on" : ""}"/>
    <circle cx="${x + 20}" cy="${y + 15}" r="4"/>`;
  return `<figure class="udder"><svg viewBox="0 0 96 76" role="img" aria-label="Cuarto ${QUARTERS[quarter]}">
      <text x="48" y="8">cabeza ↑</text>${cell("di", 6, 12)}${cell("dd", 50, 12)}${cell("ti", 6, 44)}${cell("td", 50, 44)}
    </svg><figcaption>Cuarto ${QUARTERS[quarter]}</figcaption></figure>`;
}

// La variable que más aumenta el riesgo (mayor aporte positivo), con su explicación sin
// tecnicismos y qué revisar.
function factorList(factors) {
  if (!factors.length) return `<p class="hint factors-note">Ninguna variable aumenta el riesgo frente a una vaca promedio.</p>`;
  return `<ol class="factors"><li class="factors-title">Variable más importante</li>${factors.map((f) => `
    <li><span class="name">${ico(f.icono)}${esc(f.nombre)}</span>
      <div class="value">${esc(f.valor_legible)}</div>
      ${udder(f.variable)}
      <div class="up">▲ aumenta el riesgo</div>
      <details><summary>¿Qué significa y qué hacer?</summary>
        ${f.significado ? `<p>${esc(f.significado)}</p><p class="action"><b>Qué revisar:</b> ${esc(f.accion)}</p>` : ""}
        <p class="tech">Variable ${esc(f.variable)} · valor ${esc(f.valor)} · aporte +${f.aporte.toFixed(3)} log-odds</p>
      </details>
    </li>`).join("")}</ol>`;
}

function renderResults(data) {
  const events = info.eventos;
  const times = data.tiempo;
  const header = `<tr><th>Evento</th><th>Probabilidad</th><th>Umbral de alerta</th>${times ? "<th>Si ocurre, tarda</th>" : ""}${data.real ? "<th>Real</th>" : ""}</tr>`;
  const rows = events.map((e) => {
    const r = data.resultados[e.id];
    return `<tr><td><span class="event-cell">${eventLabel(e.id)}</span></td>
      <td class="num">${pct(r.probabilidad)} ${r.alerta ? '<span class="badge alert">alerta</span>' : ""}</td>
      <td class="num">${pct(r.umbral)}</td>${times ? `<td class="num">${fmtExtra(times.eventos[e.id].extra_mediana)}</td>` : ""}${data.real ? `<td>${data.real[e.id] ? "ocurrió" : "no ocurrió"}</td>` : ""}</tr>`;
  }).join("");

  const detail = `<div class="model"><h3>${esc(info.modelo.nombre)}</h3><div class="cards">${events.map((e) => {
    const r = data.resultados[e.id];
    return `<div class="card">
      <div class="event">${eventLabel(e.id)}</div>
      <div class="prob">${pct(r.probabilidad)}</div>
      <div class="meter"><span style="width:${(100 * r.probabilidad).toFixed(1)}%"></span><i style="left:${(100 * r.umbral).toFixed(1)}%"></i></div>
      <span class="badge ${r.alerta ? "alert" : "ok"}">${r.alerta ? "Alerta" : "Sin alerta"} · umbral ${pct(r.umbral)}</span>
      ${data.real ? `<div class="real">Real: ${data.real[e.id] ? "ocurrió" : "no ocurrió"}</div>` : ""}
      ${times ? `<p class="delay">${ico("reloj")}<span>${delayText(e.id, times)}</span></p>` : ""}
      ${factorList(r.factores)}
    </div>`;
  }).join("")}</div></div>`;

  const notes = (data.avisos || []).map((a) => `<p class="notice">${ico("alerta")}${esc(a)}</p>`).join("");
  $("tab-resultados").innerHTML = `<h2 class="cow-title">${ico("vaca")}Vaca ${esc(cowName(data.vaca))} · ${fmtDate(data.inicio)}</h2>${notes}
    <div class="table-wrap"><table>${header}${rows}</table></div>${detail}`;
  $("tab-variables").innerHTML = `<p class="hint">Valores con los que se predijo la sesión, en sus unidades. Casi todas
    describen ordeños anteriores de la vaca o del hato; en la pestaña <b>Glosario</b> está qué significa cada una.</p>
    <div class="table-wrap"><table><tr><th>Variable</th><th>Valor</th></tr>${data.variables.map((v) =>
      `<tr><td><span class="var-name">${ico(v.icono)}${esc(v.nombre)}</span><div class="tech">${esc(v.variable)}</div></td>
        <td class="num">${esc(v.valor)}</td></tr>`).join("")}</table></div>`;
}

function performanceTable(name, performance) {
  const metrics = Object.keys(Object.values(performance)[0]);
  return `<div class="model"><h3>${esc(name)}</h3><div class="table-wrap"><table>
    <tr><th>Evento</th>${metrics.map((k) => `<th>${k}</th>`).join("")}</tr>
    ${info.eventos.map((e) => `<tr><td><span class="event-cell">${eventLabel(e.id)}</span></td>${metrics.map((k) => {
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
    performanceTable(info.modelo.nombre, info.modelo.desempeno) + baseline;
}

function renderGlossary() {
  $("tab-glosario").innerHTML = `<p class="hint">Qué significa cada variable que puede aparecer como la más importante y qué
    revisar cuando aparece. Son sugerencias: es una asociación que encontró el modelo, no una causa, y no sustituye
    la revisión del veterinario o del técnico.</p>
    <dl class="glossary">${info.glosario.map((g) => `<div>
      <dt><span class="glossary-icon">${ico(g.icono)}</span><span>${esc(g.nombre)}<span class="tech">${esc(g.variable)}</span></span></dt>
      <dd>${esc(g.significado)}</dd>
      <dd class="action"><b>Qué revisar:</b> ${esc(g.accion)}</dd>
    </div>`).join("")}</dl>`;
}

// --- Gráficas: velocímetros del XGBoost. «Alguna anomalía» va en grande y los otros tres eventos en
// velocímetros chicos.

// Estado de un evento: alerta desde el umbral; «vigilar» desde la mitad del umbral (regla de la interfaz).
const STATES = {
  alerta: {label: "Alerta", icon: "alerta", rank: 2},
  vigilar: {label: "Vigilar", icon: "ojo", rank: 1},
  normal: {label: "Normal", icon: "check", rank: 0},
};
const stateOf = (r) => r.alerta ? "alerta" : r.probabilidad >= r.umbral / 2 ? "vigilar" : "normal";
const stateBadge = (state) => `<span class="state st-${state}">${ico(STATES[state].icon)}${STATES[state].label}</span>`;
const realText = (eventId) => lastResult.real ? `<span class="viz-real">Real: ${lastResult.real[eventId] ? "ocurrió" : "no ocurrió"}</span>` : "";
const mainFactor = (r) => r.factores[0];
const joinNames = (names) => names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} y ${names.at(-1)}`;

function whyLine(r, withAction) {
  const f = mainFactor(r);
  if (!f) return `<p class="why muted">Ninguna variable aumenta el riesgo frente a una vaca promedio.</p>`;
  return `<p class="why">${ico(f.icono)}<span>Lo que más pesó: <b>${esc(f.nombre)}</b>${f.valor_legible ? ` · ${esc(f.valor_legible)}` : ""}</span></p>
    ${udder(f.variable)}
    ${withAction ? `<p class="todo"><b>Qué revisar:</b> ${esc(f.accion)}</p>` : ""}`;
}

// Franja de veredicto: alerta, vigilar o normal, con los eventos que lo provocan.
function verdict() {
  const results = lastResult.resultados, cow = esc(cowName(lastResult.vaca));
  const named = (state) => eventsFirstAny().filter((e) => stateOf(results[e.id]) === state).map((e) => eventName(e.id).toLowerCase());
  const alerts = named("alerta"), watch = named("vigilar");
  const [state, text] = alerts.length ? ["alerta", `Revisar a la vaca ${cow}: alerta en ${esc(joinNames(alerts))}.`]
    : watch.length ? ["vigilar", `Vigilar a la vaca ${cow}: ${esc(joinNames(watch))} cerca del umbral.`]
    : ["normal", `Sin alertas: ordeño normal para la vaca ${cow}.`];
  const main = mainDelayEvent(), t = lastResult.tiempo;
  const time = main ? `<p class="verdict-time">${ico("reloj")}<span>Si ocurre ${esc(eventName(main).toLowerCase())}, el ordeño
    tardaría unos <b>${fmtDur(t.eventos[main].extra_mediana)} más</b>${t.normal != null ? ` (${fmtDur(t.normal)} → ${fmtDur(t.eventos[main].esperado)})` : ""}.</span></p>` : "";
  return `<div class="verdict st-${state}"><span class="verdict-light">${ico(STATES[state].icon)}</span><div><p>${text}</p>${time}</div></div>`;
}

// --- Tiempo del ordeño (src/EVALUATION/duracion_por_evento.py): cuánto suele alargarse si ocurre cada evento.

// Frase de cuánto se alarga el ordeño si ocurre el evento, frente a lo normal de la vaca.
function delayText(eventId, t = lastResult.tiempo) {
  const e = t.eventos[eventId];
  const range = t.normal != null ? ` (${fmtDur(t.normal)} → ${fmtDur(e.esperado)})` : "";
  return `Si ocurre, el ordeño suele tardar <b>${fmtDur(e.extra_mediana)} más</b> que uno normal de esta vaca${range}.`;
}
const delayLine = (eventId) => lastResult.tiempo ? `<p class="delay">${ico("reloj")}<span>${delayText(eventId)}</span></p>` : "";

// El evento que más alarga el ordeño entre los que están en alerta (o, si no hay, en vigilar). Se prefieren
// los tres eventos concretos; «alguna anomalía» solo si ninguno de ellos está en esa zona.
function mainDelayEvent() {
  const r = lastResult.resultados, t = lastResult.tiempo;
  if (!t) return null;
  for (const state of ["alerta", "vigilar"]) {
    const ids = info.eventos.filter((e) => e.id !== "alguna_anomalia" && stateOf(r[e.id]) === state).map((e) => e.id);
    if (ids.length) return ids.sort((a, b) => t.eventos[b].extra_mediana - t.eventos[a].extra_mediana)[0];
    if (stateOf(r.alguna_anomalia) === state) return "alguna_anomalia";
  }
  return null;
}

// Barras de minutos extra por evento (mediana), lo normal de la vaca, lo que duró de verdad (en un ordeño
// del dataset) y la espera en la fila de la máquina de la sesión.
function timeSection() {
  const t = lastResult.tiempo, results = lastResult.resultados;
  if (!t) return "";
  const max = Math.max(...Object.values(t.eventos).map((e) => e.extra_mediana));
  const rows = eventsFirstAny().map((ev) => {
    const e = t.eventos[ev.id], state = stateOf(results[ev.id]);
    return `<div class="delay-row st-${state}" tabindex="0" data-event="${ev.id}">
      <span class="delay-name">${eventLabel(ev.id)}</span>
      <span class="delay-track"><span class="delay-bar" style="width:${(100 * e.extra_mediana / max).toFixed(1)}%"></span></span>
      <span class="delay-val">${fmtExtra(e.extra_mediana)}</span>
      <span class="delay-state">${state === "normal" ? "" : stateBadge(state)}</span>
    </div>`;
  }).join("");
  const cow = esc(cowName(lastResult.vaca));
  const normal = t.normal != null
    ? `Un ordeño normal de la vaca ${cow} dura <b>${fmtDur(t.normal)}</b> (mediana de sus últimos ${t.ordenos_normales} ordeños sin evento).`
    : `La vaca ${cow} no tiene ordeños sin evento antes de esta sesión para saber cuánto dura uno normal.`;
  const real = t.real != null && t.normal != null
    ? `<p class="time-real">${ico("historial")}<span>Este ordeño duró <b>${fmtDur(t.real)}</b>: ${fmtExtra(t.real - t.normal)} frente a lo normal.</span></p>` : "";
  const main = mainDelayEvent() ?? "alguna_anomalia", saving = t.eventos[main].ahorro_espera;
  const queue = t.espera_maquina != null
    ? `<p class="time-queue">${ico("robot")}<span>En ${esc(machineName(lastResult.maquina))} la espera estimada en la fila es de
        ${fmtDur(t.espera_maquina)}${saving != null ? `; si todos los ordeños con ${esc(eventName(main).toLowerCase())} duraran lo
        normal, bajaría unos ${fmtDur(saving)}` : ""}.</span></p>` : "";
  return `<h3 class="section-title">${ico("reloj")}Tiempo del ordeño</h3>
    <section class="time-card">
      <p class="time-normal">${normal} Si ocurre un evento, suele tardar más:</p>
      <div class="delay-chart" aria-label="Minutos extra si ocurre cada evento">${rows}</div>
      ${real}${queue}
      <p class="hint">Retraso típico (mediana) de los ordeños con ese evento frente a los ordeños sin evento de la misma
        vaca, medido en test (${fmtDate(t.periodo[0] + " 00:00").slice(0, 10)} a ${fmtDate(t.periodo[1] + " 00:00").slice(0, 10)},
        ${t.ordenos_test.toLocaleString("es-MX")} ordeños; src/EVALUATION/duracion_por_evento.py). En promedio el retraso es
        mayor porque algunos ordeños se alargan mucho: pasa el cursor sobre una barra para verlo.</p>
    </section>`;
}

// Velocímetro (medio círculo): zonas normal / vigilar / alerta sobre 0-100 %, marca del umbral y aguja en
// la probabilidad.
function gaugeSvg(r) {
  const cx = 100, cy = 100, R = 80, p = Math.min(1, r.probabilidad), u = r.umbral;
  const pt = (v, radius) => [cx - radius * Math.cos(Math.PI * v), cy - radius * Math.sin(Math.PI * v)];
  const xy = ([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`;
  const arc = (a, b, zone) => b > a ? `<path class="zone z-${zone}" d="M ${xy(pt(a, R))} A ${R} ${R} 0 0 1 ${xy(pt(b, R))}"/>` : "";
  const line = (a, b, cls) => `<line class="${cls}" x1="${a[0].toFixed(2)}" y1="${a[1].toFixed(2)}" x2="${b[0].toFixed(2)}" y2="${b[1].toFixed(2)}"/>`;
  return `<svg viewBox="0 0 200 122" class="gauge" role="img"
      aria-label="${pct0(p)}; ${STATES[stateOf(r)].label}; alerta desde ${pct0(u)}">
    ${arc(0, u / 2, "normal")}${arc(u / 2, u, "vigilar")}${arc(u, 1, "alerta")}
    ${line(pt(u, R - 13), pt(u, R + 13), "gauge-thr")}
    <text class="gauge-end" x="${cx - R}" y="${cy + 20}">0</text><text class="gauge-end" x="${cx + R}" y="${cy + 20}">100</text>
    ${line([cx, cy], pt(p, R - 24), "needle")}<circle class="hub" cx="${cx}" cy="${cy}" r="7"/>
  </svg>`;
}

// «Alguna anomalía» en grande: velocímetro, número, las tres zonas escritas con su rango (se resalta la
// zona en la que cae), una frase que lo resume y lo que más pesó.
function heroGauge() {
  const r = lastResult.resultados.alguna_anomalia, state = stateOf(r), u = r.umbral;
  const pv = pctVs(r.probabilidad, u), uv = pctVs(u, r.probabilidad);
  const zones = [["normal", `0 a ${pct0(u / 2)}`], ["vigilar", `${pct0(u / 2)} a ${uv}`], ["alerta", `${uv} a 100 %`]];
  const sentence = {
    alerta: `Está en alerta: ${pv}, por encima del umbral de ${uv}.`,
    vigilar: `Está cerca del umbral: ${pv}; la alerta empieza en ${uv}.`,
    normal: `Está en zona normal: ${pv}; la alerta empieza en ${uv}.`,
  }[state];
  return `<section class="hero st-${state}" tabindex="0" data-event="alguna_anomalia">
    <h4>${eventLabel("alguna_anomalia")} ${realText("alguna_anomalia")}</h4>
    <div class="hero-chart">${gaugeSvg(r)}<p class="hero-value">${pv}</p></div>
    <div class="zone-scale" aria-label="Zonas del velocímetro">${zones.map(([z, range]) => `<div class="zone-item p-${z}${z === state ? " is-on" : ""}">
      <i class="zone-swatch"></i><span class="zone-name">${z === state ? ico(STATES[z].icon) : ""}${STATES[z].label}</span>
      <small>${range}</small></div>`).join("")}</div>
    <p class="hero-sentence">${sentence}</p>
    ${state !== "normal" ? delayLine("alguna_anomalia") : ""}
    ${whyLine(r, state !== "normal")}
  </section>`;
}

// Velocímetros chicos de los otros tres eventos.
function smallGauges() {
  const results = lastResult.resultados;
  return `<div class="gauge-grid">${info.eventos.filter((e) => e.id !== "alguna_anomalia").map((e) => {
    const r = results[e.id], state = stateOf(r);
    return `<section class="gauge-card st-${state}" tabindex="0" data-event="${e.id}">
      <h4>${eventLabel(e.id)}</h4>
      ${gaugeSvg(r)}
      <p class="gauge-value">${pctVs(r.probabilidad, r.umbral)}</p>
      <p class="gauge-meta">${stateBadge(state)} <span>alerta desde ${pctVs(r.umbral, r.probabilidad)}</span></p>
      <p class="gauge-real">${realText(e.id)}</p>
      ${state !== "normal" ? delayLine(e.id) : ""}
      ${whyLine(r, state !== "normal")}
    </section>`;
  }).join("")}</div>`;
}

const gaugeLegend = `<p class="hint key-row"><span class="key-zone z-normal"></span>Normal <span class="key-zone z-vigilar"></span>Vigilar
  (desde la mitad del umbral) <span class="key-zone z-alerta"></span>Alerta (desde el umbral del modelo) <span class="key-tick"></span>Umbral de alerta</p>`;

function renderCharts() {
  if (!lastResult) { $("tab-graficas").innerHTML = emptyState("Elige una vaca, captura la sesión y presiona <b>Predecir</b>."); return; }
  $("tab-graficas").innerHTML = `<p class="hint">Velocímetros de ${esc(info.modelo.nombre)}. Pasa el cursor (o toca) un
      velocímetro para ver el detalle.</p>
    ${verdict()}
    ${heroGauge()}
    <h3 class="section-title">Cada evento</h3>
    ${smallGauges()}${gaugeLegend}
    ${timeSection()}`;
}

// Tooltip compartido: valor primero y después el contexto (textContent: los nombres vienen de datos).
function tip(lines, x, y) {
  const el = $("viz-tip");
  el.replaceChildren(...lines.map((text, i) => {
    const div = document.createElement("div");
    div.textContent = text;
    div.className = i === 0 ? "tip-value" : i === 1 ? "tip-label" : "";
    return div;
  }));
  el.hidden = false;
  const box = el.getBoundingClientRect();
  el.style.left = `${Math.max(8, Math.min(x + 14, window.innerWidth - box.width - 8))}px`;
  el.style.top = `${Math.max(8, y - box.height - 12)}px`;
}
const hideTip = () => { $("viz-tip").hidden = true; };

const CHART_CARDS = ".hero, .gauge-card, .delay-row";
function cardTip(el, x, y) {
  if (el.classList.contains("delay-row")) {
    const t = lastResult.tiempo, e = t.eventos[el.dataset.event];
    tip([`${fmtExtra(e.extra_mediana)} (mediana)`, `${eventName(el.dataset.event)} · tiempo extra si ocurre`,
      `En promedio: ${fmtExtra(e.extra_media)}`,
      t.normal != null ? `Duración esperada: ${fmtDur(e.esperado)} (normal ${fmtDur(t.normal)})` : ""].filter(Boolean), x, y);
    return;
  }
  const r = lastResult.resultados[el.dataset.event], f = mainFactor(r);
  tip([pct(r.probabilidad), `${info.modelo.nombre} · ${eventName(el.dataset.event)}`,
    `Umbral ${pct(r.umbral)} · ${STATES[stateOf(r)].label}`,
    f ? `Lo que más pesó: ${f.nombre}${f.valor_legible ? ` (${f.valor_legible})` : ""}` : "Ninguna variable aumenta el riesgo"], x, y);
}
$("tab-graficas").addEventListener("pointermove", (e) => {
  const el = e.target.closest(CHART_CARDS);
  if (el) cardTip(el, e.clientX, e.clientY); else hideTip();
});
$("tab-graficas").addEventListener("pointerleave", hideTip);
$("tab-graficas").addEventListener("focusin", (e) => {
  const el = e.target.closest(CHART_CARDS);
  if (el) { const b = el.getBoundingClientRect(); cardTip(el, b.left + b.width / 2, b.top); }
});
$("tab-graficas").addEventListener("focusout", hideTip);

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("error").textContent = "";
  const body = {modo: mode, vaca: $("vaca").value};
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
  button.disabled = true; button.textContent = "Corriendo el modelo...";
  showTab("graficas");  // Al predecir se muestra la pestaña «Resultado» (los velocímetros)
  ["resultados", "graficas"].forEach((t) => { $(`tab-${t}`).innerHTML = `<p class="loading">Corriendo el modelo...</p>`; });
  try {
    lastResult = await api("/api/predecir", body);
    renderResults(lastResult);
    renderCharts();
  } catch (error) {
    lastResult = null;
    $("error").textContent = error.message;
    ["resultados", "graficas"].forEach((t) => { $(`tab-${t}`).innerHTML = emptyState(esc(error.message)); });
  } finally {
    button.disabled = false; button.textContent = "Predecir";
  }
});

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  hideTip();
  ["graficas", "resultados", "variables", "desempeno", "glosario"].forEach((t) => { $(`tab-${t}`).hidden = t !== name; });
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
["temperatura", "humedad"].forEach((id) => $(id).addEventListener("input", updateThi));
$("vaca").addEventListener("change", () => loadCow().catch((e) => { $("error").textContent = e.message; }));

(async () => {
  setMode("nueva");
  $("tab-graficas").innerHTML = emptyState("Elige una vaca, captura la sesión y presiona <b>Predecir</b>.");
  $("tab-resultados").innerHTML = emptyState("Aquí aparece el detalle después de presionar <b>Predecir</b>.");
  $("tab-variables").innerHTML = emptyState("Aquí aparecen las variables calculadas para la sesión.");
  try {
    info = await api("/api/info");
    renderCharts();
    const cows = Object.keys(info.etiquetas.vacas).length ? [...info.vacas].sort(byName((v) => cowName(v.id))) : info.vacas;
    $("vaca").innerHTML = cows.map((v) => `<option value="${esc(v.id)}">${esc(cowName(v.id))} (último ${fmtDate(v.ultimo).slice(0, 10)})</option>`).join("");
    $("fecha").max = info.fecha_maxima;
    $("fecha-hint").textContent = `Fecha máxima: ${fmtDate(info.fecha_maxima + " 00:00").slice(0, 10)}, el día siguiente al fin del dataset.`;
    $("maquina").innerHTML = [...info.maquinas].sort(byName(machineName))
      .map((m) => `<option value="${esc(m)}">${esc(machineName(m))}</option>`).join("");
    renderPerformance();
    renderGlossary();
    await loadCow();
    $("predecir").disabled = false; $("predecir").textContent = "Predecir";
  } catch (error) {
    $("error").textContent = "No se pudo conectar con el servidor. Córrelo con: python3 src/INTERFAZ/app.py";
  }
})();

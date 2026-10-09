const JOINTS = ["SpineBase", "SpineMid", "Neck", "Head", "ShoulderL", "ElbowL", "WristL", "HandL", "ShoulderR", "ElbowR", "WristR", "HandR", "HipL", "KneeL", "AnkleL", "FootL", "HipR", "KneeR", "AnkleR", "FootR", "SpineShoulder", "HandTipL", "ThumbL", "HandTipR", "ThumbR"];
const EDGES = [[0, 1], [1, 20], [2, 20], [3, 2], [4, 20], [5, 4], [6, 5], [7, 6], [8, 20], [9, 8], [10, 9], [11, 10], [12, 0], [13, 12], [14, 13], [15, 14], [16, 0], [17, 16], [18, 17], [19, 18], [21, 22], [22, 7], [23, 24], [24, 11]];
const SKELETON_LAYOUT = { 3: [260, 35], 2: [260, 70], 20: [260, 110], 1: [260, 160], 0: [260, 205], 4: [215, 112], 5: [185, 145], 6: [165, 180], 7: [150, 210], 21: [138, 224], 22: [160, 224], 8: [305, 112], 9: [335, 145], 10: [355, 180], 11: [370, 210], 23: [360, 224], 24: [382, 224], 12: [232, 250], 13: [225, 295], 14: [218, 338], 15: [212, 382], 16: [288, 250], 17: [295, 295], 18: [302, 338], 19: [308, 382] };
const $ = (id) => document.getElementById(id);
const state = { catalog: null, manifest: null, fidelity: null, semantic: null, semanticDetails: null, bundle: "", heatmap: null };
const mediaState = { mode: "gif", gif: "", png: "" };
const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const formatValue = (value, digits = 3) => Number(value || 0).toFixed(digits);

async function getJson(path) { const response = await fetch(path); if (!response.ok) throw new Error(`Could not load ${path}`); return response.json(); }
function reset(select) { select.replaceChildren(); }
function option(select, value, label = value) { select.add(new Option(label, value)); }
function bundlePath(dataset, run, ensemble) { return `results/${encodeURIComponent(dataset)}/${encodeURIComponent(run)}/${encodeURIComponent(ensemble)}`; }
function selectedSample() { return state.manifest?.samples.find((sample) => sample.sample_id === $("sample-select").value); }
function setupCanvas(canvas, height, requestedWidth = canvas.clientWidth || 600) { const width = Math.max(320, requestedWidth); const ratio = window.devicePixelRatio || 1; canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); return { ctx, width, height }; }
function colorFor(value) { const ratio = Math.max(0, Math.min(1, Number(value) || 0)); return `rgb(255, ${Math.round(244 - ratio * 189)}, ${Math.round(247 - ratio * 152)})`; }
const benchmarkSeries = [
  { label: "Cross-view 60", values: [78.63, 80.38, 81.79, 82.90], color: "#0071e3" },
  { label: "Cross-sub 60", values: [74.90, 76.06, 77.73, 78.38], color: "#34c759" },
  { label: "Cross-set 120", values: [67.03, 69.56, 71.52, 72.36], color: "#ff9500" },
  { label: "Cross-sub 120", values: [65.40, 67.38, 69.25, 70.18], color: "#af52de" },
];
const pipelineDescriptions = {
  jbv: ["JBV single stream", "Uses all nine engineered channels: normalized joints, kinematic bones, and frame-to-frame velocity. It is the base expert used by every reported ensemble.", "2.206 GFLOPs · 412,017 params"],
  "jbv-v": ["JBV + Velocity", "Fuses the complete JBV expert with an independent velocity expert. This emphasizes both posture and motion dynamics.", "4.405 GFLOPs · 823,266 params"],
  "jbv-j": ["JBV + Joints", "Combines JBV with a pure-joint expert. The second stream gives the ensemble an additional coordinate-focused representation.", "4.405 GFLOPs · 823,266 params"],
  "jbv-j-v": ["JBV + Joints + Velocity", "Adds coordinate and motion specialists to the JBV representation without using the bone-only stream.", "6.604 GFLOPs · 1,234,515 params"],
  "jbv-b-v": ["JBV + Bones + Velocity", "Combines full kinematic features with explicit bone geometry and motion dynamics.", "6.604 GFLOPs · 1,234,515 params"],
  full: ["JBV + Bones + Joints + Velocity", "The full four-stream ensemble. Independent experts produce class probabilities that are combined by weighted late fusion.", "8.803 GFLOPs · 1,645,764 params"],
};
const architectureDescriptions = {
  input: ["1. NTU skeleton input", "Each sample begins as a Kinect skeleton sequence with up to two body slots, 100 standardized frames, 25 joints, and three-dimensional coordinates.", "Shape: (bodies=2, frames=100, joints=25, coordinates=3)."],
  features: ["2. Physics-inspired feature engineering", "The loader removes global position with root centering, normalizes subject scale using spine height, then derives bone vectors and frame-to-frame velocity.", "Output: nine channels consisting of joints, bones, and velocity."],
  selector: ["3. Pipeline selector", "The same engineered tensor feeds independent experts. The selector chooses all nine JBV channels or one three-channel specialist representation so each stream can learn a different view of the motion.", "Pipelines: JBV, joints, bones, and velocity."],
  gcn: ["4. Spatial GCN", "The graph layer propagates information along the biological Kinect skeleton. A normalized adjacency matrix provides the human structure, while learnable edge scales adapt the importance of each connection.", "Output: 128 features for every body, frame, and joint."],
  temporal: ["5. Temporal brain", "A global node summarizes each frame after spatial processing. Flex Attention restricts joint interactions to anatomical rooms plus the global node, then local temporal attention connects nearby frames and the two body slots.", "Output: one 128-dimensional video representation per sample."],
  classifier: ["6. Classifier and ensemble output", "A linear classifier maps the video representation to action logits. For multi-stream experiments, each expert produces probabilities and weighted late fusion combines them into the final prediction.", "Output: class probabilities, prediction confidence, and XAI attention maps."],
};
function drawBenchmarkChart() {
  const canvas = $("benchmark-canvas"); if (!canvas) return;
  const { ctx, width, height } = setupCanvas(canvas, 300); const plot = { left: 42, top: 25, right: 12, bottom: 58 }; const yMin = 60, yMax = 85;
  const chartWidth = width - plot.left - plot.right, chartHeight = height - plot.top - plot.bottom;
  const yFor = (value) => plot.top + (yMax - value) / (yMax - yMin) * chartHeight;
  ctx.clearRect(0, 0, width, height); ctx.font = "10px Inter, sans-serif";
  for (let tick = 60; tick <= 85; tick += 5) { const y = yFor(tick); ctx.strokeStyle = "rgba(0,0,0,.07)"; ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(width - plot.right, y); ctx.stroke(); ctx.fillStyle = "#8c8c91"; ctx.fillText(`${tick}%`, 4, y + 4); }
  const groupWidth = chartWidth / benchmarkSeries.length, barWidth = Math.min(13, groupWidth / 6);
  benchmarkSeries.forEach((series, groupIndex) => { series.values.forEach((value, streamIndex) => { const x = plot.left + groupIndex * groupWidth + (groupWidth - barWidth * 4) / 2 + streamIndex * barWidth; const y = yFor(value); ctx.fillStyle = ["#8c8c91", "#66b4ff", "#5ac8fa", series.color][streamIndex]; ctx.fillRect(x, y, barWidth - 2, height - plot.bottom - y); }); ctx.fillStyle = "#6e6e73"; ctx.textAlign = "center"; ctx.fillText(series.label, plot.left + groupIndex * groupWidth + groupWidth / 2, height - 31); });
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; const legendLabels = ["JBV", "2-stream", "3-stream", "4-stream"], legendColors = ["#8c8c91", "#66b4ff", "#5ac8fa", "#0071e3"]; const legendGap = 16; const legendWidth = legendLabels.reduce((total, label) => total + 20 + ctx.measureText(label).width, 0) + legendGap * (legendLabels.length - 1); let legendX = Math.max(plot.left, width - legendWidth - 8); const legendY = height - 10; legendLabels.forEach((label, index) => { ctx.fillStyle = legendColors[index]; ctx.fillRect(legendX, legendY - 4, 8, 8); ctx.fillStyle = "#6e6e73"; ctx.fillText(label, legendX + 12, legendY); legendX += 20 + ctx.measureText(label).width + legendGap; }); ctx.textBaseline = "alphabetic";
}
function selectPipeline(pipeline) { const detail = pipelineDescriptions[pipeline] || pipelineDescriptions.jbv; document.querySelectorAll(".pipeline-row").forEach((row) => row.classList.toggle("selected", row.dataset.pipeline === pipeline)); $("pipeline-detail").innerHTML = `<strong>${detail[0]}</strong><span>${detail[1]}</span><small>${detail[2]}</small>`; }
function selectArchitectureBlock(block) { const detail = architectureDescriptions[block] || architectureDescriptions.input; document.querySelectorAll(".architecture-node").forEach((node) => node.classList.toggle("selected", node.dataset.architecture === block)); $("architecture-detail").innerHTML = `<strong>${detail[0]}</strong><span>${detail[1]}</span><small>${detail[2]}</small>`; }
function updateHeatmapTooltip(event) {
  const canvas = $("heatmap-canvas"), tooltip = $("heatmap-tooltip"), peakFrame = Number(canvas.dataset.peakFrame), frameCount = Number(canvas.dataset.frameCount);
  if (!Number.isInteger(peakFrame) || !frameCount) { tooltip.style.display = "none"; return; }
  const bounds = canvas.getBoundingClientRect(), x = event.clientX - bounds.left, frame = Math.floor(x / (bounds.width / frameCount));
  if (frame !== peakFrame) { tooltip.style.display = "none"; return; }
  tooltip.textContent = `Peak frame ${peakFrame}`;
  tooltip.style.left = `${Math.max(52, Math.min(bounds.width - 52, x))}px`;
  tooltip.style.top = "18px";
  tooltip.style.display = "block";
}
function hideHeatmapTooltip() { $("heatmap-tooltip").style.display = "none"; }

function drawSkeleton(heatmap, body, frame) {
  const canvas = $("skeleton-canvas"); const height = canvas.clientWidth < 500 ? 350 : 430; const { ctx, width } = setupCanvas(canvas, height); ctx.clearRect(0, 0, width, height);
  const values = heatmap.values[body]?.[frame] || []; const scale = Math.min(width / 520, height / 440); const offsetX = (width - 520 * scale) / 2; const offsetY = (height - 440 * scale) / 2;
  const points = Object.fromEntries(Object.entries(SKELETON_LAYOUT).map(([joint, [x, y]]) => [joint, [offsetX + x * scale, offsetY + y * scale]]));
  EDGES.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(...points[a]); ctx.lineTo(...points[b]); ctx.strokeStyle = "rgba(29, 29, 31, .62)"; ctx.lineWidth = Math.max(2.5, 2.8 * scale); ctx.lineCap = "round"; ctx.stroke(); });
  Object.keys(SKELETON_LAYOUT).forEach((joint) => { const [x, y] = points[joint]; const radius = Math.max(6, 8 * scale); ctx.beginPath(); ctx.fillStyle = colorFor(values[Number(joint)]); ctx.shadowColor = Number(values[Number(joint)] || 0) > .55 ? "rgba(255,55,95,.42)" : "transparent"; ctx.shadowBlur = 12; ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0; ctx.strokeStyle = "#3f4652"; ctx.lineWidth = Math.max(1.3, 1.7 * scale); ctx.stroke(); });
}

function drawTemporal(heatmap, body, peakFrame = null) {
  const canvas = $("heatmap-canvas"); const frames = heatmap.values[body] || []; const frameWidth = 25; const labelWidth = 140; const requestedWidth = Math.max(canvas.parentElement.clientWidth, frames.length * frameWidth); const height = 760; canvas.style.width = `${requestedWidth}px`; const { ctx, width } = setupCanvas(canvas, height, requestedWidth); const left = 0, top = 10, right = 10, bottom = 38; const plotWidth = width - left - right, plotHeight = height - top - bottom, cellWidth = plotWidth / Math.max(1, frames.length), cellHeight = plotHeight / 25;
  $("heatmap-labels").innerHTML = JOINTS.map((name, joint) => `<span>${String(joint).padStart(2, "0")} &nbsp; ${name}</span>`).join(""); $("heatmap-labels").style.height = `${height}px`; $("heatmap-labels").style.width = `${labelWidth}px`;
  ctx.clearRect(0, 0, width, height); frames.forEach((frame, frameIndex) => frame.forEach((value, jointIndex) => { ctx.fillStyle = colorFor(value); ctx.fillRect(left + frameIndex * cellWidth, top + jointIndex * cellHeight, cellWidth + .4, cellHeight + .4); }));
  ctx.strokeStyle = "rgba(0,0,0,.07)"; ctx.lineWidth = 1; for (let frame = 0; frame <= frames.length; frame += 5) { const x = left + frame * cellWidth; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotHeight); ctx.stroke(); } for (let joint = 0; joint <= 25; joint += 1) { const y = top + joint * cellHeight; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotWidth, y); ctx.stroke(); }
  if (Number.isInteger(peakFrame) && peakFrame >= 0 && peakFrame < frames.length) { const peakX = left + peakFrame * cellWidth; ctx.fillStyle = "rgba(0,113,227,.14)"; ctx.fillRect(peakX, top, cellWidth, plotHeight); ctx.strokeStyle = "#0071e3"; ctx.lineWidth = 2; ctx.strokeRect(peakX + 1, top + 1, Math.max(1, cellWidth - 2), plotHeight - 2); }
  ctx.fillStyle = "#8c8c91"; ctx.font = "10px Inter, sans-serif"; frames.forEach((_, frame) => { if (frame % 10 !== 0 && frame !== frames.length - 1) return; ctx.fillText(String(frame).padStart(2, "0"), left + (frame + .25) * cellWidth, height - 15); });
  canvas.dataset.peakFrame = Number.isInteger(peakFrame) ? String(peakFrame) : "";
  canvas.dataset.frameCount = String(frames.length);
}

function stat(label, value) { return `<div class="stat"><strong>${value}</strong><small>${label}</small></div>`; }
function drawSemantic() { const semantic = state.semantic; const score = formatPercent(semantic.accuracy); $("semantic-summary").innerHTML = [stat("Pointing accuracy", score), stat("Correct hits", semantic.hits), stat("Samples", semantic.sample_count), stat("Misses", semantic.sample_count - semantic.hits)].join(""); $("validation-score").textContent = score; $("validation-stats").innerHTML = [stat("Pointing accuracy", score), stat("Correct hits", semantic.hits), stat("Samples", semantic.sample_count), stat("Misses", semantic.sample_count - semantic.hits)].join(""); }
function groupLabel(name) { return name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function groupForJoint(jointIndex) {
  const groups = state.semanticDetails?.anatomical_groups || {};
  return Object.entries(groups).find(([, joints]) => joints.includes(Number(jointIndex)))?.[0] || "unknown region";
}
function drawExplore() {
  const semantic = state.semantic || {}, details = state.semanticDetails || {};
  const means = semantic.group_confidence_mean || {};
  $("group-bars").innerHTML = Object.entries(means).sort(([, a], [, b]) => b - a).map(([name, value]) => `<div class="group-bar-row"><div class="group-bar-label"><span>${groupLabel(name)}</span><strong>${formatPercent(value)}</strong></div><div class="group-bar-track"><i style="width:${Math.max(1, value * 100)}%"></i></div></div>`).join("") || `<p class="empty-state">This result does not contain aggregate group confidence data.</p>`;
  $("confidence-definition").textContent = semantic.confidence_definition?.interpretation || "Relative attribution shares summarize how much of the total heatmap mass belongs to each anatomical group.";
  $("explore-run-summary").innerHTML = [["Metric", semantic.metric], ["Samples", semantic.sample_count], ["Hits", semantic.hits], ["Schema", `v${semantic.schema_version || 1}`]].map(([label, value]) => `<div class="detail-row"><span>${label}</span><strong>${value ?? "—"}</strong></div>`).join("");
  reset($("semantic-sample-select"));
  (details.samples || []).forEach((sample) => option($("semantic-sample-select"), sample.sample_id, `${sample.sample_id} · ${sample.action_label || "Unknown action"}`));
  renderSemanticSample();
}
function renderSemanticSample(selectedGroup = null) {
  const details = state.semanticDetails?.samples || [], sample = details.find((item) => item.sample_id === $("semantic-sample-select").value) || details[0];
  if (!sample) { $("semantic-sample-summary").innerHTML = `<p class="empty-state">No semantic detail file is available for this run. Republish the result bundle to include it.</p>`; $("joint-detail").innerHTML = ""; return; }
  $("semantic-sample-select").value = sample.sample_id;
  $("semantic-sample-heading").textContent = `${sample.sample_id} · ${sample.action_label || "Sample attribution"}`;
  const topJoints = Object.entries(sample.joint_confidence || {}).sort(([, a], [, b]) => b - a).slice(0, 3);
  const modelGroup = groupForJoint(sample.peak_joint);
  const targetGroups = sample.target_groups || [];
  const semanticComparison = sample.is_hit
    ? ""
    : `<div class="semantic-comparison"><div><span>Model semantic</span><strong>${groupLabel(modelGroup)}</strong><small>Strongest signal: ${sample.peak_joint_name || "Unknown joint"}</small></div><div><span>Should focus on</span><strong>${targetGroups.length ? targetGroups.map(groupLabel).join(" · ") : "Target region unavailable"}</strong><small>Target-group confidence: ${formatPercent(sample.target_group_confidence)}</small></div></div>`;
  $("semantic-sample-summary").innerHTML = `<div class="sample-result-head"><span class="${sample.is_hit ? "hit" : "miss"}">${sample.is_hit ? "Semantic hit" : "Semantic miss"}</span><strong>${formatPercent(sample.target_group_confidence)}</strong><small>target group confidence</small></div>${semanticComparison}<div class="detail-list"><div class="detail-row"><span>Peak joint</span><strong>${sample.peak_joint_name || "—"}</strong></div><div class="detail-row"><span>Peak body</span><strong>${Number(sample.peak_body) + 1}</strong></div><div class="detail-row"><span>Action index</span><strong>${sample.action_idx}</strong></div></div><h4>Top joints overall</h4>${topJoints.map(([name, value]) => `<div class="mini-bar-row"><span>${name}</span><i><b style="width:${Math.max(1, value * 100)}%"></b></i><strong>${formatPercent(value)}</strong></div>`).join("")}`;
  const withinGroups = sample.joint_confidence_within_group || {};
  const groupConfidence = sample.group_confidence || {};
  const jointGroups = Object.entries(withinGroups).sort(([groupA], [groupB]) => (groupConfidence[groupB] || 0) - (groupConfidence[groupA] || 0));
  const activeGroup = selectedGroup && withinGroups[selectedGroup] ? selectedGroup : null;
  const groupRows = jointGroups.map(([group, joints]) => `<button class="group-attribution-row${group === activeGroup ? " selected" : ""}" type="button" data-group="${group}"><span>${groupLabel(group)}</span><i><b style="width:${Math.max(1, (groupConfidence[group] || 0) * 100)}%"></b></i><strong>${formatPercent(groupConfidence[group] || 0)}</strong></button>`).join("");
  const activeJoints = activeGroup ? withinGroups[activeGroup] : {};
  const jointAttribution = activeGroup ? `<h4>Joint attribution · ${groupLabel(activeGroup)}</h4><div class="joint-group-rows">${Object.entries(activeJoints).sort(([, a], [, b]) => b - a).map(([joint, value]) => `<div class="joint-row"><span>${joint}</span><i><b style="width:${Math.max(1, value * 100)}%"></b></i><strong>${formatPercent(value)}</strong></div>`).join("") || `<p class="empty-state">No joint attribution data is available.</p>`}</div>` : "";
  $("joint-detail").innerHTML = `<h4>Group attribution</h4><p class="joint-detail-copy">Select a group to inspect the joint attribution within it.</p><div class="group-attribution-list">${groupRows || `<p class="empty-state">No group attribution data is available.</p>`}</div>${jointAttribution}`;
  $("joint-detail").querySelectorAll("[data-group]").forEach((button) => { button.onclick = () => renderSemanticSample(button.dataset.group === activeGroup ? null : button.dataset.group); });
}
function populateExploreSelectors() {
  const dataset = $("dataset-select").value, run = $("run-select").value, ensemble = $("ensemble-select").value;
  reset($("explore-dataset-select")); state.catalog.datasets.forEach((item) => option($("explore-dataset-select"), item.dataset)); $("explore-dataset-select").value = dataset;
  reset($("explore-run-select")); state.catalog.datasets.find((item) => item.dataset === dataset).runs.forEach((item) => option($("explore-run-select"), item.run_id)); $("explore-run-select").value = run;
  reset($("explore-ensemble-select")); state.catalog.datasets.find((item) => item.dataset === dataset).runs.find((item) => item.run_id === run).ensembles.forEach((item) => option($("explore-ensemble-select"), item)); $("explore-ensemble-select").value = ensemble;
}
function exploreBundleChanged() {
  const dataset = state.catalog.datasets.find((item) => item.dataset === $("explore-dataset-select").value);
  const run = dataset?.runs.find((item) => item.run_id === $("explore-run-select").value);
  const ensemble = $("explore-ensemble-select").value;
  if (!dataset || !run || !run.ensembles.includes(ensemble)) {
    showError(new Error("The selected semantic result bundle is unavailable."));
    return;
  }

  $("dataset-select").value = dataset.dataset;
  reset($("run-select"));
  dataset.runs.forEach((item) => option($("run-select"), item.run_id));
  $("run-select").value = run.run_id;
  reset($("ensemble-select"));
  run.ensembles.forEach((item) => option($("ensemble-select"), item));
  $("ensemble-select").value = ensemble;
  loadBundle().catch(showError);
}
function drawFidelitySeries(canvasId, ks, values, color) { const canvas = $(canvasId); const { ctx, width, height } = setupCanvas(canvas, 260); const plot = { left: 66, top: 25, right: 16, bottom: 52 }; const xFor = (i) => plot.left + i * ((width - plot.left - plot.right) / Math.max(1, ks.length - 1)); const yFor = (v) => height - plot.bottom - Math.max(0, Math.min(1, v)) * (height - plot.top - plot.bottom); ctx.clearRect(0, 0, width, height); ctx.font = "10px Inter, sans-serif"; for (let tick = 0; tick <= 4; tick += 1) { const y = yFor(tick / 4); ctx.strokeStyle = "rgba(0,0,0,.07)"; ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(width - plot.right, y); ctx.stroke(); ctx.fillStyle = "#8c8c91"; ctx.fillText((tick / 4).toFixed(2), 25, y + 4); } ctx.beginPath(); values.forEach((value, index) => { const x = xFor(index), y = yFor(value); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke(); values.forEach((value, index) => { const x = xFor(index), y = yFor(value); ctx.beginPath(); ctx.fillStyle = color; ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = "#8c8c91"; ctx.fillText(String(ks[index]), x - 4, height - 30); }); ctx.fillStyle = "#6e6e73"; ctx.textAlign = "center"; ctx.fillText("Number of joints (k)", (plot.left + width - plot.right) / 2, height - 8); ctx.save(); ctx.translate(12, (plot.top + height - plot.bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("True-class confidence", 0, 0); ctx.restore(); ctx.textAlign = "start"; }
function drawFidelity() { const aggregate = state.fidelity.aggregate, ks = state.fidelity.k_values, base = aggregate.base_confidence_mean; const deletion = [base].concat(ks.map((k) => base - aggregate.deletion_drop_mean[String(k)])); const insertion = [0].concat(ks.map((k) => aggregate.insertion_retention_mean[String(k)] * base)); $("deletion-title").textContent = "Aggregate deletion"; $("insertion-title").textContent = "Aggregate insertion"; drawFidelitySeries("deletion-canvas", [0].concat(ks), deletion, "#ff375f"); drawFidelitySeries("insertion-canvas", [0].concat(ks), insertion, "#34c759"); }
function showMedia(mode) { mediaState.mode = mode; const gif = mode === "gif"; $("gif-image").style.display = gif ? "block" : "none"; $("static-image").style.display = gif ? "none" : "block"; $("gif-toggle").classList.toggle("active", gif); $("png-toggle").classList.toggle("active", !gif); $("media-open-link").href = gif ? mediaState.gif : mediaState.png; }
function setView(view) { document.querySelectorAll(".side-nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view)); document.querySelectorAll(".view-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `${view}-view`)); document.body.classList.toggle("explore-active", view === "explore"); document.body.classList.toggle("home-active", view === "home"); $("breadcrumb-view").textContent = view[0].toUpperCase() + view.slice(1); }
function populateSample(sample) { const detail = state.semanticDetails?.samples?.find((item) => item.sample_id === sample.sample_id); const predictedAction = sample.predicted_action || "Prediction unavailable"; const trueAction = sample.true_action || detail?.action_label || "Unknown action"; const actionIndex = sample.true_label ?? detail?.action_idx ?? sample.predicted_label; const classLabel = actionIndex === undefined || actionIndex === null ? "Unavailable" : `Class ${Number(actionIndex)}`; const hasConfidence = Number.isFinite(Number(sample.predicted_confidence)); const confidence = hasConfidence ? sample.predicted_confidence : 0; $("sample-title").textContent = sample.sample_id; $("confidence-badge").textContent = hasConfidence ? `${formatPercent(confidence)} confidence` : "Confidence unavailable"; $("prediction-content").innerHTML = `<p class="prediction-label">Predicted action</p><p class="prediction-name">${predictedAction}</p><p class="prediction-meta">True label: <strong>${trueAction}</strong> · ${classLabel}</p>`; $("insight-content").innerHTML = `<div class="insight-row"><span>Prediction</span><strong>${predictedAction}</strong></div><div class="insight-row"><span>Ground truth</span><strong>${trueAction}</strong></div><div class="insight-row"><span>Class</span><strong>${classLabel}</strong></div><div class="insight-row"><span>Confidence</span><strong>${hasConfidence ? formatPercent(confidence) : "Unavailable"}</strong></div><div class="insight-row"><span>Heatmap shape</span><strong>${(sample.heatmap_shape || []).join(" × ") || "Unavailable"}</strong></div>`; }
async function loadSample() { const sample = selectedSample(); if (!sample) return; state.heatmap = await getJson(`${state.bundle}/${sample.heatmap}`); const body = Number($("body-select").value); const frames = state.heatmap.values[body] || []; const frame = frames.reduce((best, current, index) => { const value = current.reduce((sum, item) => sum + item, 0); return value > best.value ? { index, value } : best; }, { index: 0, value: -Infinity }).index; $("frame-label").textContent = `Peak frame ${frame}`; $("frame-badge").textContent = `Peak frame ${frame}`; populateSample(sample); drawSkeleton(state.heatmap, body, frame); drawTemporal(state.heatmap, body, frame); mediaState.gif = `${state.bundle}/${sample.gif}`; mediaState.png = `${state.bundle}/${sample.static_image}`; $("gif-image").src = mediaState.gif; $("static-image").src = mediaState.png; showMedia(mediaState.mode); }
async function loadBundle() { const dataset = $("dataset-select").value, run = $("run-select").value, ensemble = $("ensemble-select").value; state.bundle = bundlePath(dataset, run, ensemble); const detailsPath = `${state.bundle}/semantic/semantic_details.json`; [state.manifest, state.fidelity, state.semantic] = await Promise.all([getJson(`${state.bundle}/explanation_manifest.json`), getJson(`${state.bundle}/fidelity/fidelity_results.json`), getJson(`${state.bundle}/semantic/semantic_results.json`)]); state.semanticDetails = await getJson(detailsPath).catch(() => null); reset($("sample-select")); state.manifest.samples.forEach((sample) => { const detail = state.semanticDetails?.samples?.find((item) => item.sample_id === sample.sample_id); const activity = sample.true_action || sample.predicted_action || detail?.action_label || "Unknown activity"; option($("sample-select"), sample.sample_id, `${activity} · ${sample.sample_id}`); }); $("dataset-badge").textContent = dataset.toUpperCase(); $("hero-experiment").textContent = `${dataset} · ${ensemble}`; drawSemantic(); drawExplore(); populateExploreSelectors(); $("experiment-details").innerHTML = `<div class="detail-row"><span>Dataset</span><strong>${dataset}</strong></div><div class="detail-row"><span>Run</span><strong>${run}</strong></div><div class="detail-row"><span>Ensemble</span><strong>${ensemble}</strong></div><div class="detail-row"><span>Samples evaluated</span><strong>${state.manifest.samples.length}</strong></div>`; drawFidelity(); await loadSample(); }
function updateRuns() { const dataset = state.catalog.datasets.find((item) => item.dataset === $("dataset-select").value); reset($("run-select")); dataset.runs.forEach((run) => option($("run-select"), run.run_id)); updateEnsembles(); }
function updateEnsembles() { const dataset = state.catalog.datasets.find((item) => item.dataset === $("dataset-select").value), run = dataset.runs.find((item) => item.run_id === $("run-select").value); reset($("ensemble-select")); run.ensembles.forEach((ensemble) => option($("ensemble-select"), ensemble)); loadBundle().catch(showError); }
function showError(error) { $("bundle-status").textContent = "Unable to load results"; $("sidebar-status").textContent = "Check result bundle"; $("toast").textContent = error.message; $("toast").classList.add("show"); setTimeout(() => $("toast").classList.remove("show"), 4000); }
async function start() { try { state.catalog = await getJson("results/catalog.json"); reset($("dataset-select")); state.catalog.datasets.forEach((dataset) => option($("dataset-select"), dataset.dataset)); $("dataset-select").onchange = updateRuns; $("run-select").onchange = updateEnsembles; $("ensemble-select").onchange = () => loadBundle().catch(showError); $("sample-select").onchange = () => loadSample().catch(showError); $("body-select").onchange = () => loadSample().catch(showError); $("semantic-sample-select").onchange = renderSemanticSample; $("explore-dataset-select").onchange = populateExploreSelectors; $("explore-run-select").onchange = populateExploreSelectors; $("explore-ensemble-select").onchange = exploreBundleChanged; $("open-explore-button").onclick = () => setView("explore"); $("back-validation-button").onclick = () => setView("validation"); $("refresh-button").onclick = () => loadBundle().then(() => { $("bundle-status").textContent = "Results refreshed"; }).catch(showError); document.querySelectorAll(".side-nav-item").forEach((item) => item.onclick = () => setView(item.dataset.view)); document.querySelectorAll(".pipeline-row").forEach((row) => row.onclick = () => selectPipeline(row.dataset.pipeline)); document.querySelectorAll(".architecture-node").forEach((node) => node.onclick = () => selectArchitectureBlock(node.dataset.architecture)); $("heatmap-canvas").addEventListener("mousemove", updateHeatmapTooltip); $("heatmap-canvas").addEventListener("mouseleave", hideHeatmapTooltip); $("gif-toggle").onclick = () => showMedia("gif"); $("png-toggle").onclick = () => showMedia("png"); updateRuns(); selectPipeline("jbv"); selectArchitectureBlock("input"); drawBenchmarkChart(); setView("home"); $("bundle-status").textContent = "Results loaded"; $("sidebar-status").textContent = `${state.manifest?.samples.length || 0} samples available`; } catch (error) { showError(error); } }
start();

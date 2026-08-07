#ifndef SCULPTOR_MLIR_TOOLS_RA_TREE_REPORT_HTML_H
#define SCULPTOR_MLIR_TOOLS_RA_TREE_REPORT_HTML_H

#include "llvm/ADT/StringRef.h"

inline constexpr llvm::StringLiteral kRATreeReportHTML = R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes, viewport-fit=cover">
  <title>Sculptor RA Tree</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18202b;
      --muted: #647080;
      --line: #c8ced6;
      --line-strong: #8993a1;
      --page: #eef1f4;
      --panel: #ffffff;
      --panel-alt: #f7f8fa;
      --temporal: #a85d13;
      --temporal-soft: #f6e5d2;
      --spatial: #2669a7;
      --spatial-soft: #dceafb;
      --dependency: #6d3aa8;
      --selected: #111827;
      --success: #237052;
    }

    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
      -webkit-text-size-adjust: 100%;
    }
    h1, h2, h3, p { margin: 0; }
    button, input, select { font: inherit; letter-spacing: 0; }
    button, input, select { border-radius: 0; }

    .masthead {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: end;
      width: min(1800px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 16px;
      border-bottom: 2px solid var(--ink);
    }
    .eyebrow {
      color: var(--muted);
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
    }
    h1 {
      margin-top: 4px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 38px;
      font-weight: 500;
      line-height: 1.05;
    }
    .subtitle {
      max-width: 1000px;
      margin-top: 7px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .function-control { min-width: 230px; }
    .control-label {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      text-transform: uppercase;
    }
    select, input[type="search"] {
      height: 36px;
      border: 1px solid var(--line-strong);
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
    }
    select:focus-visible, input:focus-visible, button:focus-visible {
      outline: 2px solid var(--spatial);
      outline-offset: 2px;
    }

    main {
      width: min(1800px, calc(100% - 32px));
      margin: 0 auto;
      padding: 16px 0 28px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 1px;
      border: 1px solid var(--line);
      background: var(--line);
    }
    .metric {
      min-height: 82px;
      padding: 13px 14px;
      background: var(--panel);
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 7px;
      font-size: 21px;
      font-variant-numeric: tabular-nums;
    }
    .metric.fingerprint strong {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .baseline-note {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      margin-top: 12px;
      padding: 10px 13px;
      border: 1px solid #d3aa7d;
      background: var(--temporal-soft);
      color: #6e3b0d;
      font-size: 12px;
    }
    .baseline-note strong { font-weight: 750; }

    .plan-summary {
      margin-top: 12px;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .plan-summary-heading {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      padding: 10px 13px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-alt);
      font-size: 12px;
    }
    .plan-summary-heading strong { font-weight: 750; }
    .plan-summary-heading span { color: var(--muted); }
    .plan-table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .plan-table th, .plan-table td {
      padding: 8px 12px;
      border-bottom: 1px solid #e4e7eb;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .plan-table th:first-child, .plan-table td:first-child,
    .plan-table th:last-child, .plan-table td:last-child { text-align: left; }
    .plan-table th { color: var(--muted); background: #fafbfc; }
    .plan-table tr:last-child td { border-bottom: 0; }
    .plan-table tr.selected { background: #edf7f2; }
    .decision-selected { color: var(--success); font-weight: 750; }
    .decision-rejected { color: var(--muted); }

    .workbench {
      margin-top: 12px;
      border: 1px solid var(--line);
      background: var(--panel);
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      min-height: 58px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-alt);
    }
    .toolbar-group { display: flex; align-items: end; gap: 8px; }
    .segmented { display: grid; grid-auto-flow: column; }
    button {
      height: 36px;
      border: 1px solid var(--ink);
      background: #fff;
      color: var(--ink);
      padding: 0 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: #e9edf1; }
    button:disabled { cursor: not-allowed; opacity: .42; }
    button.toggle-active { background: var(--ink); color: #fff; }
    .segmented button { border-right-width: 0; }
    .segmented button:last-child { border-right-width: 1px; }
    .segmented button.active { background: var(--ink); color: #fff; }
    .icon-button { width: 38px; padding: 0; font-size: 17px; }
    .search-control { width: min(340px, 34vw); }
    .search-control input { width: 100%; }
    .block-control { width: 190px; }
    .block-control select { width: 100%; }

    .view-explanation {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      min-height: 42px;
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }
    .view-explanation strong { color: var(--ink); }
    .view-explanation span:last-child { text-align: right; }

    .annealing-progress {
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .annealing-progress-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 11px 12px 8px;
    }
    .annealing-progress-title {
      display: flex;
      flex-direction: column;
      gap: 3px;
      min-width: 190px;
    }
    .annealing-progress-title strong {
      font-family: Georgia, "Times New Roman", serif;
      font-size: 20px;
      font-weight: 500;
      font-variant-numeric: tabular-nums;
    }
    .annealing-stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(100px, 1fr));
      flex: 1;
      gap: 1px;
      margin: 0;
      background: var(--line);
      border: 1px solid var(--line);
    }
    .annealing-stat {
      padding: 7px 9px;
      background: var(--panel-alt);
    }
    .annealing-stat dt {
      color: var(--muted);
      font-size: 8px;
      font-weight: 750;
      text-transform: uppercase;
    }
    .annealing-stat dd {
      margin: 3px 0 0;
      color: var(--ink);
      font-size: 11px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .annealing-chart-wrap {
      height: 170px;
      padding: 0 12px;
    }
    #annealing-chart { width: 100%; height: 100%; display: block; }
    .annealing-controls {
      display: grid;
      grid-template-columns: auto minmax(120px, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 6px 12px 10px;
    }
    #annealing-slider { width: 100%; accent-color: #2f7c61; }
    .annealing-legend {
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 9px;
      white-space: nowrap;
    }
    .annealing-legend i {
      display: inline-block;
      width: 16px;
      height: 2px;
      margin-right: 4px;
      vertical-align: middle;
    }
    .annealing-legend .candidate { background: #aab0b8; }
    .annealing-legend .current { background: #2f6ea5; }
    .annealing-legend .best { background: #2f7c61; }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      min-height: 730px;
    }
    .canvas-panel {
      position: relative;
      min-width: 0;
      border-right: 1px solid var(--line);
      overflow: hidden;
      overscroll-behavior: contain;
      background-color: #fff;
      background-image:
        linear-gradient(#eef1f4 1px, transparent 1px),
        linear-gradient(90deg, #eef1f4 1px, transparent 1px);
      background-size: 24px 24px;
    }
    #graph-canvas {
      display: block;
      width: 100%;
      height: 730px;
      cursor: grab;
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
      -webkit-touch-callout: none;
    }
    #graph-canvas.dragging { cursor: grabbing; }
    .canvas-status {
      position: absolute;
      left: 10px;
      bottom: 10px;
      padding: 6px 8px;
      border: 1px solid var(--line-strong);
      background: rgba(255, 255, 255, .94);
      color: var(--muted);
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      pointer-events: none;
    }
    .tooltip {
      position: absolute;
      z-index: 5;
      width: 260px;
      padding: 9px 10px;
      border: 1px solid var(--ink);
      background: rgba(255, 255, 255, .97);
      box-shadow: 4px 4px 0 rgba(24, 32, 43, .12);
      pointer-events: none;
    }
    .tooltip strong, .tooltip span { display: block; }
    .tooltip strong { font-size: 12px; }
    .tooltip span {
      margin-top: 3px;
      color: var(--muted);
      font-size: 10px;
      overflow-wrap: anywhere;
    }

    .inspector {
      min-width: 0;
      background: var(--panel);
    }
    .inspector-heading {
      min-height: 69px;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-alt);
    }
    .inspector-heading h2 {
      margin-top: 4px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 20px;
      font-weight: 500;
    }
    .inspector-body {
      height: 660px;
      overflow-y: auto;
      padding-bottom: 18px;
    }
    .detail-section { border-bottom: 1px solid var(--line); }
    .detail-section h3 {
      padding: 10px 13px;
      background: #fafbfc;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 6px 10px;
      padding: 11px 13px;
      font-size: 11px;
    }
    .detail-grid dt { color: var(--muted); }
    .detail-grid dd {
      margin: 0;
      text-align: right;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }
    .tag-list { display: flex; flex-wrap: wrap; gap: 5px; padding: 11px 13px; }
    .tag {
      padding: 4px 6px;
      border: 1px solid var(--line-strong);
      background: #fff;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 9px;
    }
    .compact-table { width: 100%; border-collapse: collapse; font-size: 10px; }
    .compact-table th, .compact-table td {
      padding: 7px 9px;
      border-bottom: 1px solid #e4e7eb;
      text-align: left;
      font-variant-numeric: tabular-nums;
    }
    .compact-table th { color: var(--muted); background: #fafbfc; }
    pre {
      max-height: 250px;
      margin: 0;
      padding: 11px 13px;
      overflow: auto;
      background: #121923;
      color: #e5eaf0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      padding: 9px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 10px;
    }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 12px; height: 12px; border: 1px solid rgba(0, 0, 0, .18); }
    .swatch.temporal { background: var(--temporal); }
    .swatch.spatial { background: var(--spatial); }
    .swatch.leaf { background: #4c8a68; }
    .swatch.dependency { background: var(--dependency); }
    .swatch.setup { background: #245f8f; }
    .swatch.vector-tile { background: #2f7c61; }
    .swatch.array-mvm { background: #7c3aed; }
    .swatch.recombine { background: #a85d13; }
    .swatch.physical-tile { background: #2f6ea5; }
    .swatch.tile-communication { background: #6d3aa8; }

    footer {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      width: min(1800px, calc(100% - 32px));
      margin: 0 auto;
      padding: 0 0 24px;
      color: var(--muted);
      font-size: 10px;
    }

    @media (max-width: 1100px) {
      .metrics { grid-template-columns: repeat(3, 1fr); }
      .workspace { grid-template-columns: 1fr; }
      .canvas-panel { border-right: 0; border-bottom: 1px solid var(--line); }
      .inspector-body { height: auto; max-height: 620px; }
    }
    @media (max-width: 720px) {
      .masthead, main, footer { width: calc(100% - 16px); }
      .masthead { grid-template-columns: 1fr; }
      .function-control { width: 100%; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .baseline-note { flex-direction: column; }
      .plan-summary-heading { flex-direction: column; }
      .plan-table thead { display: none; }
      .plan-table, .plan-table tbody, .plan-table tr { display: block; }
      .plan-table tr {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        border-bottom: 1px solid var(--line);
      }
      .plan-table tr:last-child { border-bottom: 0; }
      .plan-table td {
        min-width: 0;
        padding: 8px 10px;
        border-bottom: 0;
        text-align: left;
        overflow-wrap: anywhere;
      }
      .plan-table td::before {
        display: block;
        margin-bottom: 3px;
        color: var(--muted);
        font-size: 9px;
        font-weight: 750;
        text-transform: uppercase;
      }
      .plan-table td:nth-child(1)::before { content: "Candidate"; }
      .plan-table td:nth-child(2)::before { content: "Latency"; }
      .plan-table td:nth-child(3)::before { content: "Crossing bytes"; }
      .plan-table td:nth-child(4)::before { content: "Communication"; }
      .plan-table td:nth-child(5)::before { content: "Resources"; }
      .plan-table td:nth-child(6)::before { content: "Decision"; }
      .toolbar, .toolbar-group { align-items: stretch; flex-direction: column; }
      button, select, input[type="search"] { min-height: 44px; }
      .icon-button { width: 44px; }
      .segmented { grid-auto-flow: row; grid-template-columns: repeat(4, 1fr); }
      .segmented button { border-right-width: 0; }
      .segmented button:last-child { border-right-width: 1px; }
      .view-explanation { flex-direction: column; }
      .view-explanation span:last-child { text-align: left; }
      .annealing-progress-header { align-items: stretch; flex-direction: column; }
      .annealing-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .annealing-controls { grid-template-columns: auto 1fr; }
      .annealing-legend { grid-column: 1 / -1; flex-wrap: wrap; }
      .search-control, .block-control { width: 100%; }
      #graph-canvas {
        height: 580px;
        height: min(68svh, 620px);
        min-height: 420px;
      }
      .workspace { min-height: 420px; }
    }
  </style>
</head>
<body>
  <header class="masthead">
    <div>
      <span class="eyebrow">Sculptor MLIR resource allocation</span>
      <h1 id="report-title">RA Tree</h1>
      <p id="report-source" class="subtitle"></p>
    </div>
    <label id="function-control" class="function-control">
      <span class="control-label">Function</span>
      <select id="function-select"></select>
    </label>
  </header>

  <main>
    <section class="metrics" aria-label="RA-tree summary">
      <div class="metric"><span>Tree nodes</span><strong id="metric-nodes">-</strong></div>
      <div class="metric"><span>Tree depth</span><strong id="metric-depth">-</strong></div>
      <div class="metric"><span>Compute leaves / overlay stages</span><strong id="metric-operations">-</strong></div>
      <div class="metric"><span>Tensors</span><strong id="metric-tensors">-</strong></div>
      <div class="metric"><span>DAG edges</span><strong id="metric-edges">-</strong></div>
      <div id="metric-placement" class="metric" hidden>
        <span id="metric-placement-label">Placement graph score</span>
        <strong id="metric-placement-score">-</strong>
      </div>
      <div class="metric fingerprint"><span>Graph fingerprint</span><strong id="metric-fingerprint">-</strong></div>
    </section>

    <section id="plan-summary" class="plan-summary" hidden>
      <div class="plan-summary-heading">
        <strong id="plan-summary-title">Selected mapping plan</strong>
        <span id="plan-summary-detail"></span>
      </div>
      <table class="plan-table">
        <thead>
          <tr><th>Candidate</th><th>Latency</th><th>Crossing bytes</th><th>Communication</th><th>Resources</th><th>Decision</th></tr>
        </thead>
        <tbody id="plan-candidates"></tbody>
      </table>
    </section>

    <section class="baseline-note">
      <strong id="tree-profile">Tree profile</strong>
      <span id="tree-profile-detail"></span>
    </section>

    <section class="workbench">
      <div class="toolbar">
        <div class="toolbar-group">
          <div class="segmented" aria-label="Graph view">
            <button type="button" class="active" data-view="tree">RA hierarchy</button>
            <button type="button" data-view="dag">Compute DAG</button>
            <button type="button" data-view="st">S-T graph</button>
            <button type="button" data-view="physical">Physical mesh</button>
          </div>
          <label class="block-control">
            <span class="control-label">Semantic block</span>
            <select id="block-select"></select>
          </label>
          <button id="toggle-setup-fill" class="toggle-active" type="button"
                  aria-pressed="true" title="Show or hide matrix setup and linalg.fill operations">
            Setup + Fill: shown
          </button>
        </div>
        <div class="toolbar-group">
          <label class="search-control">
            <span class="control-label">Operation search</span>
            <input id="operation-search" type="search" placeholder="ID, operation, section, or name">
          </label>
          <button id="zoom-out" class="icon-button" type="button" title="Zoom out" aria-label="Zoom out">−</button>
          <button id="zoom-in" class="icon-button" type="button" title="Zoom in" aria-label="Zoom in">+</button>
          <button id="fit-view" type="button">Fit</button>
        </div>
      </div>

      <section id="annealing-progress" class="annealing-progress" hidden>
        <div class="annealing-progress-header">
          <div class="annealing-progress-title">
            <span class="eyebrow">Annealing trajectory · log score axis</span>
            <strong id="annealing-live-score">-</strong>
          </div>
          <dl class="annealing-stats">
            <div class="annealing-stat"><dt>Iteration</dt><dd id="annealing-iteration">-</dd></div>
            <div class="annealing-stat"><dt>Candidate</dt><dd id="annealing-candidate">-</dd></div>
            <div class="annealing-stat"><dt>Current</dt><dd id="annealing-current">-</dd></div>
            <div class="annealing-stat"><dt>Improvement</dt><dd id="annealing-improvement">-</dd></div>
          </dl>
        </div>
        <div class="annealing-chart-wrap">
          <canvas id="annealing-chart" aria-label="Annealing score trajectory"></canvas>
        </div>
        <div class="annealing-controls">
          <button id="annealing-replay" type="button">Replay</button>
          <input id="annealing-slider" type="range" min="0" max="0" value="0"
                 aria-label="Annealing iteration">
          <div class="annealing-legend">
            <span><i class="candidate"></i>Candidate</span>
            <span><i class="current"></i>Current</span>
            <span><i class="best"></i>Best</span>
          </div>
        </div>
      </section>

      <div class="view-explanation">
        <strong id="view-title">RA hierarchy</strong>
        <span id="view-detail">Cut nodes show how the mapping hierarchy owns operation leaves.</span>
      </div>

      <div class="workspace">
        <div class="canvas-panel">
          <canvas id="graph-canvas" aria-label="Interactive mapping visualization"></canvas>
          <div id="graph-tooltip" class="tooltip" hidden></div>
          <div id="canvas-status" class="canvas-status"></div>
        </div>
        <aside class="inspector">
          <div class="inspector-heading">
            <span class="eyebrow">Selection</span>
            <h2 id="inspector-title">RA-tree root</h2>
          </div>
          <div id="inspector-body" class="inspector-body"></div>
        </aside>
      </div>

      <div class="legend">
        <span><i class="swatch temporal"></i>Temporal cut</span>
        <span><i class="swatch spatial"></i>Spatial cut</span>
        <span><i class="swatch leaf"></i>Operation leaf</span>
        <span><i class="swatch dependency"></i>Tensor dependency</span>
        <span><i class="swatch setup"></i>Matrix setup</span>
        <span><i class="swatch vector-tile"></i>Vector tile</span>
        <span><i class="swatch array-mvm"></i>Array MVM</span>
        <span><i class="swatch recombine"></i>Recombine</span>
        <span><i class="swatch physical-tile"></i>Placed logical tile</span>
        <span><i class="swatch tile-communication"></i>Inter-tile communication</span>
      </div>
    </section>
  </main>

  <footer>
    <span>Validated against the current SSA compute graph before export.</span>
    <span id="schema-version"></span>
  </footer>

  <script id="report-data" type="application/json">__SCULPTOR_RA_TREE_REPORT_DATA__</script>
  <script>
    "use strict";

    const report = JSON.parse(document.querySelector("#report-data").textContent);
    const canvas = document.querySelector("#graph-canvas");
    const context = canvas.getContext("2d");
    const tooltip = document.querySelector("#graph-tooltip");
    const palette = ["#2f6ea5", "#2f7c61", "#a15339", "#7050a0", "#287f89", "#a13f62", "#687348", "#8a642a"];

    let currentFunction = null;
    const requestedView = new URLSearchParams(window.location.search).get("view");
    let currentView = ["tree", "dag", "st", "physical"].includes(requestedView)
      ? requestedView
      : "tree";
    let blockFilter = "all";
    let showSetupAndFill = true;
    let layout = {nodes: [], links: [], bounds: {minX: 0, minY: 0, maxX: 1, maxY: 1}};
    let renderedLayout = layout;
    let viewport = {x: 0, y: 0, scale: 1};
    let selectedNode = null;
    let hoveredNode = null;
    let dragging = null;
    let annealingFrame = 0;
    let annealingTimer = null;
    const activePointers = new Map();
    let pinchGesture = null;

    const functionSelect = document.querySelector("#function-select");
    const blockSelect = document.querySelector("#block-select");
    const searchInput = document.querySelector("#operation-search");
    const setupFillButton = document.querySelector("#toggle-setup-fill");
    const annealingProgress = document.querySelector("#annealing-progress");
    const annealingChart = document.querySelector("#annealing-chart");
    const annealingSlider = document.querySelector("#annealing-slider");
    const annealingReplay = document.querySelector("#annealing-replay");

    function escapeHTML(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function formatBytes(value, unknown = false) {
      if (unknown && value === 0) return "dynamic";
      if (!Number.isFinite(value) || value < 0) return "dynamic";
      if (value < 1024) return `${value} B`;
      if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
      return `${(value / (1024 * 1024)).toFixed(2)} MiB`;
    }

    function formatNanoseconds(value) {
      if (!Number.isFinite(value)) return "-";
      if (value < 1000) return `${value.toFixed(2)} ns`;
      if (value < 1e6) return `${(value / 1000).toFixed(2)} µs`;
      return `${(value / 1e6).toFixed(2)} ms`;
    }

    function semanticBlock(operation) {
      const value = operation?.semantic?.block_index;
      return Number.isInteger(value) ? value : null;
    }

    function isSetupOrFillOperation(operation) {
      if (!operation) return false;
      return operation.kind === "matrix_setup" ||
        operation.name.toLowerCase() === "linalg.fill";
    }

    function isSuppressedOperation(operation) {
      return !showSetupAndFill && isSetupOrFillOperation(operation);
    }

    function isSuppressedStage(stage) {
      return !showSetupAndFill && stage?.kind === "sculptor.matrix_setup";
    }

    function visibleRealizationStages(func, operationId) {
      return (func.realizationByOperation.get(operationId) || [])
        .filter(stage => !isSuppressedStage(stage));
    }

    function physicalTileColor(tile) {
      const maximum = Math.max(1, currentFunction?.maximumDigitalWork || 1);
      const intensity = Math.sqrt(Math.max(0, tile.digital_work) / maximum);
      const lightness = 94 - intensity * 42;
      return `hsl(207 28% ${lightness}%)`;
    }

    function nodeColor(node) {
      if (node.aggregate) return node.aggregateColor;
      if (node.logicalTile) return physicalTileColor(node.logicalTile);
      if (node.kind === "temporal_cut") return "#a85d13";
      if (node.kind === "spatial_cut") return "#2669a7";
      if (node.realization) {
        if (node.realization.kind === "sculptor.matrix_setup") return "#245f8f";
        if (node.realization.kind === "digital.vector_tile") return "#2f7c61";
        if (node.realization.kind === "sculptor.mvm") return "#7c3aed";
        if (node.realization.kind === "digital.tile_recombine") return "#a85d13";
        return "#4c5969";
      }
      if (node.operation?.kind === "matrix_setup") return "#245f8f";
      if (node.operation?.kind === "vector_tile") return "#2f7c61";
      if (node.operation?.kind === "analog_mvm" ||
          node.operation?.kind === "physical_mvm") return "#7c3aed";
      if (node.operation?.kind === "tile_recombine") return "#a85d13";
      const block = semanticBlock(node.operation);
      return block === null ? "#4c8a68" : palette[Math.abs(block) % palette.length];
    }

    function nodeVisible(node) {
      if (node.aggregate) return true;
      if (node.logicalTile) return true;
      if (blockFilter === "all" || !node.operation) return true;
      return String(semanticBlock(node.operation)) === blockFilter;
    }

    function mapFunctionData(func) {
      func.operationMap = new Map(func.operations.map(operation => [operation.id, operation]));
      func.tensorMap = new Map(func.tensors.map(tensor => [tensor.id, tensor]));
      func.mvmWaves = func.mvm_waves || [];
      func.mvmWaveMap = new Map(func.mvmWaves.map(wave => [wave.id, wave]));
      func.treeNodeMap = new Map(func.tree.nodes.map(node => [node.id, node]));
      func.workUnitMap = new Map(
        (func.tree.work_units || []).map(workUnit => [workUnit.id, workUnit])
      );
      func.workUnitEdges = func.tree.work_unit_edges || [];
      func.leavesByOperation = new Map();
      for (const node of func.tree.nodes) {
        if (node.operation_id < 0) continue;
        if (!func.leavesByOperation.has(node.operation_id))
          func.leavesByOperation.set(node.operation_id, []);
        func.leavesByOperation.get(node.operation_id).push(node);
      }
      func.leafByOperation = new Map(
        [...func.leavesByOperation].map(([operationId, leaves]) =>
          [operationId, leaves.length === 1 ? leaves[0] : null]
        )
      );
      func.planNodeEvaluationMap = new Map(
        (func.plan?.node_evaluations || []).map(evaluation =>
          [evaluation.node_id, evaluation]
        )
      );
      func.mappingRealization = func.plan?.realization || {
        logical_tile_count: 0,
        analog_lanes_per_tile: 0,
        digital_work_per_tile: [],
        node_allocations: [],
        leaf_assignments: [],
      };
      func.mappingNodeAllocationMap = new Map(
        func.mappingRealization.node_allocations.map(allocation =>
          [allocation.node_id, allocation]
        )
      );
      func.mappingLeafAssignmentMap = new Map(
        func.mappingRealization.leaf_assignments.map(assignment =>
          [assignment.leaf_id, assignment]
        )
      );
      func.realization = func.realization || {stage_count: 0, stages: []};
      func.realizationStageMap = new Map(
        func.realization.stages.map(stage => [stage.id, stage])
      );
      func.realizationByOperation = new Map();
      for (const stage of func.realization.stages) {
        if (!func.realizationByOperation.has(stage.operation_id)) {
          func.realizationByOperation.set(stage.operation_id, []);
        }
        func.realizationByOperation.get(stage.operation_id).push(stage);
      }
      for (const stages of func.realizationByOperation.values()) {
        stages.sort((left, right) => left.stage_index - right.stage_index);
      }
      func.logicalTileGraph = func.logical_tile_graph || {
        logical_tile_capacity: 0,
        analog_lanes_per_tile: 0,
        tiles: [],
        edges: [],
      };
      func.logicalTileMap = new Map(
        func.logicalTileGraph.tiles.map(tile => [tile.id, tile])
      );
      func.maximumDigitalWork = Math.max(
        1,
        ...func.logicalTileGraph.tiles.map(tile => tile.digital_work || 0)
      );
      func.physicalPlacement = func.physical_placement || null;
      func.annealingTrace = func.physicalPlacement?.annealing_trace || null;
      func.physicalAssignmentMap = new Map(
        (func.physicalPlacement?.assignments || []).map(assignment =>
          [assignment.logical_tile_id, assignment]
        )
      );
      func.placedTileEdgeMap = new Map(
        (func.physicalPlacement?.edges || []).map(edge => [edge.edge_id, edge])
      );
      return func;
    }

    function effectiveLaneBindingGroup(func, operation) {
      return operation?.lane_binding_group ?? -1;
    }

    function treeDepth(func) {
      const memo = new Map();
      function depth(nodeId) {
        if (memo.has(nodeId)) return memo.get(nodeId);
        const node = func.treeNodeMap.get(nodeId);
        const value = !node || node.parent_id < 0 ? 0 : depth(node.parent_id) + 1;
        memo.set(nodeId, value);
        return value;
      }
      return Math.max(...func.tree.nodes.map(node => depth(node.id)), 0);
    }

    function calculateBounds(nodes) {
      if (!nodes.length) return {minX: 0, minY: 0, maxX: 1, maxY: 1};
      return nodes.reduce((bounds, node) => ({
        minX: Math.min(bounds.minX, node.x - node.width / 2),
        minY: Math.min(bounds.minY, node.y - node.height / 2),
        maxX: Math.max(bounds.maxX, node.x + node.width / 2),
        maxY: Math.max(bounds.maxY, node.y + node.height / 2),
      }), {minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity});
    }

    function getRenderedLayout() {
      return layout;
    }

    const realizationKindOrder = [
      "sculptor.matrix_setup",
      "digital.vector_tile",
      "sculptor.mvm",
      "digital.tile_recombine",
    ];

    function groupRealizationStages(stages) {
      const groups = new Map();
      for (const stage of stages) {
        if (!groups.has(stage.kind)) groups.set(stage.kind, []);
        groups.get(stage.kind).push(stage);
      }
      return [...groups.entries()].sort((left, right) => {
        const leftRank = realizationKindOrder.indexOf(left[0]);
        const rightRank = realizationKindOrder.indexOf(right[0]);
        return (leftRank < 0 ? realizationKindOrder.length : leftRank) -
          (rightRank < 0 ? realizationKindOrder.length : rightRank);
      });
    }

    function appendPhysicalDependencyLinks(nodes, links) {
      const nodeByStageId = new Map(
        nodes.filter(node => node.realization)
          .map(node => [node.realization.id, node])
      );
      for (const target of nodeByStageId.values()) {
        for (const predecessorId of target.realization.predecessor_stage_ids || []) {
          const source = nodeByStageId.get(predecessorId);
          if (source) links.push({source, target, kind: "physical"});
        }
      }
    }

    function indexNodesByOperation(nodes) {
      const nodesByOperation = new Map();
      for (const node of nodes) {
        if (!nodesByOperation.has(node.operation.id)) {
          nodesByOperation.set(node.operation.id, []);
        }
        nodesByOperation.get(node.operation.id).push(node);
      }
      return nodesByOperation;
    }

    function dataEntryNodes(nodesByOperation, operationId) {
      const nodes = nodesByOperation.get(operationId) || [];
      const vectorTiles = nodes.filter(
        node => node.realization?.kind === "digital.vector_tile"
      );
      if (vectorTiles.length) return vectorTiles;
      const arrayMVMs = nodes.filter(
        node => node.realization?.kind === "sculptor.mvm"
      );
      return arrayMVMs.length ? arrayMVMs : nodes;
    }

    function dataExitNodes(nodesByOperation, operationId) {
      const nodes = nodesByOperation.get(operationId) || [];
      const recombinations = nodes.filter(
        node => node.realization?.kind === "digital.tile_recombine"
      );
      if (recombinations.length) return recombinations;
      const arrayMVMs = nodes.filter(
        node => node.realization?.kind === "sculptor.mvm"
      );
      return arrayMVMs.length ? arrayMVMs : nodes;
    }

    function appendLogicalDependencyLinks(func, nodes, links) {
      const nodesByOperation = indexNodesByOperation(nodes);
      const refinedOperationEdges = new Set(func.workUnitEdges.map(edge =>
        `${edge.source_operation_id}:${edge.target_operation_id}`
      ));

      function endpointNodes(operationId, workUnitId, direction) {
        let candidates = nodesByOperation.get(operationId) || [];
        if (workUnitId >= 0) {
          candidates = candidates.filter(node =>
            node.treeNode?.work_unit_id === workUnitId
          );
        }
        const scoped = new Map([[operationId, candidates]]);
        return direction === "source"
          ? dataExitNodes(scoped, operationId)
          : dataEntryNodes(scoped, operationId);
      }

      for (const edge of func.workUnitEdges) {
        const sources = endpointNodes(
          edge.source_operation_id, edge.source_work_unit_id, "source"
        );
        const targets = endpointNodes(
          edge.target_operation_id, edge.target_work_unit_id, "target"
        );
        for (const source of sources) {
          for (const target of targets) {
            links.push({source, target, kind: "dependency", edge,
              workUnitDependency: true});
          }
        }
      }

      for (const edge of func.edges) {
        if (refinedOperationEdges.has(`${edge.source}:${edge.target}`))
          continue;
        for (const source of dataExitNodes(nodesByOperation, edge.source)) {
          for (const target of dataEntryNodes(nodesByOperation, edge.target)) {
            links.push({source, target, kind: "dependency", edge});
          }
        }
      }
    }

    function buildTreeLayout(func) {
      const positions = new Map();
      const links = [];
      let leafCursor = 0;
      const horizontalSpacing = 430;
      const realizationSpacing = 400;
      const operationOrder = new Map(
        func.topological_order.map((operationId, index) => [operationId, index])
      );
      const subtreeOrderMemo = new Map();

      function subtreeOrder(nodeId) {
        if (subtreeOrderMemo.has(nodeId)) return subtreeOrderMemo.get(nodeId);
        const treeNode = func.treeNodeMap.get(nodeId);
        if (!treeNode) return Number.MAX_SAFE_INTEGER;
        let order = treeNode.operation_id >= 0
          ? operationOrder.get(treeNode.operation_id) ?? treeNode.operation_id
          : Number.MAX_SAFE_INTEGER;
        for (const childId of treeNode.child_ids)
          order = Math.min(order, subtreeOrder(childId));
        subtreeOrderMemo.set(nodeId, order);
        return order;
      }

      function orderedChildren(treeNode) {
        // Temporal child order is part of the mapping plan's execution
        // semantics. In particular, setup-first places the matrix-setup S-cut
        // before ordinary DAG operations whose raw operation IDs are lower.
        if (treeNode.kind === "temporal_cut")
          return [...treeNode.child_ids];
        return [...treeNode.child_ids].sort((left, right) =>
          subtreeOrder(left) - subtreeOrder(right) || left - right
        );
      }

      function leafSpan(operation) {
        const stages = operation
          ? visibleRealizationStages(func, operation.id)
          : [];
        if (!stages.length) return horizontalSpacing;
        const widestGroup = Math.max(
          ...groupRealizationStages(stages).map(([, group]) => group.length),
          1
        );
        return Math.max(
          horizontalSpacing,
          widestGroup * realizationSpacing + 20
        );
      }

      function place(nodeId, depth) {
        const treeNode = func.treeNodeMap.get(nodeId);
        if (!treeNode) return null;
        let x;
        if (!treeNode.child_ids.length) {
          const operation = treeNode.operation_id >= 0
            ? func.operationMap.get(treeNode.operation_id)
            : null;
          if (isSuppressedOperation(operation)) return null;
          const span = leafSpan(operation);
          x = leafCursor + span / 2;
          leafCursor += span + 16;
        } else {
          const childPositions = orderedChildren(treeNode).flatMap(childId => {
            const childX = place(childId, depth + 1);
            if (childX === null) return [];
            links.push({source: nodeId, target: childId});
            return [childX];
          });
          if (!childPositions.length) return null;
          x = childPositions.reduce((sum, value) => sum + value, 0) / childPositions.length;
        }
        const operation = treeNode.operation_id >= 0
          ? func.operationMap.get(treeNode.operation_id)
          : null;
        positions.set(nodeId, {
          key: `tree-${nodeId}`,
          id: nodeId,
          treeNode,
          operation,
          kind: treeNode.kind,
          x,
          y: depth,
          hierarchyDepth: depth,
          treeRow: depth,
          baseWidth: operation ? 400 : 600,
          maximumScreenWidth: operation ? 96 : 140,
          screenHeight: operation ? 30 : 36,
          width: operation ? 220 : 280,
          height: operation ? 30 : 36,
        });
        return x;
      }

      place(func.tree.root_id, 0);
      const nodes = [...positions.values()];
      const physicalStartRow = treeDepth(func) + 1;
      for (const logicalNode of [...nodes]) {
        const stages = logicalNode.operation
          ? visibleRealizationStages(func, logicalNode.operation.id)
          : [];
        if (!stages.length) continue;
        const localStageIds = new Set(stages.map(stage => stage.id));
        groupRealizationStages(stages).forEach(([, group], groupIndex) => {
          group.forEach((stage, stageIndex) => {
            const physicalNode = {
              key: `tree-physical-${stage.id}`,
              id: `physical-${stage.id}`,
              treeNode: logicalNode.treeNode,
              operation: logicalNode.operation,
              realization: stage,
              kind: "physical",
              x: logicalNode.x +
                (stageIndex - (group.length - 1) / 2) * realizationSpacing,
              y: physicalStartRow + groupIndex * .75,
              treeRow: physicalStartRow + groupIndex * .75,
              baseWidth: 400,
              maximumScreenWidth: 96,
              screenHeight: 30,
              width: 220,
              height: 30,
            };
            nodes.push(physicalNode);
            const hasLocalPredecessor = (stage.predecessor_stage_ids || [])
              .some(predecessor => localStageIds.has(predecessor));
            if (!hasLocalPredecessor) {
              links.push({
                source: logicalNode.treeNode.id,
                targetPhysical: physicalNode,
                kind: "ownership",
              });
            }
          });
        });
      }
      const resolvedLinks = links.map(link => ({
        source: positions.get(link.source),
        target: link.targetPhysical || positions.get(link.target),
        kind: link.kind || "tree",
      }));
      appendPhysicalDependencyLinks(nodes, resolvedLinks);
      const hierarchyRows = [...new Set(
        [...positions.values()].map(node => node.treeRow)
      )].sort((left, right) => left - right);
      function descendantLeaves(nodeId, leaves = []) {
        const treeNode = func.treeNodeMap.get(nodeId);
        if (!treeNode) return leaves;
        if (!treeNode.child_ids.length) {
          const node = positions.get(nodeId);
          if (node?.operation) leaves.push(node);
          return leaves;
        }
        for (const childId of treeNode.child_ids)
          descendantLeaves(childId, leaves);
        return leaves;
      }
      const root = func.treeNodeMap.get(func.tree.root_id);
      const setupPhaseRanges = (root?.child_ids || []).flatMap(childId => {
        const leaves = descendantLeaves(childId);
        if (!leaves.length ||
            !leaves.every(node => node.operation.kind === "matrix_setup"))
          return [];
        return [{
          nodeId: childId,
          operationCount: leaves.length,
          leaves,
          minX: Math.min(...leaves.map(node => node.x - node.width / 2)),
          maxX: Math.max(...leaves.map(node => node.x + node.width / 2)),
        }];
      });
      const horizontalBounds = {
        minX: Math.min(...nodes.map(node => node.x - node.baseWidth / 2)),
        maxX: Math.max(...nodes.map(node => node.x + node.baseWidth / 2)),
      };
      return {
        nodes,
        links: resolvedLinks,
        bounds: calculateBounds(nodes),
        hierarchyRows,
        setupPhaseRanges,
        horizontalBounds,
        maximumTreeRow: Math.max(...nodes.map(node => node.treeRow), 1),
        ordered: true,
      };
    }

    function buildDAGLayout(func) {
      const order = new Map(func.topological_order.map((id, index) => [id, index]));
      const orderedOperations = func.operations
        .filter(operation => !isSuppressedOperation(operation))
        .sort((left, right) =>
          (order.get(left.id) ?? left.id) - (order.get(right.id) ?? right.id)
        );
      const operationMap = new Map(
        orderedOperations.map(operation => [operation.id, operation])
      );
      const predecessors = new Map(
        orderedOperations.map(operation => [operation.id, new Set()])
      );
      const successors = new Map(
        orderedOperations.map(operation => [operation.id, new Set()])
      );
      for (const edge of func.edges) {
        if (!operationMap.has(edge.source) || !operationMap.has(edge.target) ||
            edge.source === edge.target) continue;
        predecessors.get(edge.target).add(edge.source);
        successors.get(edge.source).add(edge.target);
      }

      // Longest-path ranks expose graph fan-out and joins while preserving every
      // dependency. The graph remains a DAG; this layout does not duplicate
      // shared consumers to force it into a strict tree.
      const rankByOperation = new Map();
      const layers = new Map();
      for (const operation of orderedOperations) {
        const rank = Math.max(
          0,
          ...[...predecessors.get(operation.id)]
            .map(id => (rankByOperation.get(id) ?? -1) + 1)
        );
        rankByOperation.set(operation.id, rank);
        if (!layers.has(rank)) layers.set(rank, []);
        layers.get(rank).push(operation);
      }
      const maximumRank = Math.max(...layers.keys(), 0);
      const operationPosition = new Map();

      function updateOperationPositions() {
        operationPosition.clear();
        for (const [rank, operations] of layers) {
          operations.forEach((operation, index) => {
            const normalized = operations.length === 1
              ? .5
              : index / (operations.length - 1);
            operationPosition.set(operation.id, {rank, index, normalized});
          });
        }
      }

      function reorderLayer(rank, neighborMap) {
        const operations = layers.get(rank) || [];
        const decorated = operations.map((operation, index) => {
          const neighbors = [...neighborMap.get(operation.id)]
            .map(id => operationPosition.get(id)?.normalized)
            .filter(value => value !== undefined);
          return {
            operation,
            barycenter: neighbors.length
              ? neighbors.reduce((sum, value) => sum + value, 0) /
                  neighbors.length
              : operationPosition.get(operation.id)?.normalized ?? .5,
            index,
          };
        });
        decorated.sort((left, right) =>
          left.barycenter - right.barycenter ||
          left.index - right.index ||
          (order.get(left.operation.id) ?? left.operation.id) -
            (order.get(right.operation.id) ?? right.operation.id)
        );
        layers.set(rank, decorated.map(entry => entry.operation));
        updateOperationPositions();
      }

      updateOperationPositions();
      for (let iteration = 0; iteration < 6; ++iteration) {
        for (let rank = 1; rank <= maximumRank; ++rank)
          reorderLayer(rank, predecessors);
        for (let rank = maximumRank - 1; rank >= 0; --rank)
          reorderLayer(rank, successors);
      }

      const stageWidth = 82;
      const stageHeight = 38;
      const stageGap = 16;
      const laneGap = 10;
      const operationNodeWidth = 82;
      const operationNodeHeight = 40;
      const horizontalGap = 92;
      const verticalGap = 34;

      function realizationGroups(operation) {
        return groupRealizationStages(
          visibleRealizationStages(func, operation.id)
        );
      }

      function operationWidth(operation) {
        const groups = realizationGroups(operation);
        return groups.length
          ? groups.length * stageWidth + Math.max(0, groups.length - 1) * stageGap
          : operationNodeWidth;
      }

      function operationHeight(operation) {
        const groups = realizationGroups(operation);
        const maximumLanes = Math.max(
          ...groups.map(([, group]) => group.length),
          1
        );
        return groups.length
          ? maximumLanes * stageHeight + Math.max(0, maximumLanes - 1) * laneGap
          : operationNodeHeight;
      }

      const rankWidths = new Map();
      for (const [rank, operations] of layers) {
        rankWidths.set(rank, Math.max(...operations.map(operationWidth), 1));
      }
      const rankCenters = [];
      let horizontalCursor = 0;
      for (let rank = 0; rank <= maximumRank; ++rank) {
        const width = rankWidths.get(rank) || operationNodeWidth;
        rankCenters[rank] = horizontalCursor + width / 2;
        horizontalCursor += width + horizontalGap;
      }

      const operationY = new Map();
      for (const [, operations] of layers) {
        const totalHeight = operations.reduce(
          (sum, operation) => sum + operationHeight(operation),
          Math.max(0, operations.length - 1) * verticalGap
        );
        let verticalCursor = -totalHeight / 2;
        for (const operation of operations) {
          const height = operationHeight(operation);
          operationY.set(operation.id, verticalCursor + height / 2);
          verticalCursor += height + verticalGap;
        }
      }

      const nodes = [];
      for (const operation of orderedOperations) {
        const rank = rankByOperation.get(operation.id);
        const rowCenter = operationY.get(operation.id);
        const centerX = rankCenters[rank];
        const stages = visibleRealizationStages(func, operation.id);
        if (!stages.length) {
          nodes.push({
            key: `dag-${operation.id}`,
            id: operation.id,
            treeNode: func.leafByOperation.get(operation.id),
            operation,
            kind: "leaf",
            x: centerX,
            y: rowCenter,
            width: operationNodeWidth,
            height: operationNodeHeight,
          });
          continue;
        }

        const groups = groupRealizationStages(stages);
        const realizationWidth = operationWidth(operation);
        groups.forEach(([, group], groupIndex) => {
          group.forEach((stage, laneIndex) => {
            nodes.push({
              key: `dag-physical-${stage.id}`,
              id: `physical-${stage.id}`,
              treeNode: func.leafByOperation.get(operation.id),
              operation,
              realization: stage,
              kind: "physical",
              x: centerX - realizationWidth / 2 + stageWidth / 2 +
                groupIndex * (stageWidth + stageGap),
              y: rowCenter +
                (laneIndex - (group.length - 1) / 2) *
                  (stageHeight + laneGap),
              width: stageWidth,
              height: stageHeight,
            });
          });
        });
      }
      const links = [];
      appendLogicalDependencyLinks(func, nodes, links);
      appendPhysicalDependencyLinks(nodes, links);
      return {
        nodes,
        links,
        bounds: calculateBounds(nodes),
        rankCenters,
      };
    }

    function buildExplicitSTLayout(func) {
      const assignments = (func.mappingRealization?.leaf_assignments || [])
        .filter(assignment =>
          !isSuppressedOperation(func.operationMap.get(assignment.operation_id))
        );
      if (!assignments.length) return null;

      const leafWidth = 88;
      const leafHeight = 42;
      const temporalGap = 14;
      const temporalPitch = leafWidth + temporalGap;
      const lanePitch = 64;
      const tileGap = 32;
      const offsetX = 176;
      const offsetY = 66;
      const laneKeys = [...new Set(assignments.map(assignment =>
        `${assignment.tile_id}:${assignment.lane_kind}:${assignment.lane_index}`
      ))].map(key => {
        const [tileId, laneKind, laneIndex] = key.split(":");
        return {
          key,
          tileId: Number(tileId),
          laneKind,
          laneIndex: Number(laneIndex),
        };
      }).sort((left, right) =>
        left.tileId - right.tileId ||
        (left.laneKind === "digital" ? -1 : 1) -
          (right.laneKind === "digital" ? -1 : 1) ||
        left.laneIndex - right.laneIndex
      );
      const laneByKey = new Map();
      const tileRanges = new Map();
      let laneY = offsetY;
      let previousTile = null;
      laneKeys.forEach((lane, index) => {
        if (previousTile !== null && lane.tileId !== previousTile)
          laneY += tileGap;
        lane.index = index;
        lane.y = laneY;
        laneByKey.set(lane.key, lane);
        const range = tileRanges.get(lane.tileId) || {
          tileId: lane.tileId,
          minY: laneY,
          maxY: laneY,
        };
        range.minY = Math.min(range.minY, laneY);
        range.maxY = Math.max(range.maxY, laneY);
        tileRanges.set(lane.tileId, range);
        laneY += lanePitch;
        previousTile = lane.tileId;
      });

      // The evaluator reports physical nanoseconds. A proportional axis makes
      // a long setup phase consume almost the entire canvas and causes short,
      // sequential operations to overlap once they receive a readable minimum
      // width. Map distinct event boundaries to ordered columns instead. This
      // preserves ordering and concurrency while exact times remain attached
      // to every assignment and are shown in the inspector.
      const eventTimes = [...new Set(assignments.flatMap(assignment => [
        assignment.start_ns,
        assignment.finish_ns,
      ]))].sort((left, right) => left - right);
      const eventIndex = new Map(
        eventTimes.map((time, index) => [time, index])
      );
      const nodes = [];
      for (const assignment of assignments) {
        const treeNode = func.treeNodeMap.get(assignment.leaf_id);
        const operation = func.operationMap.get(assignment.operation_id);
        if (!treeNode || !operation) continue;
        const laneKey =
          `${assignment.tile_id}:${assignment.lane_kind}:${assignment.lane_index}`;
        const lane = laneByKey.get(laneKey);
        if (!lane) continue;
        const durationNs = Math.max(0, assignment.finish_ns - assignment.start_ns);
        const startEvent = eventIndex.get(assignment.start_ns);
        const finishEvent = eventIndex.get(assignment.finish_ns);
        const eventSpan = Math.max(1, finishEvent - startEvent);
        const visualStart = offsetX + startEvent * temporalPitch;
        const visualFinish = visualStart + eventSpan * temporalPitch;
        const width = visualFinish - visualStart - temporalGap;
        nodes.push({
          key: `st-realized-${assignment.leaf_id}`,
          id: treeNode.work_unit_id >= 0
            ? `${assignment.operation_id}.${treeNode.work_unit_id}`
            : assignment.operation_id,
          treeNode,
          operation,
          mappingAssignment: assignment,
          kind: "leaf",
          x: (visualStart + visualFinish) / 2,
          y: lane.y,
          width,
          height: leafHeight,
          stLane: lane.index,
          stLaneBindingGroup: operation.lane_binding_group,
          stStart: assignment.start_ns,
          stDuration: durationNs,
          stStartEvent: startEvent,
          stFinishEvent: finishEvent,
        });
      }
      nodes.sort((left, right) =>
        left.stStartEvent - right.stStartEvent ||
        left.stLane - right.stLane ||
        left.stFinishEvent - right.stFinishEvent ||
        left.operation.id - right.operation.id
      );

      const links = [];
      appendLogicalDependencyLinks(func, nodes, links);
      const digitalWork =
        func.mappingRealization?.digital_work_per_tile || [];
      const laneLabels = new Map(
        laneKeys.map(lane => [
          lane.y,
          lane.laneKind === "digital"
            ? `TILE ${lane.tileId} · DIGITAL · WORK ${
                (digitalWork[lane.tileId] || 0).toLocaleString()
              }`
            : `TILE ${lane.tileId} · ANALOG ${lane.laneIndex}`,
        ])
      );
      const guideStride = Math.max(1, Math.ceil(eventTimes.length / 14));
      const timeGuides = eventTimes.flatMap((time, index) =>
        index === 0 || index === eventTimes.length - 1 || index % guideStride === 0
          ? [{x: offsetX + index * temporalPitch, time, event: index}]
          : []
      );
      const timelineBounds = calculateBounds(nodes);
      return {
        nodes,
        links,
        bounds: {
          minX: timelineBounds.minX - 168,
          minY: timelineBounds.minY - 48,
          maxX: timelineBounds.maxX + 24,
          maxY: timelineBounds.maxY + 24,
        },
        timelineBounds,
        resourceLaneCount: laneKeys.length,
        laneLabels,
        tileRanges: [...tileRanges.values()],
        timeGuides,
        explicitRealization: true,
        cyclicSpatialDependency: false,
      };
    }

    function buildInferredSTLayout(func) {
      const leafWidth = 88;
      const leafHeight = 42;
      const temporalGap = 28;
      const spatialGap = 34;
      const operationSets = new Map();
      let cyclicSpatialDependency = false;

      function collectOperations(nodeId) {
        if (operationSets.has(nodeId)) return operationSets.get(nodeId);
        const treeNode = func.treeNodeMap.get(nodeId);
        const operations = new Set();
        if (treeNode?.operation_id >= 0) {
          const operation = func.operationMap.get(treeNode.operation_id);
          if (!isSuppressedOperation(operation))
            operations.add(treeNode.operation_id);
        } else {
          for (const childId of treeNode?.child_ids || []) {
            for (const operationId of collectOperations(childId)) operations.add(operationId);
          }
        }
        operationSets.set(nodeId, operations);
        return operations;
      }

      function spatialStarts(childIds, childLayouts) {
        const operationToChild = new Map();
        childIds.forEach((childId, index) => {
          for (const operationId of collectOperations(childId)) {
            operationToChild.set(operationId, index);
          }
        });

        const successors = childIds.map(() => new Set());
        const indegree = childIds.map(() => 0);
        for (const edge of func.edges) {
          const source = operationToChild.get(edge.source);
          const target = operationToChild.get(edge.target);
          if (source === undefined || target === undefined || source === target ||
              successors[source].has(target)) continue;
          successors[source].add(target);
          indegree[target] += 1;
        }

        const ready = childIds.map((_, index) => index)
          .filter(index => indegree[index] === 0);
        const order = [];
        while (ready.length) {
          ready.sort((left, right) => left - right);
          const source = ready.shift();
          order.push(source);
          for (const target of successors[source]) {
            indegree[target] -= 1;
            if (indegree[target] === 0) ready.push(target);
          }
        }

        if (order.length !== childIds.length) {
          cyclicSpatialDependency = true;
          return {
            starts: childIds.map((_, index) => index * (leafWidth + temporalGap)),
            stages: childIds.map((_, index) => index),
          };
        }

        const starts = childIds.map(() => 0);
        const stages = childIds.map(() => 0);
        for (const source of order) {
          for (const target of successors[source]) {
            starts[target] = Math.max(
              starts[target], starts[source] + childLayouts[source].width + temporalGap
            );
            stages[target] = Math.max(stages[target], stages[source] + 1);
          }
        }
        return {starts, stages};
      }

      function arrange(nodeId) {
        const treeNode = func.treeNodeMap.get(nodeId);
        if (!treeNode) return {nodes: [], width: 1, height: 1};
        if (treeNode.operation_id >= 0) {
          const operation = func.operationMap.get(treeNode.operation_id);
          if (isSuppressedOperation(operation))
            return {nodes: [], width: 0, height: 0};
          const stages = visibleRealizationStages(func, operation.id);
          if (stages.length) {
            const stageWidth = 86;
            const stageHeight = 38;
            const stageGap = 16;
            const laneGap = 10;
            const groups = groupRealizationStages(stages);
            const maximumLanes = Math.max(
              ...groups.map(([, group]) => group.length), 1
            );
            const realizationNodes = [];
            groups.forEach(([, group], groupIndex) => {
              group.forEach((stage, laneIndex) => {
                realizationNodes.push({
                  key: `st-physical-${stage.id}`,
                  id: `physical-${stage.id}`,
                  treeNode,
                  operation,
                  realization: stage,
                  kind: "physical",
                  x: groupIndex * (stageWidth + stageGap) + stageWidth / 2,
                  y: ((maximumLanes - group.length) / 2 + laneIndex) *
                    (stageHeight + laneGap) + stageHeight / 2,
                  width: stageWidth,
                  height: stageHeight,
                  stStage: groupIndex,
                });
              });
            });
            return {
              nodes: realizationNodes,
              width: groups.length * stageWidth +
                Math.max(0, groups.length - 1) * stageGap,
              height: maximumLanes * stageHeight +
                Math.max(0, maximumLanes - 1) * laneGap,
            };
          }
          const node = {
            key: `st-${treeNode.operation_id}-${treeNode.work_unit_id}`,
            id: treeNode.work_unit_id >= 0
              ? `${treeNode.operation_id}.${treeNode.work_unit_id}`
              : treeNode.operation_id,
            treeNode,
            operation,
            kind: "leaf",
            x: leafWidth / 2,
            y: leafHeight / 2,
            width: leafWidth,
            height: leafHeight,
            stStage: 0,
          };
          return {nodes: [node], width: leafWidth, height: leafHeight};
        }

        const childEntries = treeNode.child_ids
          .map(childId => ({childId, layout: arrange(childId)}))
          .filter(entry => entry.layout.nodes.length);
        if (!childEntries.length)
          return {nodes: [], width: 0, height: 0};
        const childIds = childEntries.map(entry => entry.childId);
        const childLayouts = childEntries.map(entry => entry.layout);
        if (treeNode.kind === "temporal_cut") {
          const height = Math.max(...childLayouts.map(child => child.height), leafHeight);
          let cursor = 0;
          for (const child of childLayouts) {
            for (const node of child.nodes) {
              node.x += cursor;
            }
            cursor += child.width + temporalGap;
          }
          return {
            nodes: childLayouts.flatMap(child => child.nodes),
            width: Math.max(1, cursor - temporalGap),
            height,
          };
        }

        const schedule = spatialStarts(childIds, childLayouts);
        let cursorY = 0;
        let width = 1;
        childLayouts.forEach((child, index) => {
          for (const node of child.nodes) {
            node.x += schedule.starts[index];
            node.y += cursorY;
            node.stStage += schedule.stages[index];
          }
          width = Math.max(width, schedule.starts[index] + child.width);
          cursorY += child.height + spatialGap;
        });
        return {
          nodes: childLayouts.flatMap(child => child.nodes),
          width,
          height: Math.max(1, cursorY - spatialGap),
        };
      }

      const arranged = arrange(func.tree.root_id);
      const analogBindingGroups = [...new Set(
        arranged.nodes
          .map(node => effectiveLaneBindingGroup(func, node.operation))
          .filter(group => group >= 0)
      )].sort((left, right) => left - right);
      const structuralDigitalLanes = [...new Set(
        arranged.nodes
          .filter(node => effectiveLaneBindingGroup(func, node.operation) < 0)
          .map(node => node.y)
      )].sort((left, right) => left - right);
      const analogLaneByGroup = new Map(
        analogBindingGroups.map((group, index) => [group, index])
      );
      const digitalLaneByCoordinate = new Map(
        structuralDigitalLanes.map((coordinate, index) =>
          [coordinate, analogBindingGroups.length + index]
        )
      );
      const lanePitch = leafHeight + spatialGap;
      for (const node of arranged.nodes) {
        const bindingGroup = effectiveLaneBindingGroup(func, node.operation);
        const lane = bindingGroup >= 0
          ? analogLaneByGroup.get(bindingGroup)
          : digitalLaneByCoordinate.get(node.y);
        node.y = lane * lanePitch + leafHeight / 2;
        node.stLaneBindingGroup = bindingGroup;
      }

      const offsetX = 92;
      const offsetY = 66;
      for (const node of arranged.nodes) {
        node.x += offsetX;
        node.y += offsetY;
      }

      const laneCoordinates = [...new Set(arranged.nodes.map(node => node.y))]
        .sort((left, right) => left - right);
      const laneByCoordinate = new Map(
        laneCoordinates.map((coordinate, index) => [coordinate, index])
      );
      const minimumStart = Math.min(
        ...arranged.nodes.map(node => node.x - node.width / 2),
        offsetX
      );
      for (const node of arranged.nodes) {
        node.stLane = laneByCoordinate.get(node.y);
        node.stStart = (node.x - node.width / 2 - minimumStart) /
          (leafWidth + temporalGap);
        node.stDuration = node.width / leafWidth;
      }

      const links = [];
      appendLogicalDependencyLinks(func, arranged.nodes, links);
      appendPhysicalDependencyLinks(arranged.nodes, links);
      return {
        nodes: arranged.nodes,
        links,
        bounds: calculateBounds(arranged.nodes),
        resourceLaneCount: laneCoordinates.length,
        cyclicSpatialDependency,
      };
    }

    function buildSTLayout(func) {
      return buildExplicitSTLayout(func) || buildInferredSTLayout(func);
    }

    function buildPhysicalLayout(func) {
      const placement = func.physicalPlacement;
      if (!placement) {
        return {
          nodes: [],
          links: [],
          bounds: {minX: 0, minY: 0, maxX: 1, maxY: 1},
        };
      }

      const cellWidth = 150;
      const cellHeight = 112;
      const margin = 48;
      const nodes = placement.assignments.map(assignment => ({
        key: `physical-tile-${assignment.physical_tile_id}`,
        logicalTile: func.logicalTileMap.get(assignment.logical_tile_id),
        physicalAssignment: assignment,
        kind: "physical_tile",
        x: margin + assignment.column * cellWidth + cellWidth / 2,
        y: margin + assignment.row * cellHeight + cellHeight / 2,
        width: 116,
        height: 70,
      }));
      const nodeByLogicalTile = new Map(
        nodes.map(node => [node.logicalTile.id, node])
      );
      const links = placement.edges.flatMap(edge => {
        const source = nodeByLogicalTile.get(edge.source_tile_id);
        const target = nodeByLogicalTile.get(edge.target_tile_id);
        return source && target
          ? [{source, target, kind: "tile_communication", placedEdge: edge}]
          : [];
      });
      const linksByLogicalTile = new Map();
      for (const link of links) {
        for (const tileId of [
          link.placedEdge.source_tile_id,
          link.placedEdge.target_tile_id,
        ]) {
          if (!linksByLogicalTile.has(tileId))
            linksByLogicalTile.set(tileId, []);
          linksByLogicalTile.get(tileId).push(link);
        }
      }

      return {
        nodes,
        links,
        bounds: {
          minX: margin,
          minY: margin,
          maxX: margin + placement.mesh_cols * cellWidth,
          maxY: margin + placement.mesh_rows * cellHeight,
        },
        physicalPlacement: placement,
        linksByLogicalTile,
        cellWidth,
        cellHeight,
        margin,
      };
    }

    function rebuildLayout(fit = false) {
      if (currentView === "tree") layout = buildTreeLayout(currentFunction);
      else if (currentView === "dag") layout = buildDAGLayout(currentFunction);
      else if (currentView === "physical") layout = buildPhysicalLayout(currentFunction);
      else layout = buildSTLayout(currentFunction);
      renderedLayout = layout;
      updateViewExplanation();
      if (fit) fitView(true);
      else draw();
    }

    function updateViewURL() {
      const url = new URL(window.location.href);
      url.searchParams.set("view", currentView);
      window.history.replaceState({}, "", url);
    }

    function compactScore(value) {
      if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
      if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
      if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
      return value.toLocaleString();
    }

    function stopAnnealingReplay() {
      if (annealingTimer !== null) {
        window.clearInterval(annealingTimer);
        annealingTimer = null;
      }
    }

    function drawAnnealingChart(frame) {
      const trace = currentFunction?.annealingTrace;
      if (!trace || annealingProgress.hidden) return;
      const samples = trace.samples;
      const rect = annealingChart.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const ratio = window.devicePixelRatio || 1;
      annealingChart.width = Math.max(1, Math.floor(rect.width * ratio));
      annealingChart.height = Math.max(1, Math.floor(rect.height * ratio));
      const ctx = annealingChart.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const padding = {left: 58, right: 18, top: 14, bottom: 28};
      const plotWidth = Math.max(1, rect.width - padding.left - padding.right);
      const plotHeight = Math.max(1, rect.height - padding.top - padding.bottom);
      let minimumLog = Infinity;
      let maximumLog = -Infinity;
      const logScore = score => Math.log10(Math.max(1, score));
      for (const sample of samples) {
        minimumLog = Math.min(
          minimumLog, logScore(sample.candidate_score),
          logScore(sample.current_score), logScore(sample.best_score)
        );
        maximumLog = Math.max(
          maximumLog, logScore(sample.candidate_score),
          logScore(sample.current_score), logScore(sample.best_score)
        );
      }
      const logPadding = Math.max(.02, (maximumLog - minimumLog) * .06);
      minimumLog = Math.max(0, minimumLog - logPadding);
      maximumLog += logPadding;
      const logRange = Math.max(.01, maximumLog - minimumLog);
      const x = sample => padding.left +
        (sample.iteration / Math.max(1, trace.evaluations)) * plotWidth;
      const y = score => padding.top +
        (maximumLog - logScore(score)) / logRange * plotHeight;

      ctx.font = "9px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let line = 0; line <= 4; ++line) {
        const fraction = line / 4;
        const score = Math.pow(10, maximumLog - logRange * fraction);
        const coordinate = padding.top + plotHeight * fraction;
        ctx.strokeStyle = "#e1e5ea";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, coordinate);
        ctx.lineTo(padding.left + plotWidth, coordinate);
        ctx.stroke();
        ctx.fillStyle = "#687384";
        ctx.fillText(compactScore(score), padding.left - 7, coordinate);
      }
      ctx.save();
      ctx.fillStyle = "#687384";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.translate(10, padding.top + plotHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("LOG10 SCORE", 0, 0);
      ctx.restore();

      const visible = samples.slice(0, frame + 1);
      function drawSeries(field, color, width) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        visible.forEach((sample, index) => {
          const px = x(sample);
          const py = y(sample[field]);
          if (index === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }
      drawSeries("candidate_score", "#aab0b8", 1);
      drawSeries("current_score", "#2f6ea5", 1.5);
      drawSeries("best_score", "#2f7c61", 2.5);

      const active = samples[frame];
      const markerX = x(active);
      ctx.strokeStyle = "#202a37";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(markerX, padding.top);
      ctx.lineTo(markerX, padding.top + plotHeight);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#687384";
      ctx.textBaseline = "top";
      ctx.textAlign = "left";
      ctx.fillText("0", padding.left, padding.top + plotHeight + 8);
      ctx.textAlign = "right";
      ctx.fillText(
        trace.evaluations.toLocaleString(),
        padding.left + plotWidth,
        padding.top + plotHeight + 8
      );
      ctx.textAlign = "center";
      ctx.fillText("EVALUATED MOVES", padding.left + plotWidth / 2,
                   padding.top + plotHeight + 8);
    }

    function renderAnnealingFrame(frame) {
      const trace = currentFunction?.annealingTrace;
      if (!trace) return;
      annealingFrame = Math.max(0, Math.min(frame, trace.samples.length - 1));
      const sample = trace.samples[annealingFrame];
      const improvement = trace.initial_score - sample.best_score;
      const improvementPercent = trace.initial_score > 0
        ? improvement / trace.initial_score * 100
        : 0;
      annealingSlider.value = annealingFrame;
      document.querySelector("#annealing-live-score").textContent =
        `${sample.current_score.toLocaleString()} current · ` +
        `${sample.best_score.toLocaleString()} best`;
      document.querySelector("#annealing-iteration").textContent =
        `${sample.iteration.toLocaleString()} / ${trace.evaluations.toLocaleString()}`;
      document.querySelector("#annealing-candidate").textContent =
        `${sample.candidate_score.toLocaleString()} · ${sample.accepted ? "accepted" : "rejected"}`;
      document.querySelector("#annealing-current").textContent =
        sample.current_score.toLocaleString();
      document.querySelector("#annealing-improvement").textContent =
        `+${improvement.toLocaleString()} (${improvementPercent.toFixed(2)}%)`;
      document.querySelector("#metric-placement-score").textContent =
        sample.best_score.toLocaleString();
      drawAnnealingChart(annealingFrame);
    }

    function playAnnealingTrace() {
      const trace = currentFunction?.annealingTrace;
      if (!trace) return;
      stopAnnealingReplay();
      renderAnnealingFrame(0);
      const lastFrame = trace.samples.length - 1;
      const frameStep = Math.max(1, Math.ceil(lastFrame / 240));
      annealingTimer = window.setInterval(() => {
        const nextFrame = Math.min(lastFrame, annealingFrame + frameStep);
        renderAnnealingFrame(nextFrame);
        if (nextFrame === lastFrame) stopAnnealingReplay();
      }, 33);
    }

    function setupAnnealingProgress() {
      stopAnnealingReplay();
      const trace = currentFunction?.annealingTrace;
      annealingProgress.hidden = !trace;
      if (!trace) return;
      annealingSlider.max = trace.samples.length - 1;
      window.requestAnimationFrame(playAnnealingTrace);
    }

    function canvasSize() {
      const rect = canvas.getBoundingClientRect();
      return {width: rect.width, height: rect.height};
    }

    function projectTreeLayout(size) {
      if (currentView !== "tree" || !layout.ordered) return;
      const scale = Math.max(viewport.scale, .000001);
      const availableHeight = Math.max(220, size.height - 100);
      const rowSpacingPixels = Math.max(
        68,
        Math.min(118, availableHeight / Math.max(1, layout.maximumTreeRow))
      );
      for (const node of layout.nodes) {
        node.y = node.treeRow * rowSpacingPixels / scale;
        const screenWidth = Math.min(
          node.baseWidth * scale,
          node.maximumScreenWidth
        );
        node.width = screenWidth / scale;
        node.height = node.screenHeight / scale;
      }
      layout.hierarchyRows = [...new Set(
        layout.nodes
          .filter(node => !node.realization)
          .map(node => node.hierarchyDepth * rowSpacingPixels / scale)
      )].sort((left, right) => left - right);
      for (const phase of layout.setupPhaseRanges || []) {
        phase.minX = Math.min(...phase.leaves.map(
          node => node.x - node.width / 2
        ));
        phase.maxX = Math.max(...phase.leaves.map(
          node => node.x + node.width / 2
        ));
      }
      layout.bounds = calculateBounds(layout.nodes);
    }

    function resizeCanvas() {
      const size = canvasSize();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(size.width * ratio));
      canvas.height = Math.max(1, Math.floor(size.height * ratio));
      if (currentView === "tree") {
        projectTreeLayout(size);
        const worldHeight = layout.bounds.maxY - layout.bounds.minY;
        viewport.y = (size.height - worldHeight * viewport.scale) / 2 -
          layout.bounds.minY * viewport.scale;
      }
      draw();
    }

    function fitView() {
      renderedLayout = getRenderedLayout();
      const size = canvasSize();
      if (currentView === "tree") {
        const horizontalBounds = renderedLayout.horizontalBounds;
        const horizontalWidth = Math.max(
          1,
          horizontalBounds.maxX - horizontalBounds.minX
        );
        viewport.scale = Math.min(
          1.25,
          Math.max(.0001, (size.width - 90) / horizontalWidth)
        );
        projectTreeLayout(size);
      }
      const bounds = renderedLayout.bounds;
      const worldWidth = Math.max(1, bounds.maxX - bounds.minX);
      const worldHeight = Math.max(1, bounds.maxY - bounds.minY);
      if (currentView !== "tree") {
        viewport.scale = Math.min(
          1.25,
          Math.max(.001, Math.min((size.width - 90) / worldWidth, (size.height - 90) / worldHeight))
        );
      }
      viewport.x = (size.width - worldWidth * viewport.scale) / 2 - bounds.minX * viewport.scale;
      viewport.y = (size.height - worldHeight * viewport.scale) / 2 - bounds.minY * viewport.scale;
      draw();
    }

    function focusSTStart() {
      const size = canvasSize();
      const bounds = layout.bounds;
      const worldHeight = Math.max(1, bounds.maxY - bounds.minY);
      viewport.scale = Math.min(
        1.1,
        Math.max(.55, (size.height - 48) / worldHeight)
      );
      viewport.x = 16 - bounds.minX * viewport.scale;
      viewport.y = (size.height - worldHeight * viewport.scale) / 2 -
        bounds.minY * viewport.scale;
      draw();
    }

    function worldPoint(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (clientX - rect.left - viewport.x) / viewport.scale,
        y: (clientY - rect.top - viewport.y) / viewport.scale,
      };
    }

    function nodeAt(clientX, clientY) {
      const point = worldPoint(clientX, clientY);
      const activeLayout = getRenderedLayout();
      for (let index = activeLayout.nodes.length - 1; index >= 0; --index) {
        const node = activeLayout.nodes[index];
        if (point.x >= node.x - node.width / 2 && point.x <= node.x + node.width / 2 &&
            point.y >= node.y - node.height / 2 && point.y <= node.y + node.height / 2) {
          return node;
        }
      }
      return null;
    }

    function drawTreeGuides(ctx, activeLayout) {
      if (currentView !== "tree" || viewport.scale < .0001 ||
          !activeLayout.hierarchyRows?.length) return;
      const bounds = activeLayout.bounds;
      ctx.save();
      ctx.setLineDash([4 / viewport.scale, 7 / viewport.scale]);
      ctx.strokeStyle = "#c9d0d8";
      ctx.lineWidth = .65 / viewport.scale;
      ctx.fillStyle = "#657181";
      ctx.font = `800 ${9 / viewport.scale}px sans-serif`;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      activeLayout.hierarchyRows.forEach((coordinate, level) => {
        ctx.beginPath();
        ctx.moveTo(bounds.minX, coordinate);
        ctx.lineTo(bounds.maxX, coordinate);
        ctx.stroke();
        const label = level === 0
          ? "ROOT"
          : level === activeLayout.hierarchyRows.length - 1
          ? "OPERATION LEAVES"
          : `CUT LEVEL ${level}`;
        ctx.fillText(label, bounds.minX, coordinate - 8 / viewport.scale);
      });
      const phaseY = (activeLayout.hierarchyRows[1] ?? bounds.minY) -
        23 / viewport.scale;
      for (const phase of activeLayout.setupPhaseRanges || []) {
        ctx.setLineDash([]);
        ctx.strokeStyle = "#2f6ea5";
        ctx.lineWidth = 1.5 / viewport.scale;
        ctx.beginPath();
        ctx.moveTo(phase.minX, phaseY + 6 / viewport.scale);
        ctx.lineTo(phase.minX, phaseY);
        ctx.lineTo(phase.maxX, phaseY);
        ctx.lineTo(phase.maxX, phaseY + 6 / viewport.scale);
        ctx.stroke();
        ctx.fillStyle = "#244f78";
        ctx.textAlign = "center";
        ctx.fillText(
          `MATRIX SETUP PHASE · ${phase.operationCount.toLocaleString()}`,
          (phase.minX + phase.maxX) / 2,
          phaseY - 4 / viewport.scale
        );
      }
      ctx.setLineDash([]);
      ctx.textAlign = "center";
      ctx.fillText(
        "RA EXECUTION ORDER  →",
        (bounds.minX + bounds.maxX) / 2,
        bounds.minY - 20 / viewport.scale
      );
      ctx.restore();
    }

    function drawSTGuides(ctx, activeLayout) {
      if (currentView !== "st" || viewport.scale < .04) return;
      const laneCoordinates = [...new Set(activeLayout.nodes.map(node => node.y))]
        .sort((left, right) => left - right);
      const bounds = activeLayout.timelineBounds || activeLayout.bounds;
      ctx.save();
      for (const [index, tile] of (activeLayout.tileRanges || []).entries()) {
        const top = tile.minY - 29;
        const bottom = tile.maxY + 29;
        ctx.fillStyle = index % 2 === 0 ? "#f5f8fb" : "#fafbfc";
        ctx.fillRect(
          bounds.minX - 156,
          top,
          bounds.maxX - bounds.minX + 180,
          bottom - top
        );
        ctx.fillStyle = "#354254";
        ctx.font = "800 10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`LOGICAL TILE ${tile.tileId}`, bounds.minX - 150, top - 5);
      }
      ctx.setLineDash([3, 7]);
      ctx.strokeStyle = "#d1d6dd";
      ctx.fillStyle = "#647080";
      ctx.font = "700 9px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      for (const guide of activeLayout.timeGuides || []) {
        ctx.beginPath();
        ctx.moveTo(guide.x, bounds.minY - 18);
        ctx.lineTo(guide.x, bounds.maxY + 18);
        ctx.stroke();
        ctx.fillText(formatNanoseconds(guide.time), guide.x, bounds.minY - 23);
      }
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "#aeb6c1";
      ctx.fillStyle = "#647080";
      ctx.font = "700 9px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      laneCoordinates.forEach((coordinate, index) => {
        ctx.beginPath();
        ctx.moveTo(bounds.minX - 18, coordinate);
        ctx.lineTo(bounds.maxX + 18, coordinate);
        ctx.stroke();
        const label = activeLayout.laneLabels?.get(coordinate) || `LANE ${index}`;
        ctx.fillText(label, bounds.minX - 24, coordinate);
      });
      ctx.setLineDash([]);
      ctx.textAlign = "left";
      ctx.fillText(
        activeLayout.explicitRealization ? "TEMPORAL EVENT ORDER →" : "NORMALIZED TIME →",
        bounds.minX,
        bounds.minY - 42
      );
      ctx.restore();
    }

    function drawDAGGuides(ctx, activeLayout) {
      if (currentView !== "dag" || viewport.scale < .04 ||
          !activeLayout.rankCenters?.length) return;
      const bounds = activeLayout.bounds;
      ctx.save();
      ctx.setLineDash([4, 6]);
      ctx.strokeStyle = "#c4cad2";
      ctx.fillStyle = "#647080";
      ctx.font = "700 9px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      activeLayout.rankCenters.forEach((coordinate, rank) => {
        ctx.beginPath();
        ctx.moveTo(coordinate, bounds.minY - 16);
        ctx.lineTo(coordinate, bounds.maxY + 16);
        ctx.stroke();
        ctx.fillText(`LEVEL ${rank}`, coordinate, bounds.minY - 22);
      });
      ctx.restore();
    }

    function drawPhysicalMeshGuides(ctx, activeLayout) {
      if (currentView !== "physical" || !activeLayout.physicalPlacement) return;
      const placement = activeLayout.physicalPlacement;
      ctx.save();
      ctx.strokeStyle = "#cbd2da";
      ctx.fillStyle = "#f8fafc";
      ctx.lineWidth = 1;
      ctx.font = "700 9px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      for (let row = 0; row < placement.mesh_rows; ++row) {
        for (let column = 0; column < placement.mesh_cols; ++column) {
          const x = activeLayout.margin + column * activeLayout.cellWidth;
          const y = activeLayout.margin + row * activeLayout.cellHeight;
          ctx.fillRect(x, y, activeLayout.cellWidth, activeLayout.cellHeight);
          ctx.strokeRect(x, y, activeLayout.cellWidth, activeLayout.cellHeight);
          if (viewport.scale >= .42) {
            ctx.fillStyle = "#7b8491";
            ctx.fillText(`CORE ${row * placement.mesh_cols + column}`, x + 7, y + 6);
          }
          ctx.fillStyle = "#f8fafc";
        }
      }
      ctx.fillStyle = "#445064";
      ctx.font = "800 10px sans-serif";
      ctx.fillText(
        `${placement.mesh_rows} × ${placement.mesh_cols} PHYSICAL MESH`,
        activeLayout.margin,
        activeLayout.margin - 22
      );
      ctx.restore();
    }

    function drawLink(ctx, link) {
      const source = link.source;
      const target = link.target;
      const sourceVisible = nodeVisible(source);
      const targetVisible = nodeVisible(target);
      const selectedKey = selectedNode?.key;
      const incident = selectedKey !== undefined &&
        (source.key === selectedKey || target.key === selectedKey);
      const denseOverview = viewport.scale < .08;
      ctx.globalAlpha = sourceVisible && targetVisible
        ? currentView === "tree"
          ? denseOverview ? .26 : .62
          : denseOverview ? .10 : .48
        : .025;
      if (link.aggregate) {
        ctx.globalAlpha = .38;
        ctx.lineWidth = Math.min(
          2.6,
          1 + Math.log2(Math.max(1, link.edgeCount)) / 5
        ) / viewport.scale;
      } else if (link.kind === "tile_communication") {
        ctx.globalAlpha = selectedNode?.logicalTile
          ? incident ? .82 : .006
          : viewport.scale < .5 ? .045 : .12;
        ctx.lineWidth = Math.min(
          6,
          1 + Math.log2(Math.max(1, link.placedEdge.byte_size)) / 7
        );
      } else {
        ctx.lineWidth = link.kind === "dependency" || link.kind === "physical" ? 1.4 : 1.1;
        if (selectedKey !== undefined)
          ctx.globalAlpha = incident ? .88 : Math.min(ctx.globalAlpha, .035);
      }
      ctx.lineWidth = currentView === "tree"
        ? (incident ? 2.2 : 1.2) / viewport.scale
        : Math.max(ctx.lineWidth, .35 / viewport.scale);
      ctx.strokeStyle = link.aggregate
        ? link.kind === "tile_communication" ? "#6c5885" : "#788696"
        : currentView === "tree" ? "#657487"
        : link.kind === "dependency" || link.kind === "tile_communication"
        ? "#6d3aa8"
        : link.kind === "physical" ? "#4f5b6a" : "#8993a1";
      ctx.beginPath();
      if (currentView === "tree" && !link.aggregate) {
        const startY = source.y + source.height / 2;
        const endY = target.y - target.height / 2;
        const midpointY = (startY + endY) / 2;
        ctx.moveTo(source.x, startY);
        ctx.lineTo(source.x, midpointY);
        ctx.lineTo(target.x, midpointY);
        ctx.lineTo(target.x, endY);
      } else if (link.kind === "ra_summary") {
        const startY = source.y + source.height / 2;
        const endY = target.y - target.height / 2;
        const midpointY = (startY + endY) / 2;
        ctx.moveTo(source.x, startY);
        ctx.lineTo(source.x, midpointY);
        ctx.lineTo(target.x, midpointY);
        ctx.lineTo(target.x, endY);
      } else if (link.kind === "summary_dependency") {
        if (Math.abs(source.y - target.y) > 1) {
          const startY = source.y + source.height / 2;
          const endY = target.y - target.height / 2;
          const midpointY = (startY + endY) / 2;
          ctx.moveTo(source.x, startY);
          ctx.lineTo(source.x, midpointY);
          ctx.lineTo(target.x, midpointY);
          ctx.lineTo(target.x, endY);
        } else {
          const direction = target.x >= source.x ? 1 : -1;
          const startX = source.x + direction * source.width / 2;
          const endX = target.x - direction * target.width / 2;
          const midpointX = (startX + endX) / 2;
          ctx.moveTo(startX, source.y);
          ctx.lineTo(midpointX, source.y);
          ctx.lineTo(midpointX, target.y);
          ctx.lineTo(endX, target.y);
        }
      } else if (link.kind === "tile_communication") {
        ctx.moveTo(source.x, source.y);
        const midpointX = (source.x + target.x) / 2;
        ctx.bezierCurveTo(midpointX, source.y, midpointX, target.y, target.x, target.y);
      } else if (link.kind === "dependency") {
        const startX = source.x + source.width / 2;
        const endX = target.x - target.width / 2;
        const bend = Math.max(22, (endX - startX) * .42);
        ctx.moveTo(startX, source.y);
        ctx.bezierCurveTo(startX + bend, source.y, endX - bend, target.y, endX, target.y);
      } else if (link.kind === "physical" && currentView !== "tree") {
        ctx.moveTo(source.x + source.width / 2, source.y);
        ctx.lineTo(target.x - target.width / 2, target.y);
      } else {
        ctx.moveTo(source.x, source.y + source.height / 2);
        ctx.lineTo(target.x, target.y - target.height / 2);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    function nodeLabel(node) {
      if (node.aggregate) return node.aggregateLabel;
      if (node.logicalTile)
        return `LOGICAL ${node.logicalTile.id}`;
      if (node.realization) return `stage ${node.realization.stage_index}`;
      if (!node.operation) {
        const waveId = node.treeNode?.mvm_wave_id ?? -1;
        if (waveId >= 0) {
          return node.kind === "temporal_cut"
            ? `WAVE ${waveId} · T-CUT`
            : `WAVE ${waveId} · S-CUT`;
        }
        return node.kind === "temporal_cut" ? `T-CUT ${node.id}` : `S-CUT ${node.id}`;
      }
      return `op ${node.operation.id}`;
    }

    function shortOperationName(operation) {
      return operation.name.replace(/^.*\./, "");
    }

    function shortRealizationKind(stage) {
      if (stage.kind === "sculptor.matrix_setup") return "matrix setup";
      if (stage.kind === "digital.vector_tile") return "vector tile";
      if (stage.kind === "sculptor.mvm") return "array MVM";
      if (stage.kind === "digital.tile_recombine") return "recombine";
      return stage.kind.replace(/^.*\./, "");
    }

    function drawNode(ctx, node) {
      const visible = nodeVisible(node);
      const selected = selectedNode?.key === node.key;
      const hovered = hoveredNode?.key === node.key;
      const screenWidth = node.width * viewport.scale;
      const screenHeight = node.height * viewport.scale;
      ctx.globalAlpha = visible ? 1 : .16;
      ctx.fillStyle = node.aggregate ? "#f7f9fb" : nodeColor(node);
      if ((selected || hovered) && (screenWidth < 18 || screenHeight < 14)) {
        const marker = (selected ? 16 : 12) / viewport.scale;
        ctx.fillRect(node.x - marker / 2, node.y - marker / 2, marker, marker);
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = (selected ? 3 : 2) / viewport.scale;
        ctx.strokeRect(node.x - marker / 2, node.y - marker / 2, marker, marker);
        ctx.fillStyle = "#111827";
        ctx.font = `800 ${10 / viewport.scale}px sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(
          nodeLabel(node),
          node.x + marker / 2 + 6 / viewport.scale,
          node.y
        );
        ctx.globalAlpha = 1;
        return;
      }
      if (!selected && !hovered && (screenWidth < 7 || screenHeight < 6)) {
        const marker = 2.2 / viewport.scale;
        ctx.globalAlpha = visible ? .82 : .10;
        ctx.fillRect(node.x - marker / 2, node.y - marker / 2, marker, marker);
        ctx.globalAlpha = 1;
        return;
      }
      ctx.strokeStyle = selected || hovered
        ? "#111827"
        : node.aggregate ? node.aggregateColor : "rgba(24, 32, 43, .48)";
      ctx.lineWidth = node.aggregate
        ? (selected ? 3 : hovered ? 2 : 1.2) / viewport.scale
        : (selected ? 3 : hovered ? 2 : 1) / viewport.scale;
      ctx.fillRect(node.x - node.width / 2, node.y - node.height / 2, node.width, node.height);
      ctx.strokeRect(node.x - node.width / 2, node.y - node.height / 2, node.width, node.height);
      if (node.aggregate) {
        ctx.fillStyle = node.aggregateColor;
        ctx.fillRect(
          node.x - node.width / 2,
          node.y - node.height / 2,
          5 / viewport.scale,
          node.height
        );
      }
      ctx.fillStyle = node.aggregate || node.logicalTile ? "#243142" : "#fff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = node.aggregate
        ? `700 ${10 / viewport.scale}px sans-serif`
        : `700 ${(node.operation ? 10 : 11) / viewport.scale}px sans-serif`;
      const showPrimaryLabel = selected || hovered ||
        (currentView === "tree"
          ? screenWidth >= 24 && screenHeight >= 22
          : screenWidth >= 52 && screenHeight >= 22);
      if (showPrimaryLabel) {
        let label = nodeLabel(node);
        if (currentView === "tree" && node.operation && screenWidth < 42) {
          label = String(node.operation.id);
          ctx.font = `800 ${8 / viewport.scale}px sans-serif`;
        }
        const maximumCharacters = Math.max(
          currentView === "tree" ? 3 : 5,
          Math.floor(
            (screenWidth - (currentView === "tree" ? 6 : 12)) /
            (currentView === "tree" ? 5 : 6.1)
          )
        );
        if (label.length > maximumCharacters)
          label = `${label.slice(0, Math.max(2, maximumCharacters - 1))}…`;
        ctx.fillText(
          label,
          node.x,
          node.y - (node.operation || node.logicalTile || node.aggregate
            ? 7 / viewport.scale
            : 0)
        );
      }
      if (node.aggregate) {
        if (node.aggregateType !== "ra_subtree" &&
            node.width * viewport.scale >= 110) {
          ctx.font = `${8.5 / viewport.scale}px sans-serif`;
          ctx.fillText(
            `${node.operationIds.size.toLocaleString()} operations`,
            node.x,
            node.y + 9 / viewport.scale
          );
        }
      } else if (node.logicalTile && screenWidth >= 94 && screenHeight >= 46) {
        const analogAssignments = node.logicalTile.analog_lanes.reduce(
          (sum, lane) => sum + lane.assignments.length,
          0
        );
        ctx.font = `${9 / viewport.scale}px sans-serif`;
        ctx.fillText(
          `P${node.physicalAssignment.physical_tile_id} · D${node.logicalTile.digital_assignments.length} · A${analogAssignments}`,
          node.x,
          node.y + 10 / viewport.scale
        );
      } else if (node.operation && screenWidth >= 94 && screenHeight >= 34) {
        ctx.font = `${9 / viewport.scale}px sans-serif`;
        const name = node.realization
          ? shortRealizationKind(node.realization)
          : shortOperationName(node.operation);
        ctx.fillText(name.length > 12 ? `${name.slice(0, 11)}…` : name,
                     node.x, node.y + 9 / viewport.scale);
      }
      ctx.globalAlpha = 1;
    }

    function visibleWorldBounds(size, paddingPixels = 80) {
      const padding = paddingPixels / viewport.scale;
      return {
        minX: -viewport.x / viewport.scale - padding,
        minY: -viewport.y / viewport.scale - padding,
        maxX: (size.width - viewport.x) / viewport.scale + padding,
        maxY: (size.height - viewport.y) / viewport.scale + padding,
      };
    }

    function nodeIntersectsBounds(node, bounds) {
      return node.x + node.width / 2 >= bounds.minX &&
        node.x - node.width / 2 <= bounds.maxX &&
        node.y + node.height / 2 >= bounds.minY &&
        node.y - node.height / 2 <= bounds.maxY;
    }

    function linkIntersectsBounds(link, bounds) {
      return Math.max(link.source.x, link.target.x) >= bounds.minX &&
        Math.min(link.source.x, link.target.x) <= bounds.maxX &&
        Math.max(link.source.y, link.target.y) >= bounds.minY &&
        Math.min(link.source.y, link.target.y) <= bounds.maxY;
    }

    function drawDenseLinkLayer(ctx, links) {
      ctx.save();
      ctx.globalAlpha = currentView === "tree"
        ? .24
        : currentView === "st" ? .10 : .08;
      ctx.strokeStyle = currentView === "tree" ? "#7d8794" : "#73509b";
      ctx.lineWidth = (currentView === "tree" ? 1.05 : .42) / viewport.scale;
      ctx.beginPath();
      for (const link of links) {
        if (!nodeVisible(link.source) || !nodeVisible(link.target)) continue;
        const source = link.source;
        const target = link.target;
        if (link.kind === "dependency") {
          const startX = source.x + source.width / 2;
          const endX = target.x - target.width / 2;
          const bend = Math.max(22, (endX - startX) * .42);
          ctx.moveTo(startX, source.y);
          ctx.bezierCurveTo(
            startX + bend, source.y,
            endX - bend, target.y,
            endX, target.y
          );
        } else {
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
        }
      }
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      const size = canvasSize();
      const ratio = window.devicePixelRatio || 1;
      renderedLayout = getRenderedLayout();
      const visibleBounds = visibleWorldBounds(size);
      const visibleNodes = renderedLayout.nodes.filter(node =>
        nodeIntersectsBounds(node, visibleBounds)
      );
      const visibleLinks = renderedLayout.links.filter(link =>
        linkIntersectsBounds(link, visibleBounds)
      );
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, size.width, size.height);
      context.save();
      context.translate(viewport.x, viewport.y);
      context.scale(viewport.scale, viewport.scale);
      drawTreeGuides(context, renderedLayout);
      drawDAGGuides(context, renderedLayout);
      drawSTGuides(context, renderedLayout);
      drawPhysicalMeshGuides(context, renderedLayout);
      if (viewport.scale < .04 && selectedNode === null)
        drawDenseLinkLayer(context, visibleLinks);
      else
        for (const link of visibleLinks) drawLink(context, link);
      for (const node of visibleNodes) drawNode(context, node);
      context.restore();
      const viewName = currentView === "tree"
        ? "RA hierarchy"
        : currentView === "dag" ? "Compute DAG"
        : currentView === "physical" ? "Physical mesh"
        : layout.explicitRealization ? "Realized S-T graph"
        : "Structural S-T graph";
      const laneSummary = currentView === "st"
        ? ` · ${renderedLayout.resourceLaneCount} resource lane${renderedLayout.resourceLaneCount === 1 ? "" : "s"}`
        : "";
      const placementSummary = currentView === "physical" && layout.physicalPlacement
        ? ` · score ${layout.physicalPlacement.total_transfer_cost.toLocaleString()}`
        : "";
      const graphSummary = ` · full graph · ${renderedLayout.nodes.length.toLocaleString()} nodes · ` +
        `${renderedLayout.links.length.toLocaleString()} edges · ` +
        `${visibleNodes.length.toLocaleString()} nodes visible`;
      const scalePercent = viewport.scale * 100;
      const scaleLabel = scalePercent < 1
        ? `${scalePercent.toFixed(1)}%`
        : `${Math.round(scalePercent)}%`;
      document.querySelector("#canvas-status").textContent =
        `${viewName}${graphSummary}${laneSummary}${placementSummary} · ` +
        scaleLabel;
    }

    function zoomAt(factor, clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const worldX = (x - viewport.x) / viewport.scale;
      const worldY = (y - viewport.y) / viewport.scale;
      viewport.scale = Math.min(
        6,
        Math.max(currentView === "tree" ? .0001 : .001, viewport.scale * factor)
      );
      if (currentView === "tree") projectTreeLayout(canvasSize());
      viewport.x = x - worldX * viewport.scale;
      if (currentView === "tree") {
        const size = canvasSize();
        const worldHeight = layout.bounds.maxY - layout.bounds.minY;
        viewport.y = (size.height - worldHeight * viewport.scale) / 2 -
          layout.bounds.minY * viewport.scale;
      } else {
        viewport.y = y - worldY * viewport.scale;
      }
      draw();
    }

    function pointerDistance(first, second) {
      return Math.hypot(second.x - first.x, second.y - first.y);
    }

    function beginPinchGesture() {
      const pointers = [...activePointers.entries()].slice(0, 2);
      if (pointers.length !== 2) return;
      const first = pointers[0][1];
      const second = pointers[1][1];
      const rect = canvas.getBoundingClientRect();
      const midpointX = (first.x + second.x) / 2 - rect.left;
      const midpointY = (first.y + second.y) / 2 - rect.top;
      pinchGesture = {
        pointerIds: [pointers[0][0], pointers[1][0]],
        startDistance: Math.max(1, pointerDistance(first, second)),
        startScale: viewport.scale,
        worldX: (midpointX - viewport.x) / viewport.scale,
        worldY: (midpointY - viewport.y) / viewport.scale,
      };
      dragging = null;
      tooltip.hidden = true;
      canvas.classList.add("dragging");
    }

    function updatePinchGesture() {
      if (!pinchGesture) return false;
      const first = activePointers.get(pinchGesture.pointerIds[0]);
      const second = activePointers.get(pinchGesture.pointerIds[1]);
      if (!first || !second) return false;
      const rect = canvas.getBoundingClientRect();
      const midpointX = (first.x + second.x) / 2 - rect.left;
      const midpointY = (first.y + second.y) / 2 - rect.top;
      const scale = pinchGesture.startScale *
        pointerDistance(first, second) / pinchGesture.startDistance;
      viewport.scale = Math.min(
        6,
        Math.max(currentView === "tree" ? .0001 : .02, scale)
      );
      if (currentView === "tree") projectTreeLayout(canvasSize());
      viewport.x = midpointX - pinchGesture.worldX * viewport.scale;
      if (currentView === "tree") {
        const size = canvasSize();
        const worldHeight = layout.bounds.maxY - layout.bounds.minY;
        viewport.y = (size.height - worldHeight * viewport.scale) / 2 -
          layout.bounds.minY * viewport.scale;
      } else {
        viewport.y = midpointY - pinchGesture.worldY * viewport.scale;
      }
      draw();
      return true;
    }

    function beginDrag(pointerId, point, moved = false) {
      dragging = {
        pointerId,
        startX: point.x,
        startY: point.y,
        viewX: viewport.x,
        viewY: viewport.y,
        moved,
      };
      canvas.classList.add("dragging");
    }

    function finishPointer(event, cancelled) {
      const wasPinching = pinchGesture !== null;
      const completedDrag = dragging?.pointerId === event.pointerId
        ? dragging
        : null;
      activePointers.delete(event.pointerId);
      if (canvas.hasPointerCapture?.(event.pointerId))
        canvas.releasePointerCapture(event.pointerId);

      if (wasPinching) {
        pinchGesture = null;
        if (activePointers.size >= 2) {
          beginPinchGesture();
        } else if (activePointers.size === 1) {
          const [pointerId, point] = activePointers.entries().next().value;
          beginDrag(pointerId, point, true);
        } else {
          dragging = null;
          canvas.classList.remove("dragging");
        }
        draw();
        return;
      }

      if (completedDrag && !cancelled && !completedDrag.moved) {
        const hit = nodeAt(event.clientX, event.clientY);
        selectedNode = hit;
        renderInspector(selectedNode);
      }
      if (completedDrag)
        dragging = null;
      if (activePointers.size === 0)
        canvas.classList.remove("dragging");
      draw();
    }

    function centerNode(node) {
      const size = canvasSize();
      viewport.scale = Math.max(viewport.scale, currentView === "tree" ? .13 : 1.15);
      if (currentView === "tree") projectTreeLayout(size);
      viewport.x = size.width / 2 - node.x * viewport.scale;
      if (currentView === "tree") {
        const worldHeight = layout.bounds.maxY - layout.bounds.minY;
        viewport.y = (size.height - worldHeight * viewport.scale) / 2 -
          layout.bounds.minY * viewport.scale;
      } else {
        viewport.y = size.height / 2 - node.y * viewport.scale;
      }
      selectedNode = node;
      renderInspector(node);
      draw();
    }

    function tensorRows(ids) {
      return ids.map(id => {
        const tensor = currentFunction.tensorMap.get(id);
        if (!tensor) return "";
        const boundary = tensor.is_function_input ? "input" : tensor.is_function_output ? "output" : "internal";
        return `<tr><td>${tensor.id}</td><td>${escapeHTML(tensor.type)}</td><td>${formatBytes(tensor.byte_size)}</td><td>${boundary}</td></tr>`;
      }).join("");
    }

    function renderPhysicalTileInspector(node, title, body) {
      const tile = node.logicalTile;
      const assignment = node.physicalAssignment;
      const placement = currentFunction.physicalPlacement;
      const incidentEdges = placement.edges.filter(edge =>
        edge.source_tile_id === tile.id || edge.target_tile_id === tile.id
      );
      const incoming = incidentEdges.filter(edge => edge.target_tile_id === tile.id);
      const outgoing = incidentEdges.filter(edge => edge.source_tile_id === tile.id);
      const incomingBytes = incoming.reduce((sum, edge) => sum + edge.byte_size, 0);
      const outgoingBytes = outgoing.reduce((sum, edge) => sum + edge.byte_size, 0);
      const operations = [
        ...tile.digital_assignments.map(entry => ({lane: "digital", entry})),
        ...tile.analog_lanes.flatMap(lane => lane.assignments.map(entry => ({
          lane: `analog ${lane.lane_index}`,
          entry,
        }))),
      ];

      title.textContent =
        `Core ${assignment.physical_tile_id} · logical tile ${tile.id}`;
      let html = `<section class="detail-section"><h3>Physical assignment</h3><dl class="detail-grid">`;
      html += `<dt>Schedule</dt><dd>${escapeHTML(placement.schedule)}</dd>`;
      html += `<dt>Coordinate</dt><dd>(${assignment.row}, ${assignment.column})</dd>`;
      html += `<dt>Mesh</dt><dd>${placement.mesh_rows} × ${placement.mesh_cols}</dd>`;
      html += `<dt>Arrays per core</dt><dd>${placement.arrays_per_core}</dd>`;
      html += `<dt>Digital work</dt><dd>${tile.digital_work}</dd>`;
      html += `<dt>Model inputs</dt><dd>${tile.model_input_tensor_ids.length}</dd>`;
      html += `<dt>Model outputs</dt><dd>${tile.model_output_tensor_ids.length}</dd>`;
      html += `</dl></section>`;

      html += `<section class="detail-section"><h3>Assigned operations</h3>`;
      html += `<table class="compact-table"><thead><tr><th>Lane</th><th>Operation</th><th>Name</th></tr></thead><tbody>`;
      html += operations.map(({lane, entry}) => {
        const operation = currentFunction.operationMap.get(entry.operation_id);
        return `<tr><td>${escapeHTML(lane)}</td><td>${entry.operation_id}</td>` +
          `<td>${escapeHTML(operation?.name || "-")}</td></tr>`;
      }).join("");
      html += `</tbody></table></section>`;

      html += `<section class="detail-section"><h3>Analog lane bindings</h3>`;
      html += `<table class="compact-table"><thead><tr><th>Lane</th><th>Binding</th><th>Operations</th></tr></thead><tbody>`;
      html += tile.analog_lanes.map(lane =>
        `<tr><td>${lane.lane_index}</td>` +
        `<td>${lane.lane_binding_group >= 0 ? lane.lane_binding_group : "-"}</td>` +
        `<td>${lane.assignments.map(entry => entry.operation_id).join(", ") || "-"}</td></tr>`
      ).join("");
      html += `</tbody></table></section>`;

      html += `<section class="detail-section"><h3>Communication</h3><dl class="detail-grid">`;
      html += `<dt>Incoming</dt><dd>${formatBytes(incomingBytes)} across ${incoming.length} edges</dd>`;
      html += `<dt>Outgoing</dt><dd>${formatBytes(outgoingBytes)} across ${outgoing.length} edges</dd>`;
      html += `</dl>`;
      if (incidentEdges.length) {
        html += `<table class="compact-table"><thead><tr><th>Direction</th><th>Peer</th><th>Bytes</th><th>Hops</th><th>Cost</th></tr></thead><tbody>`;
        html += incidentEdges.map(edge => {
          const isOutgoing = edge.source_tile_id === tile.id;
          const peerLogicalTile = isOutgoing ? edge.target_tile_id : edge.source_tile_id;
          const peer = currentFunction.physicalAssignmentMap.get(peerLogicalTile);
          return `<tr><td>${isOutgoing ? "out" : "in"}</td>` +
            `<td>L${peerLogicalTile} / C${peer?.physical_tile_id ?? "-"}</td>` +
            `<td>${formatBytes(edge.byte_size)}</td>` +
            `<td>${edge.manhattan_hops}</td>` +
            `<td>${edge.transfer_cost.toLocaleString()}</td></tr>`;
        }).join("");
        html += `</tbody></table>`;
      }
      html += `</section>`;
      body.innerHTML = html;
    }

    function renderInspector(node) {
      const title = document.querySelector("#inspector-title");
      const body = document.querySelector("#inspector-body");
      if (!node) {
        title.textContent = "No selection";
        body.innerHTML = "";
        return;
      }
      if (node.aggregate) {
        const kind = node.aggregateType === "ra_subtree" ? "RA subtree" :
          node.aggregateType === "dag_stage" ? "DAG stage" : "S-T lane group";
        title.textContent = `${kind} · ${node.aggregateLabel}`;
        body.innerHTML =
          `<section class="detail-section"><h3>${escapeHTML(kind)}</h3><dl class="detail-grid">` +
          `<dt>Detail</dt><dd>${escapeHTML(node.tier)}</dd>` +
          `<dt>Items</dt><dd>${node.members.length.toLocaleString()}</dd>` +
          `<dt>Operations</dt><dd>${node.operationIds.size.toLocaleString()}</dd>` +
          `<dt>Logical tiles</dt><dd>${node.logicalTileIds.size.toLocaleString()}</dd>` +
          `<dt>Cut nodes</dt><dd>${node.cutCount.toLocaleString()}</dd>` +
          `</dl></section>`;
        return;
      }
      if (node.logicalTile) {
        renderPhysicalTileInspector(node, title, body);
        return;
      }

      const treeNode = node.treeNode;
      const operation = node.operation;
      const evaluation = treeNode
        ? currentFunction.planNodeEvaluationMap.get(treeNode.id)
        : null;
      const workUnit = treeNode?.work_unit_id >= 0
        ? currentFunction.workUnitMap.get(treeNode.work_unit_id)
        : null;
      title.textContent = node.realization
        ? `Operation ${operation.id} · stage ${node.realization.stage_index}`
        : operation
          ? `Operation ${operation.id}${workUnit ? ` · tile ${workUnit.id}` : ""}`
          : `RA node ${treeNode.id}`;
      let html = `<section class="detail-section"><h3>Tree identity</h3><dl class="detail-grid">`;
      html += `<dt>Node</dt><dd>${treeNode?.id ?? "-"}</dd>`;
      html += `<dt>Kind</dt><dd>${escapeHTML(treeNode?.kind ?? node.kind)}</dd>`;
      html += `<dt>Parent</dt><dd>${treeNode?.parent_id ?? "-"}</dd>`;
      html += `<dt>Children</dt><dd>${treeNode?.child_ids?.length ?? 0}</dd>`;
      html += `<dt>Work groups</dt><dd>${treeNode?.work_group_count ?? 1}</dd>`;
      html += `<dt>Work unit</dt><dd>${workUnit?.id ?? "-"}</dd>`;
      html += `<dt>MVM wave</dt><dd>${(treeNode?.mvm_wave_id ?? -1) >= 0 ? treeNode.mvm_wave_id : "-"}</dd>`;
      html += `</dl></section>`;

      if (workUnit) {
        html += `<section class="detail-section"><h3>Selected tile</h3><dl class="detail-grid">`;
        html += `<dt>Result</dt><dd>${workUnit.result_number}</dd>`;
        html += `<dt>Result offsets</dt><dd>[${workUnit.result_offsets.join(", ")}]</dd>`;
        html += `<dt>Result sizes</dt><dd>[${workUnit.result_sizes.join(", ")}]</dd>`;
        html += `<dt>Iteration offsets</dt><dd>[${workUnit.iteration_offsets.join(", ")}]</dd>`;
        html += `<dt>Iteration sizes</dt><dd>[${workUnit.iteration_sizes.join(", ")}]</dd>`;
        html += `</dl></section>`;
      }

      if (evaluation) {
        html += `<section class="detail-section"><h3>Reference evaluation</h3><dl class="detail-grid">`;
        html += `<dt>Feasible</dt><dd>${evaluation.feasible ? "yes" : "no"}</dd>`;
        html += `<dt>Latency</dt><dd>${formatNanoseconds(evaluation.estimated_latency_ns)}</dd>`;
        html += `<dt>Crossing bytes</dt><dd>${formatBytes(evaluation.crossing_bytes)}</dd>`;
        html += `<dt>Communication</dt><dd>${formatNanoseconds(evaluation.estimated_communication_ns)}</dd>`;
        html += `<dt>Resources</dt><dd>${evaluation.required_resource_units}</dd>`;
        html += `<dt>Pipeline stages</dt><dd>${evaluation.pipeline_stages}</dd>`;
        if (evaluation.infeasibility_reason) {
          html += `<dt>Reason</dt><dd>${escapeHTML(evaluation.infeasibility_reason)}</dd>`;
        }
        html += `</dl></section>`;
      }

      if (operation) {
        html += `<section class="detail-section"><h3>Operation</h3><dl class="detail-grid">`;
        html += `<dt>ID</dt><dd>${operation.id}</dd>`;
        html += `<dt>Name</dt><dd>${escapeHTML(operation.name)}</dd>`;
        html += `<dt>Kind</dt><dd>${escapeHTML(operation.kind)}</dd>`;
        html += `<dt>Required lane</dt><dd>${escapeHTML(operation.required_lane)}</dd>`;
        html += `<dt>Lane-binding group</dt><dd>${operation.lane_binding_group >= 0 ? operation.lane_binding_group : "-"}</dd>`;
        const assignment = node.mappingAssignment;
        html += `<dt>S-T assignment</dt><dd>${assignment ? `tile ${assignment.tile_id} / ${assignment.lane_kind} ${assignment.lane_index}` : "-"}</dd>`;
        html += `<dt>MVM wave</dt><dd>${operation.mvm_wave_id >= 0 ? operation.mvm_wave_id : "-"}</dd>`;
        html += `<dt>Wave member</dt><dd>${operation.mvm_wave_member >= 0 ? `${operation.mvm_wave_member + 1} / ${operation.mvm_wave_size}` : "-"}</dd>`;
        html += `<dt>Location</dt><dd>${escapeHTML(operation.location)}</dd>`;
        html += `<dt>Inputs</dt><dd>${operation.input_tensors.length}</dd>`;
        html += `<dt>Outputs</dt><dd>${operation.output_tensors.length}</dd>`;
        html += `</dl></section>`;

        if (operation.analog_mvm) {
          html += `<section class="detail-section"><h3>Logical analog MVM</h3><dl class="detail-grid">`;
          html += `<dt>Output rows</dt><dd>${operation.analog_mvm.output_rows}</dd>`;
          html += `<dt>Input columns</dt><dd>${operation.analog_mvm.input_columns}</dd>`;
          html += `</dl></section>`;
        }

        const realizationStages =
          currentFunction.realizationByOperation.get(operation.id) || [];
        if (realizationStages.length) {
          html += `<section class="detail-section"><h3>Golem realization</h3>`;
          html += `<table class="compact-table"><thead><tr><th>Stage</th><th>Kind</th><th>Array calls</th></tr></thead><tbody>`;
          html += realizationStages.map(stage => {
            const arrayCalls = stage.array_set_count + stage.array_load_count +
              stage.array_execute_count + stage.array_store_count;
            return `<tr><td>${stage.stage_index}</td><td>${escapeHTML(shortRealizationKind(stage))}</td><td>${arrayCalls}</td></tr>`;
          }).join("");
          html += `</tbody></table></section>`;
        }

        if (node.realization) {
          const stage = node.realization;
          html += `<section class="detail-section"><h3>Selected physical stage</h3><dl class="detail-grid">`;
          html += `<dt>Kind</dt><dd>${escapeHTML(stage.kind)}</dd>`;
          html += `<dt>Name</dt><dd>${escapeHTML(stage.name || "-")}</dd>`;
          html += `<dt>Inputs</dt><dd>${stage.input_count}</dd>`;
          html += `<dt>Outputs</dt><dd>${stage.output_count}</dd>`;
          html += `<dt>Array set</dt><dd>${stage.array_set_count}</dd>`;
          html += `<dt>Array load</dt><dd>${stage.array_load_count}</dd>`;
          html += `<dt>Array execute</dt><dd>${stage.array_execute_count}</dd>`;
          html += `<dt>Array store</dt><dd>${stage.array_store_count}</dd>`;
          html += `</dl></section>`;
          const physicalAttributes = Object.entries(stage.attributes || {});
          if (physicalAttributes.length) {
            html += `<section class="detail-section"><h3>Physical geometry</h3><div class="tag-list">`;
            html += physicalAttributes.map(([key, value]) =>
              `<span class="tag">${escapeHTML(key)}=${escapeHTML(value)}</span>`
            ).join("");
            html += `</div></section>`;
          }
          html += `<section class="detail-section"><h3>Expanded MLIR</h3><pre>${escapeHTML(stage.mlir)}</pre></section>`;
        }

        if (currentView === "st") {
          html += `<section class="detail-section"><h3>${node.mappingAssignment ? "Realized S-T position" : "Structural S-T position"}</h3><dl class="detail-grid">`;
          html += `<dt>Resource lane</dt><dd>${node.stLane}</dd>`;
          html += `<dt>Lane-binding group</dt><dd>${node.stLaneBindingGroup >= 0 ? node.stLaneBindingGroup : "-"}</dd>`;
          if (node.mappingAssignment) {
            html += `<dt>Logical tile</dt><dd>${node.mappingAssignment.tile_id}</dd>`;
            html += `<dt>Lane kind</dt><dd>${escapeHTML(node.mappingAssignment.lane_kind)}</dd>`;
            html += `<dt>Lane index</dt><dd>${node.mappingAssignment.lane_index}</dd>`;
            html += `<dt>Estimated start</dt><dd>${formatNanoseconds(node.stStart)}</dd>`;
            html += `<dt>Estimated duration</dt><dd>${formatNanoseconds(node.stDuration)}</dd>`;
          } else {
            html += `<dt>Pipeline stage</dt><dd>${node.stStage}</dd>`;
            html += `<dt>Normalized start</dt><dd>${node.stStart.toFixed(2)}</dd>`;
            html += `<dt>Normalized duration</dt><dd>${node.stDuration.toFixed(2)}</dd>`;
            html += `<dt>Interpretation</dt><dd>structural, not measured</dd>`;
          }
          html += `</dl></section>`;
        }

        const semanticEntries = Object.entries(operation.semantic || {});
        if (semanticEntries.length) {
          html += `<section class="detail-section"><h3>Semantic provenance</h3><div class="tag-list">`;
          html += semanticEntries.map(([key, value]) =>
            `<span class="tag">${escapeHTML(key)}=${escapeHTML(value)}</span>`
          ).join("");
          html += `</div></section>`;
        }

        if (operation.iteration_domain.length) {
          html += `<section class="detail-section"><h3>Iteration domain</h3><table class="compact-table"><thead><tr><th>Loop</th><th>Kind</th><th>Extent</th></tr></thead><tbody>`;
          html += operation.iteration_domain.map(dimension =>
            `<tr><td>${dimension.loop}</td><td>${escapeHTML(dimension.kind)}</td><td>${dimension.extent < 0 ? "dynamic" : dimension.extent}</td></tr>`
          ).join("");
          html += `</tbody></table></section>`;
        }

        const tensorIds = [...new Set([...operation.input_tensors, ...operation.output_tensors])];
        if (tensorIds.length) {
          html += `<section class="detail-section"><h3>Tensors</h3><table class="compact-table"><thead><tr><th>ID</th><th>Type</th><th>Bytes</th><th>Role</th></tr></thead><tbody>${tensorRows(tensorIds)}</tbody></table></section>`;
        }
        html += `<section class="detail-section"><h3>MLIR</h3><pre>${escapeHTML(operation.mlir)}</pre></section>`;
      }
      body.innerHTML = html;
    }

    function showTooltip(node, event) {
      if (!node) {
        tooltip.hidden = true;
        return;
      }
      if (node.aggregate) {
        tooltip.innerHTML =
          `<strong>${escapeHTML(node.aggregateLabel)}</strong>` +
          `<span>${node.operationIds.size.toLocaleString()} operations · ` +
          `${node.logicalTileIds.size.toLocaleString()} logical tiles</span>`;
        const panel = canvas.parentElement.getBoundingClientRect();
        tooltip.style.left =
          `${Math.min(event.clientX - panel.left + 12, panel.width - 275)}px`;
        tooltip.style.top =
          `${Math.min(event.clientY - panel.top + 12, panel.height - 75)}px`;
        tooltip.hidden = false;
        return;
      }
      if (node.logicalTile) {
        const assignment = node.physicalAssignment;
        tooltip.innerHTML =
          `<strong>Core ${assignment.physical_tile_id} · logical tile ${node.logicalTile.id}</strong>` +
          `<span>(${assignment.row}, ${assignment.column}) · ` +
          `${node.logicalTile.digital_assignments.length} digital assignments</span>`;
        const panel = canvas.parentElement.getBoundingClientRect();
        tooltip.style.left =
          `${Math.min(event.clientX - panel.left + 12, panel.width - 275)}px`;
        tooltip.style.top =
          `${Math.min(event.clientY - panel.top + 12, panel.height - 75)}px`;
        tooltip.hidden = false;
        return;
      }
      const operation = node.operation;
      const heading = node.realization
        ? `Operation ${operation.id} · stage ${node.realization.stage_index}`
        : operation ? `Operation ${operation.id}` : `RA node ${node.treeNode.id}`;
      const position = currentView === "st" && operation
        ? ` · lane ${node.stLane}, t=${node.stStart.toFixed(2)}`
        : "";
      const detail = node.realization
        ? `${shortRealizationKind(node.realization)} · ${node.realization.name || "unnamed"}`
        : operation
        ? `${operation.name}${operation.semantic?.name ? ` · ${operation.semantic.name}` : ""}${position}`
        : node.kind;
      tooltip.innerHTML = `<strong>${escapeHTML(heading)}</strong><span>${escapeHTML(detail)}</span>`;
      const panel = canvas.parentElement.getBoundingClientRect();
      tooltip.style.left = `${Math.min(event.clientX - panel.left + 12, panel.width - 275)}px`;
      tooltip.style.top = `${Math.min(event.clientY - panel.top + 12, panel.height - 75)}px`;
      tooltip.hidden = false;
    }

    function populateBlockFilter() {
      const blocks = [...new Set(currentFunction.operations.map(semanticBlock).filter(value => value !== null))].sort((a, b) => a - b);
      blockSelect.innerHTML = `<option value="all">All blocks</option>` + blocks.map(block => {
        const operation = currentFunction.operations.find(candidate => semanticBlock(candidate) === block);
        const kind = operation?.semantic?.block_kind ? ` · ${operation.semantic.block_kind}` : "";
        return `<option value="${block}">Block ${block}${escapeHTML(kind)}</option>`;
      }).join("");
      blockFilter = "all";
    }

    function updateProfile() {
      const depth = treeDepth(currentFunction);
      const root = currentFunction.treeNodeMap.get(currentFunction.tree.root_id);
      const isFlatTemporal = root?.kind === "temporal_cut" && depth === 1 &&
        root.child_ids.length === currentFunction.operations.length;
      document.querySelector("#tree-profile").textContent = isFlatTemporal
        ? "Deterministic temporal baseline"
        : "Hierarchical resource-allocation tree";
      document.querySelector("#tree-profile-detail").textContent = isFlatTemporal
        ? `One temporal root with ${root.child_ids.length} direct operation leaves.`
        : `${currentFunction.tree.nodes.length} nodes across ${depth + 1} levels.`;
      if (currentFunction.realization.stage_count) {
        document.querySelector("#tree-profile-detail").textContent +=
          ` ${currentFunction.realization.stage_count} expanded Golem stages are overlaid.`;
      }
      document.querySelector("#metric-nodes").textContent = currentFunction.tree.nodes.length.toLocaleString();
      document.querySelector("#metric-depth").textContent = depth;
      document.querySelector("#metric-operations").textContent =
        currentFunction.realization.stage_count
          ? `${currentFunction.operations.length.toLocaleString()} / ${currentFunction.realization.stage_count.toLocaleString()}`
          : currentFunction.operations.length.toLocaleString();
      document.querySelector("#metric-tensors").textContent = currentFunction.tensors.length.toLocaleString();
      document.querySelector("#metric-edges").textContent = currentFunction.edges.length.toLocaleString();
      const placementMetric = document.querySelector("#metric-placement");
      placementMetric.hidden = !currentFunction.physicalPlacement;
      if (currentFunction.physicalPlacement) {
        const schedule = currentFunction.physicalPlacement.schedule;
        document.querySelector("#metric-placement-label").textContent =
          `${schedule[0].toUpperCase()}${schedule.slice(1)} graph score`;
        document.querySelector("#metric-placement-score").textContent =
          currentFunction.physicalPlacement.total_transfer_cost.toLocaleString();
      }
      document.querySelector("#metric-fingerprint").textContent = currentFunction.tree.fingerprint.slice(0, 16);
      renderPlanSummary();
    }

    function renderPlanSummary() {
      const section = document.querySelector("#plan-summary");
      const plan = currentFunction.plan;
      section.hidden = !plan;
      if (!plan) return;

      document.querySelector("#plan-summary-title").textContent =
        `${plan.planner} · ${plan.objective}`;
      document.querySelector("#plan-summary-detail").textContent =
        `${formatNanoseconds(plan.estimated_latency_ns)} · ${plan.pipeline_stages} pipeline stages · ${plan.required_resource_units} resource units`;

      const selected = plan.candidates.find(candidate => candidate.selected);
      document.querySelector("#plan-candidates").innerHTML = plan.candidates.map(candidate => {
        let decision;
        if (candidate.selected) {
          decision = `<span class="decision-selected">Selected</span>`;
        } else if (!candidate.feasible) {
          decision = `<span class="decision-rejected">${escapeHTML(candidate.infeasibility_reason || "Infeasible")}</span>`;
        } else {
          const delta = candidate.estimated_latency_ns - selected.estimated_latency_ns;
          const reason = delta > 0
            ? `${formatNanoseconds(delta)} slower`
            : "Lost objective tie-break";
          decision = `<span class="decision-rejected">${reason}</span>`;
        }
        return `<tr class="${candidate.selected ? "selected" : ""}">` +
          `<td>${escapeHTML(candidate.name)}</td>` +
          `<td>${formatNanoseconds(candidate.estimated_latency_ns)}</td>` +
          `<td>${formatBytes(candidate.crossing_bytes)}</td>` +
          `<td>${formatNanoseconds(candidate.estimated_communication_ns)}</td>` +
          `<td>${candidate.required_resource_units}</td>` +
          `<td>${decision}</td></tr>`;
      }).join("");
    }

    function updateViewExplanation() {
      const title = document.querySelector("#view-title");
      const detail = document.querySelector("#view-detail");
      if (currentView === "tree") {
        title.textContent = "RA hierarchy";
        detail.textContent = currentFunction.realization.stage_count
          ? "Temporal and spatial cuts own logical operation leaves. Expanded Golem stages appear directly beneath each sculptor.mvm leaf."
          : "The complete temporal and spatial hierarchy is ordered from root to leaves. Temporal phases advance from left to right in planned execution order; parallel children remain ordered within each spatial phase.";
        return;
      }
      if (currentView === "dag") {
        title.textContent = "Compute DAG";
        detail.textContent = currentFunction.realization.stage_count
          ? "Dependency levels advance from left to right. Branches expose fan-out, joins expose synchronization, and each sculptor.mvm node is refined into its expanded stages."
          : "Dependency levels advance from left to right. Branches expose tensor fan-out and joins expose synchronization between structured operations.";
        return;
      }
      if (currentView === "physical") {
        const placement = currentFunction.physicalPlacement;
        title.textContent = "Physical mesh";
        detail.textContent = placement
          ? `${placement.schedule} maps ${placement.assignments.length} logical tiles onto a ${placement.mesh_rows} × ${placement.mesh_cols} mesh. Purple links show inter-tile communication; width scales with bytes.`
          : "Run --sculptor-place-logical-tiles to produce a physical placement.";
        return;
      }
      title.textContent = layout.explicitRealization
        ? "Realized S-T graph"
        : "Structural S-T graph";
      if (layout.explicitRealization) {
        detail.textContent = "The selected RA tree is projected onto explicit logical tiles and resource lanes. Distinct timing boundaries form ordered columns, while equal boundaries expose parallel work; exact nanoseconds remain available per operation.";
        return;
      }
      detail.textContent = layout.cyclicSpatialDependency
        ? "The inferred child dependency graph contains a cycle. The view uses deterministic fallback stages and is not a legal evaluated mapping."
        : currentFunction.plan
          ? currentFunction.realization.stage_count
            ? "Temporal cuts advance along time and spatial cuts create lanes. Every sculptor.mvm footprint is refined into its matrix-setup, vector-tile, array-MVM, and recombination stages."
            : "Temporal cuts advance along normalized time and spatial cuts create resource lanes. The inspector shows reference-evaluator costs; physical core IDs remain unassigned."
          : "Temporal cuts advance along normalized time. Spatial cuts create resource lanes. Core IDs and nanosecond timing require mapping evaluation.";
    }

    function updateSetupFillButton() {
      setupFillButton.textContent = showSetupAndFill
        ? "Setup + Fill: shown"
        : "Setup + Fill: hidden";
      setupFillButton.setAttribute("aria-pressed", String(showSetupAndFill));
      setupFillButton.classList.toggle("toggle-active", showSetupAndFill);
    }

    function loadFunction(index) {
      currentFunction = mapFunctionData(report.functions[index]);
      if (!requestedView && currentFunction.physicalPlacement)
        currentView = "physical";
      if (currentView === "physical" && !currentFunction.physicalPlacement)
        currentView = "tree";
      updateViewURL();
      document.querySelectorAll("[data-view]").forEach(candidate => {
        candidate.disabled = candidate.dataset.view === "physical" &&
          !currentFunction.physicalPlacement;
        candidate.classList.toggle("active", candidate.dataset.view === currentView);
      });
      populateBlockFilter();
      updateProfile();
      setupAnnealingProgress();
      updateSetupFillButton();
      selectedNode = null;
      searchInput.value = "";
      rebuildLayout(true);
      renderInspector(null);
      draw();
    }

    function selectSearchResult() {
      const query = searchInput.value.trim().toLowerCase();
      if (!query) return;
      const numeric = Number(query.replace(/^op\s*/, ""));
      const operation = currentFunction.operations.find(candidate =>
        (Number.isInteger(numeric) && candidate.id === numeric) ||
        candidate.name.toLowerCase().includes(query) ||
        Object.values(candidate.semantic || {}).some(value => String(value).toLowerCase().includes(query))
      );
      if (!operation) return;
      const node = currentView === "physical"
        ? layout.nodes.find(candidate =>
            candidate.logicalTile?.digital_assignments.some(
              assignment => assignment.operation_id === operation.id
            ) || candidate.logicalTile?.analog_lanes.some(lane =>
              lane.assignments.some(assignment =>
                assignment.operation_id === operation.id
              )
            )
          )
        : layout.nodes.find(candidate => candidate.operation?.id === operation.id);
      if (node) centerNode(node);
    }

    document.querySelector("#report-title").textContent = report.title;
    document.querySelector("#report-source").textContent = report.expanded_source
      ? `${report.source} · expanded: ${report.expanded_source}`
      : report.source;
    document.querySelector("#schema-version").textContent = `Report schema ${report.schema_version}`;
    functionSelect.innerHTML = report.functions.map((func, index) =>
      `<option value="${index}">@${escapeHTML(func.symbol)}</option>`
    ).join("");
    document.querySelector("#function-control").hidden = report.functions.length === 1;

    functionSelect.addEventListener("change", () => loadFunction(Number(functionSelect.value)));
    blockSelect.addEventListener("change", () => {
      blockFilter = blockSelect.value;
      draw();
    });
    setupFillButton.addEventListener("click", () => {
      showSetupAndFill = !showSetupAndFill;
      updateSetupFillButton();
      selectedNode = null;
      rebuildLayout(true);
      renderInspector(null);
      draw();
    });
    document.querySelectorAll("[data-view]").forEach(button => {
      button.addEventListener("click", () => {
        currentView = button.dataset.view;
        updateViewURL();
        document.querySelectorAll("[data-view]").forEach(candidate =>
          candidate.classList.toggle("active", candidate === button)
        );
        selectedNode = null;
        rebuildLayout(true);
        renderInspector(null);
        draw();
      });
    });
    document.querySelector("#fit-view").addEventListener("click", () => fitView(true));
    document.querySelector("#zoom-in").addEventListener("click", () => {
      const rect = canvas.getBoundingClientRect();
      zoomAt(
        currentView === "tree" ? 1.2 : 1.35,
        rect.left + rect.width / 2,
        rect.top + rect.height / 2
      );
    });
    document.querySelector("#zoom-out").addEventListener("click", () => {
      const rect = canvas.getBoundingClientRect();
      zoomAt(
        currentView === "tree" ? 1 / 1.2 : 1 / 1.35,
        rect.left + rect.width / 2,
        rect.top + rect.height / 2
      );
    });
    searchInput.addEventListener("keydown", event => {
      if (event.key === "Enter") selectSearchResult();
    });
    annealingReplay.addEventListener("click", playAnnealingTrace);
    annealingSlider.addEventListener("input", () => {
      stopAnnealingReplay();
      renderAnnealingFrame(Number(annealingSlider.value));
    });

    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      const factor = currentView === "tree" ? 1.08 : 1.12;
      zoomAt(event.deltaY < 0 ? factor : 1 / factor, event.clientX, event.clientY);
    }, {passive: false});
    canvas.addEventListener("pointerdown", event => {
      event.preventDefault();
      activePointers.set(event.pointerId, {
        x: event.clientX,
        y: event.clientY,
      });
      canvas.setPointerCapture(event.pointerId);
      if (activePointers.size === 1) {
        beginDrag(event.pointerId, activePointers.get(event.pointerId));
      } else if (activePointers.size === 2) {
        beginPinchGesture();
      }
    });
    canvas.addEventListener("pointermove", event => {
      if (activePointers.has(event.pointerId)) {
        event.preventDefault();
        activePointers.set(event.pointerId, {
          x: event.clientX,
          y: event.clientY,
        });
        if (updatePinchGesture()) return;
      }
      if (dragging) {
        const dx = event.clientX - dragging.startX;
        const dy = event.clientY - dragging.startY;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragging.moved = true;
        viewport.x = dragging.viewX + dx;
        viewport.y = dragging.viewY + dy;
        tooltip.hidden = true;
        draw();
        return;
      }
      hoveredNode = nodeAt(event.clientX, event.clientY);
      showTooltip(hoveredNode, event);
      draw();
    });
    canvas.addEventListener("pointerup", event => finishPointer(event, false));
    canvas.addEventListener("pointercancel", event => finishPointer(event, true));
    canvas.addEventListener("pointerleave", () => {
      if (activePointers.size === 0) {
        hoveredNode = null;
        tooltip.hidden = true;
        draw();
      }
    });

    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(canvas.parentElement);
    const annealingObserver = new ResizeObserver(() => {
      if (!annealingProgress.hidden) renderAnnealingFrame(annealingFrame);
    });
    annealingObserver.observe(annealingChart.parentElement);
    loadFunction(0);
    resizeCanvas();
  </script>
</body>
</html>
)HTML";

#endif // SCULPTOR_MLIR_TOOLS_RA_TREE_REPORT_HTML_H

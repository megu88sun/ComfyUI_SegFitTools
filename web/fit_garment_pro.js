// ==============================================================================
// File Name:    fit_garment_pro.js
// Description:  Implements the interactive Fit Garment Pro editor interface.
// Proposer:     MeguPaPa3
// Author:       OpenAI Codex
// Created:      2026-07-16
//
// Change Log:
// Date        Author        Description
// ----------  ------------  --------------------------------------------------
// 2026-07-16  OpenAI Codex  Initial release.
// 2026-07-17  OpenAI Codex  Updated fixed editor sizing, selection safety, and transform state.
// ==============================================================================
import { app } from "../../scripts/app.js";

const DEFAULT_STATE = {
  version: 1,
  selected: "bra",
  bra: { cx: 0.50, cy: 0.38, width: 0.46, scale_y: 1.0, rotation: 0.0 },
  panties: { cx: 0.50, cy: 0.70, width: 0.38, scale_y: 1.0, rotation: 0.0 },
};

const EDITOR_NODE_WIDTH = 548;
const EDITOR_CANVAS_WIDTH = 520;
const EDITOR_FRAME_BORDER = 3;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function imageUrl(info) {
  const query = new URLSearchParams({
    filename: info.filename,
    subfolder: info.subfolder || "",
    type: info.type || "temp",
  });
  return `/view?${query.toString()}`;
}

function loadImage(info) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load ${info.filename}`));
    image.src = imageUrl(info);
  });
}

function hideWidget(widget) {
  if (!widget) return;
  widget.computeSize = () => [0, -4];
  widget.type = "converted-widget";
}

app.registerExtension({
  name: "SegFitTools.FitGarmentPro",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "FitGarmentPro") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const node = this;
      const stateWidget = node.widgets?.find((w) => w.name === "transform_state");
      const actionWidget = node.widgets?.find((w) => w.name === "action");
      hideWidget(stateWidget);
      hideWidget(actionWidget);

      // Never restore a serialized "run" state when a workflow is opened.
      // Setup must load the editor first; only the editor RUN button may
      // release fitting outputs to downstream generation nodes.
      const originalConfigured = node.onConfigure;
      node.onConfigure = function () {
        const value = originalConfigured?.apply(this, arguments);
        const restoredAction = this.widgets?.find((w) => w.name === "action");
        if (restoredAction) restoredAction.value = "setup";
        return value;
      };
      if (actionWidget) actionWidget.value = "setup";

      node._segfitState = clone(DEFAULT_STATE);
      node._segfitImages = { person: null, bra: null, panties: null };
      node._segfitDrag = null;
      node._segfitPointerInside = false;
      node._segfitInitialState = clone(DEFAULT_STATE);
      node._segfitEditorMinHeight = 420;

      const root = document.createElement("div");
      root.style.cssText = `display:flex;flex-direction:column;gap:6px;width:${EDITOR_NODE_WIDTH - 12}px;min-width:${EDITOR_NODE_WIDTH - 12}px;max-width:${EDITOR_NODE_WIDTH - 12}px;height:auto;box-sizing:border-box;overflow:hidden;background:#171717;padding:5px;pointer-events:auto;`;
      const toolbar = document.createElement("div");
      toolbar.style.cssText = `display:flex;gap:5px;align-items:center;flex-wrap:wrap;width:${EDITOR_CANVAS_WIDTH}px;min-width:${EDITOR_CANVAS_WIDTH}px;max-width:${EDITOR_CANVAS_WIDTH}px;font:12px sans-serif;`;
      const frame = document.createElement("div");
      frame.style.cssText = `display:block;width:${EDITOR_CANVAS_WIDTH}px;min-width:${EDITOR_CANVAS_WIDTH}px;max-width:${EDITOR_CANVAS_WIDTH}px;margin:0 auto;box-sizing:content-box;border:${EDITOR_FRAME_BORDER}px solid #f39c12;border-radius:3px;background:#252525;overflow:hidden;`;
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      canvas.style.cssText = `display:block;width:${EDITOR_CANVAS_WIDTH}px;min-width:${EDITOR_CANVAS_WIDTH}px;max-width:${EDITOR_CANVAS_WIDTH}px;height:auto;margin:0 auto;background:#252525;cursor:default;touch-action:none;`;
      frame.appendChild(canvas);
      root.append(toolbar, frame);
      const ctx = canvas.getContext("2d");

      function button(label, handler) {
        const value = document.createElement("button");
        value.textContent = label;
        value.style.cssText = "padding:5px 9px;border:1px solid #555;border-radius:4px;background:#303030;color:#ddd;cursor:pointer;";
        value.addEventListener("mousedown", (event) => event.stopPropagation());
        value.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          handler();
        });
        toolbar.appendChild(value);
        return value;
      }

      const braButton = button("Bra", () => select("bra"));
      const pantiesButton = button("Panties", () => select("panties"));
      button("Auto Fit", () => {
        const selected = node._segfitState.selected;
        node._segfitState = clone(DEFAULT_STATE);
        node._segfitState.selected = selected;
        syncState();
        draw();
      });
      button("Reset", () => {
        node._segfitState = clone(node._segfitInitialState);
        syncState();
        draw();
      });
      const runButton = button("RUN", async () => {
        syncState();
        if (actionWidget) actionWidget.value = "run";
        runButton.disabled = true;
        runButton.textContent = "Running...";
        try {
          await app.queuePrompt(0, 1);
        } finally {
          runButton.disabled = false;
          runButton.textContent = "RUN";
        }
      });
      runButton.style.background = "#347a46";
      runButton.style.marginLeft = "auto";

      const status = document.createElement("span");
      status.textContent = "Queue once to open the editor";
      status.style.cssText = "color:#aaa;margin-left:3px;flex:1 1 180px;min-width:0;white-space:normal;";
      toolbar.appendChild(status);

      function select(name) {
        node._segfitState.selected = name;
        syncState();
        updateButtons();
        draw();
      }

      function updateButtons() {
        const selected = node._segfitState.selected;
        braButton.style.borderColor = selected === "bra" ? "#54c7ec" : "#555";
        pantiesButton.style.borderColor = selected === "panties" ? "#f39cdd" : "#555";
      }

      function syncState() {
        if (stateWidget) stateWidget.value = JSON.stringify(node._segfitState);
        node.graph?.setDirtyCanvas(true, true);
      }

      function geometry(name) {
        const image = node._segfitImages[name];
        if (!image) return null;
        const state = node._segfitState[name];
        const width = state.width * canvas.width;
        const height = width * (image.naturalHeight / image.naturalWidth) * state.scale_y;
        return {
          image,
          state,
          cx: state.cx * canvas.width,
          cy: state.cy * canvas.height,
          width,
          height,
          radians: state.rotation * Math.PI / 180,
        };
      }

      function localPoint(geo, x, y) {
        const dx = x - geo.cx;
        const dy = y - geo.cy;
        const c = Math.cos(-geo.radians);
        const s = Math.sin(-geo.radians);
        return { x: dx * c - dy * s, y: dx * s + dy * c };
      }

      function worldPoint(geo, lx, ly) {
        const c = Math.cos(geo.radians);
        const s = Math.sin(geo.radians);
        return { x: geo.cx + lx * c - ly * s, y: geo.cy + lx * s + ly * c };
      }

      function handles(geo) {
        const w = geo.width / 2;
        const h = geo.height / 2;
        return [
          worldPoint(geo, -w, -h), worldPoint(geo, w, -h),
          worldPoint(geo, w, h), worldPoint(geo, -w, h),
        ];
      }

      function drawGarment(name, selected) {
        const geo = geometry(name);
        if (!geo) return;
        ctx.save();
        ctx.translate(geo.cx, geo.cy);
        ctx.rotate(geo.radians);
        ctx.globalAlpha = selected ? 0.92 : 0.82;
        ctx.drawImage(geo.image, -geo.width / 2, -geo.height / 2, geo.width, geo.height);
        if (selected) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = name === "bra" ? "#54c7ec" : "#f39cdd";
          ctx.lineWidth = Math.max(2, canvas.width / 300);
          ctx.setLineDash([8, 5]);
          ctx.strokeRect(-geo.width / 2, -geo.height / 2, geo.width, geo.height);
        }
        ctx.restore();
        if (!selected) return;
        ctx.setLineDash([]);
        ctx.fillStyle = name === "bra" ? "#54c7ec" : "#f39cdd";
        for (const point of handles(geo)) {
          ctx.fillRect(point.x - 6, point.y - 6, 12, 12);
        }
        const rotationHandle = worldPoint(geo, 0, -geo.height / 2 - 28);
        const top = worldPoint(geo, 0, -geo.height / 2);
        ctx.strokeStyle = ctx.fillStyle;
        ctx.beginPath();
        ctx.moveTo(top.x, top.y);
        ctx.lineTo(rotationHandle.x, rotationHandle.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(rotationHandle.x, rotationHandle.y, 7, 0, Math.PI * 2);
        ctx.fill();
      }

      function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const person = node._segfitImages.person;
        if (!person) {
          ctx.fillStyle = "#aaa";
          ctx.font = "16px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("Queue once to load images", canvas.width / 2, canvas.height / 2);
          return;
        }
        ctx.drawImage(person, 0, 0, canvas.width, canvas.height);
        const selected = node._segfitState.selected;
        if (selected !== "bra" && selected !== "panties") {
          drawGarment("bra", false);
          drawGarment("panties", false);
          updateButtons();
          return;
        }
        drawGarment(selected === "bra" ? "panties" : "bra", false);
        drawGarment(selected, true);
        updateButtons();

        // =================================================================
        // --- 動的シードポイント（赤い点）の描画 ---
        // =================================================================
        const points = node._segfitSeedPoints || [];

        if (points.length > 0) {
          ctx.fillStyle = "#ff0000";
          ctx.strokeStyle = "#aaa";
          ctx.lineWidth = 1.5;

          for (const pt of points) {
            // [x, y] 形式であることを確認
            if (!Array.isArray(pt) || pt.length < 2) continue;
            const [nx, ny] = pt;

            const px = nx * canvas.width;
            const py = ny * canvas.height;

            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }
        // =================================================================
      }

      function pointerPosition(event) {
        const rect = canvas.getBoundingClientRect();
        return {
          x: (event.clientX - rect.left) * canvas.width / rect.width,
          y: (event.clientY - rect.top) * canvas.height / rect.height,
        };
      }

      function hitTest(name, point) {
        const geo = geometry(name);
        if (!geo) return null;
        const rotationHandle = worldPoint(geo, 0, -geo.height / 2 - 28);
        if (Math.hypot(point.x - rotationHandle.x, point.y - rotationHandle.y) <= 14) return "rotate";
        for (const handle of handles(geo)) {
          if (Math.hypot(point.x - handle.x, point.y - handle.y) <= 14) return "scale";
        }
        const local = localPoint(geo, point.x, point.y);
        if (Math.abs(local.x) <= geo.width / 2 && Math.abs(local.y) <= geo.height / 2) return "move";
        return null;
      }

      canvas.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const point = pointerPosition(event);
        let selected = node._segfitState.selected;
        let mode = selected ? hitTest(selected, point) : null;
        if (!mode && selected) {
          const other = selected === "bra" ? "panties" : "bra";
          if (hitTest(other, point) === "move") {
            selected = other;
            select(other);
            mode = "move";
          }
        }
        if (!mode && !selected) {
          for (const name of ["bra", "panties"]) {
            const hit = hitTest(name, point);
            if (hit) {
              selected = name;
              select(name);
              mode = hit;
              break;
            }
          }
        }
        if (!mode) return;
        const geo = geometry(selected);
        node._segfitDrag = {
          pointerId: event.pointerId,
          name: selected,
          mode,
          start: point,
          state: clone(node._segfitState[selected]),
          startDistance: Math.max(1, Math.hypot(point.x - geo.cx, point.y - geo.cy)),
          startAngle: Math.atan2(point.y - geo.cy, point.x - geo.cx),
        };
        canvas.setPointerCapture(event.pointerId);
      });

      canvas.addEventListener("pointermove", (event) => {
        const drag = node._segfitDrag;
        if (!drag || drag.pointerId !== event.pointerId) return;
        event.preventDefault();
        const point = pointerPosition(event);
        const state = node._segfitState[drag.name];
        if (drag.mode === "move") {
          state.cx = drag.state.cx + (point.x - drag.start.x) / canvas.width;
          state.cy = drag.state.cy + (point.y - drag.start.y) / canvas.height;
        } else if (drag.mode === "scale") {
          const geo = geometry(drag.name);
          const distance = Math.hypot(point.x - geo.cx, point.y - geo.cy);
          state.width = Math.max(0.03, Math.min(2.0, drag.state.width * distance / drag.startDistance));
        } else if (drag.mode === "rotate") {
          const geo = geometry(drag.name);
          const angle = Math.atan2(point.y - geo.cy, point.x - geo.cx);
          state.rotation = drag.state.rotation + (angle - drag.startAngle) * 180 / Math.PI;
        }
        syncState();
        draw();
      });

      function endDrag(event) {
        if (!node._segfitDrag || node._segfitDrag.pointerId !== event.pointerId) return;
        node._segfitDrag = null;
        canvas.releasePointerCapture?.(event.pointerId);
        syncState();
        if (!node._segfitPointerInside) select(null);
      }
      canvas.addEventListener("pointerup", endDrag);
      canvas.addEventListener("pointercancel", endDrag);
      canvas.addEventListener("pointerenter", () => {
        node._segfitPointerInside = true;
      });
      canvas.addEventListener("pointerleave", () => {
        node._segfitPointerInside = false;
        if (!node._segfitDrag) select(null);
      });
      canvas.addEventListener("wheel", (event) => {
        const selected = node._segfitState.selected;
        if (selected !== "bra" && selected !== "panties") return;
        event.preventDefault();
        event.stopPropagation();
        const state = node._segfitState[selected];
        state.width = Math.max(0.03, Math.min(2.0, state.width * (event.deltaY < 0 ? 1.03 : 0.97)));
        syncState();
        draw();
      }, { passive: false });

      node._segfitLoad = async function (payload) {
        try {
          const [person, bra, panties] = await Promise.all([
            loadImage(payload.person), loadImage(payload.bra), loadImage(payload.panties),
          ]);
          node._segfitImages = { person, bra, panties };
          node._segfitState = clone(payload.state || DEFAULT_STATE);
          node._segfitInitialState = clone(node._segfitState);
          if (stateWidget) stateWidget.value = JSON.stringify(node._segfitState);
          if (actionWidget?.value === "run") actionWidget.value = "setup";
          canvas.width = person.naturalWidth;
          canvas.height = person.naturalHeight;
          status.textContent = "Adjust garments, then RUN";
          draw();
          const aspect = person.naturalHeight / person.naturalWidth;
          const canvasHeight = EDITOR_CANVAS_WIDTH * aspect;
          node._segfitEditorMinHeight = 52 + canvasHeight + EDITOR_FRAME_BORDER * 2;
          node.setSize([EDITOR_NODE_WIDTH, 82 + node._segfitEditorMinHeight]);
          node.graph?.setDirtyCanvas(true, true);
        } catch (error) {
          status.textContent = error.message;
          console.error("Fit Garment Pro:", error);
        }
      };

      const originalExecuted = node.onExecuted;
      node.onExecuted = function (message) {
        const value = originalExecuted?.apply(this, arguments);
        const payload = message?.segfit_pro?.[0];
        if (payload) node._segfitLoad(payload);
        return value;
      };

      node.addDOMWidget("fit_garment_editor", "FitGarmentEditor", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => node._segfitEditorMinHeight,
      });
      const originalResized = node.onResize;
      node.onResize = function (size) {
        size[0] = EDITOR_NODE_WIDTH;
        size[1] = Math.max(size[1], 82 + node._segfitEditorMinHeight);
        return originalResized?.apply(this, arguments);
      };
      node.setSize([EDITOR_NODE_WIDTH, 650]);
      updateButtons();
      draw();
      return result;
    };
  },
});

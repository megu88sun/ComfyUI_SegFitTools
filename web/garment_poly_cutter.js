// ==============================================================================
// File Name  : garment_poly_cutter.js
// Description: Interactive frontend editor for polygon garment cutout masking.
// Proposer   : MeguPaPa3
// Author     : Google Gemini
// Created    : 2026-09-18
//
// Change Log:
// Date       Author         Description
// ---------- -------------- ----------------------------------------------------
// 2026-09-18 Google Gemini  Initial unified implementation matching FitGarmentPro
//                           conventions, layout constraints, and event handling.
// 2026-10-08 OpenAI Codex  Added locked raster overlays for established masks.
// ==============================================================================

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DEFAULT_STATE = {
  version: 1,
  selected: "bra",
  bra_points: [],
  panties_points: [],
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
  const endpoint = `/view?${query.toString()}`;
  return api?.apiURL ? api.apiURL(endpoint) : endpoint;
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
  name: "SegFitTools.GarmentPolyCutter",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "GarmentPolyCutter") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const node = this;
      const stateWidget = node.widgets?.find((w) => w.name === "poly_state");
      const actionWidget = node.widgets?.find((w) => w.name === "action");
      hideWidget(stateWidget);
      hideWidget(actionWidget);

      const originalConfigured = node.onConfigure;
      node.onConfigure = function () {
        const value = originalConfigured?.apply(this, arguments);
        const restoredAction = this.widgets?.find((w) => w.name === "action");
        if (restoredAction) restoredAction.value = "setup";
        return value;
      };
      if (actionWidget) actionWidget.value = "setup";

      node._segfitState = clone(DEFAULT_STATE);
      node._segfitPerson = null;
      node._segfitLocks = { bra: false, panties: false };
      node._segfitLockedImages = { bra: null, panties: null };
      node._dragPointIndex = -1;
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
      canvas.style.cssText = `display:block;width:${EDITOR_CANVAS_WIDTH}px;min-width:${EDITOR_CANVAS_WIDTH}px;max-width:${EDITOR_CANVAS_WIDTH}px;height:auto;margin:0 auto;background:#252525;cursor:crosshair;touch-action:none;`;
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
      const clearButton = button("Clear", () => {
        if (node._segfitLocks[node._segfitState.selected]) return;
        const current = node._segfitState.selected === "bra" ? "bra_points" : "panties_points";
        node._segfitState[current] = [];
        syncState();
        draw();
      });
      const resetButton = button("Reset All", () => {
        if (!node._segfitLocks.bra) node._segfitState.bra_points = [];
        if (!node._segfitLocks.panties) node._segfitState.panties_points = [];
        syncState();
        draw();
      });

      const runButton = button("RUN", async () => {
        syncState();
        if (actionWidget) actionWidget.value = "run";
        runButton.disabled = true;
        runButton.textContent = "Cutting...";
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
      status.textContent = "Queue once to load person image";
      status.style.cssText = "color:#aaa;margin-left:3px;flex:1 1 180px;min-width:0;white-space:normal;";
      toolbar.appendChild(status);

      function select(name) {
        if (node._segfitLocks[name]) return;
        node._segfitState.selected = name;
        syncState();
        updateButtons();
        draw();
      }

      function updateButtons() {
        const selected = node._segfitState.selected;
        braButton.disabled = node._segfitLocks.bra;
        pantiesButton.disabled = node._segfitLocks.panties;
        clearButton.disabled = node._segfitLocks[selected];
        resetButton.disabled = node._segfitLocks.bra && node._segfitLocks.panties;
        runButton.disabled = node._segfitLocks.bra && node._segfitLocks.panties;
        braButton.style.opacity = braButton.disabled ? "0.45" : "1";
        pantiesButton.style.opacity = pantiesButton.disabled ? "0.45" : "1";
        braButton.style.borderColor = selected === "bra" ? "#54c7ec" : "#555";
        pantiesButton.style.borderColor = selected === "panties" ? "#f39cdd" : "#555";
      }

      function syncState() {
        if (stateWidget) stateWidget.value = JSON.stringify(node._segfitState);
        node.graph?.setDirtyCanvas(true, true);
      }

      function pointerPosition(event) {
        const rect = canvas.getBoundingClientRect();
        return {
          x: (event.clientX - rect.left) * canvas.width / rect.width,
          y: (event.clientY - rect.top) * canvas.height / rect.height,
        };
      }

      function drawPolygon(points, color, isSelected) {
        if (!points || points.length === 0) return;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color.replace(")", ", 0.25)").replace("rgb", "rgba");
        ctx.lineWidth = Math.max(2, canvas.width / 300);

        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const px = points[i][0] * canvas.width;
          const py = points[i][1] * canvas.height;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        if (points.length >= 3) {
          ctx.closePath();
          ctx.fill();
        }
        ctx.stroke();

        if (isSelected) {
          ctx.fillStyle = color;
          for (let i = 0; i < points.length; i++) {
            const px = points[i][0] * canvas.width;
            const py = points[i][1] * canvas.height;
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!node._segfitPerson) {
          ctx.fillStyle = "#aaa";
          ctx.font = "16px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("Queue once to load person image", canvas.width / 2, canvas.height / 2);
          return;
        }
        ctx.drawImage(node._segfitPerson, 0, 0, canvas.width, canvas.height);

        if (node._segfitLockedImages.bra) {
          ctx.drawImage(node._segfitLockedImages.bra, 0, 0, canvas.width, canvas.height);
        }
        if (node._segfitLockedImages.panties) {
          ctx.drawImage(node._segfitLockedImages.panties, 0, 0, canvas.width, canvas.height);
        }

        const isBra = node._segfitState.selected === "bra";
        if (!node._segfitLocks.bra) {
          drawPolygon(node._segfitState.bra_points, "rgb(84, 199, 236)", isBra);
        }
        if (!node._segfitLocks.panties) {
          drawPolygon(node._segfitState.panties_points, "rgb(243, 156, 221)", !isBra);
        }
        updateButtons();
      }

      canvas.addEventListener("pointerdown", (event) => {
        if (!node._segfitPerson) return;
        if (node._segfitLocks[node._segfitState.selected]) return;
        event.preventDefault();
        event.stopPropagation();
        const pos = pointerPosition(event);
        const currentKey = node._segfitState.selected === "bra" ? "bra_points" : "panties_points";
        const points = node._segfitState[currentKey];

        let hitIndex = -1;
        for (let i = 0; i < points.length; i++) {
          const px = points[i][0] * canvas.width;
          const py = points[i][1] * canvas.height;
          if (Math.hypot(pos.x - px, pos.y - py) <= 12) {
            hitIndex = i;
            break;
          }
        }

        if (hitIndex !== -1) {
          node._dragPointIndex = hitIndex;
        } else {
          points.push([pos.x / canvas.width, pos.y / canvas.height]);
          node._dragPointIndex = points.length - 1;
        }

        canvas.setPointerCapture(event.pointerId);
        syncState();
        draw();
      });

      canvas.addEventListener("pointermove", (event) => {
        if (node._dragPointIndex === -1) return;
        event.preventDefault();
        const pos = pointerPosition(event);
        const currentKey = node._segfitState.selected === "bra" ? "bra_points" : "panties_points";
        const points = node._segfitState[currentKey];
        if (points[node._dragPointIndex]) {
          points[node._dragPointIndex] = [
            Math.max(0, Math.min(1, pos.x / canvas.width)),
            Math.max(0, Math.min(1, pos.y / canvas.height)),
          ];
          syncState();
          draw();
        }
      });

      function endDrag(event) {
        node._dragPointIndex = -1;
        canvas.releasePointerCapture?.(event.pointerId);
        syncState();
      }
      canvas.addEventListener("pointerup", endDrag);
      canvas.addEventListener("pointercancel", endDrag);

      canvas.addEventListener("pointerleave", () => {
        node._dragPointIndex = -1;
      });

      node._segfitLoad = async function (payload) {
        try {
          const [person, lockedBra, lockedPanties] = await Promise.all([
            loadImage(payload.person),
            payload.bra_locked ? loadImage(payload.bra_locked) : Promise.resolve(null),
            payload.panties_locked ? loadImage(payload.panties_locked) : Promise.resolve(null),
          ]);
          node._segfitPerson = person;
          node._segfitLocks = { bra: false, panties: false, ...(payload.locks || {}) };
          node._segfitLockedImages = { bra: lockedBra, panties: lockedPanties };
          node._segfitState = clone(payload.state || DEFAULT_STATE);
          if (node._segfitLocks[node._segfitState.selected]) {
            if (!node._segfitLocks.bra) node._segfitState.selected = "bra";
            else if (!node._segfitLocks.panties) node._segfitState.selected = "panties";
          }
          if (stateWidget) stateWidget.value = JSON.stringify(node._segfitState);
          if (actionWidget?.value === "run") actionWidget.value = "setup";
          canvas.width = person.naturalWidth;
          canvas.height = person.naturalHeight;
          if (node._segfitLocks.bra && node._segfitLocks.panties) {
            status.textContent = "Both masks established; Poly Cutter bypassed";
          } else if (node._segfitLocks.bra) {
            status.textContent = "Bra locked; edit panties, then RUN";
          } else if (node._segfitLocks.panties) {
            status.textContent = "Panties locked; edit bra, then RUN";
          } else {
            status.textContent = "Click around underwear on skin, then RUN";
          }
          draw();
          const aspect = person.naturalHeight / person.naturalWidth;
          const canvasHeight = EDITOR_CANVAS_WIDTH * aspect;
          node._segfitEditorMinHeight = 52 + canvasHeight + EDITOR_FRAME_BORDER * 2;
          node.setSize([EDITOR_NODE_WIDTH, 82 + node._segfitEditorMinHeight]);
          node.graph?.setDirtyCanvas(true, true);
        } catch (error) {
          status.textContent = error.message;
          console.error("[GarmentPolyCutter]", error);
        }
      };

      const originalExecuted = node.onExecuted;
      node.onExecuted = function (message) {
        const value = originalExecuted?.apply(this, arguments);
        const payload = message?.segfit_poly?.[0];
        if (payload) node._segfitLoad(payload);
        return value;
      };

      node.addDOMWidget("poly_cutter_editor", "PolyCutterEditor", root, {
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

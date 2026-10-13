// ==============================================================================
// File Name:    image_auto_switching.js
// Description:  Synchronizes dynamic IMAGE inputs for Image Auto Switching.
// Proposer:     MeguPaPa3
// Author:       OpenAI Codex
// Created:      2026-07-19
//
// Change Log:
// Date        Author        Description
// ----------  ------------  --------------------------------------------------
// 2026-07-19  OpenAI Codex  Initial release.
// ==============================================================================
import { app } from "../../scripts/app.js";

const MAX_INPUTS = 8;

function syncInputs(node) {
  const widget = node.widgets?.find((item) => item.name === "input_count");
  const desired = Math.max(2, Math.min(MAX_INPUTS, Number(widget?.value ?? 2)));

  for (let index = node.inputs.length - 1; index >= 0; index -= 1) {
    const match = /^image_(\d+)$/.exec(node.inputs[index]?.name ?? "");
    if (match && Number(match[1]) > desired) node.removeInput(index);
  }
  for (let index = 1; index <= desired; index += 1) {
    if (!node.inputs?.some((input) => input.name === `image_${index}`)) {
      node.addInput(`image_${index}`, "IMAGE");
    }
  }
  node.setSize(node.computeSize());
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "SegFitTools.ImageAutoSwitching",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "ImageAutoSwitching") return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const widget = this.widgets?.find((item) => item.name === "input_count");
      if (widget) {
        const originalCallback = widget.callback;
        widget.callback = (value, ...args) => {
          const callbackResult = originalCallback?.call(widget, value, ...args);
          syncInputs(this);
          return callbackResult;
        };
      }
      queueMicrotask(() => syncInputs(this));
      return result;
    };

    const originalConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigured?.apply(this, arguments);
      queueMicrotask(() => syncInputs(this));
      return result;
    };
  },
});

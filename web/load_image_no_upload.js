// ==============================================================================
// File Name:    load_image_no_upload.js
// Description:  Hides upload controls while retaining the standard image preview.
// Proposer:     MeguPaPa3
// Author:       OpenAI Codex
// Created:      2026-07-21
//
// Change Log:
// Date        Author        Description
// ----------  ------------  --------------------------------------------------
// 2026-07-21  OpenAI Codex  Initial release.
// 2026-07-25  OpenAI Codex  Added folder-scoped image option refresh.
// 2026-07-26  OpenAI Codex  Restored encoded paths from saved workflows.
// 2026-07-27  OpenAI Codex  Added support for the low-profile loader variant.
// ==============================================================================

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function hideUploadWidget(widget) {
  widget.computeSize = () => [0, -4];
  widget.hidden = true;
}

function decodeWidgetValue(value) {
  let decoded = String(value || "");
  const textarea = document.createElement("textarea");
  for (let index = 0; index < 4; index += 1) {
    textarea.innerHTML = decoded;
    const nextValue = textarea.value;
    if (nextValue === decoded) break;
    decoded = nextValue;
  }
  return decoded;
}

async function refreshImageOptions(node, folderWidget, imageWidget) {
  const folderName = String(folderWidget?.value || "").trim();
  const response = await api.fetchApi(
    `/segfittools/input-images?folder_name=${encodeURIComponent(folderName)}`
  );
  if (!response.ok) return;

  const values = await response.json();
  // const previous = imageWidget.value;
  const previous = decodeWidgetValue(imageWidget.value);
  imageWidget.options.values = values.length ? values : [""];
  imageWidget.value = values.includes(previous) ? previous : (values[0] || "");
  imageWidget.callback?.(imageWidget.value);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "SegFitTools.LoadImageNoUpload",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    // if (nodeData?.name !== "LoadImageNoUpload") return;
    if (!["LoadImageNoUpload", "LoadImageNoUploadLM"].includes(nodeData?.name)) {
      return;
    }

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const node = this;
      const folderWidget = node.widgets?.find((w) => w.name === "folder_name");
      const imageWidget = node.widgets?.find((w) => w.name === "image");

      for (const widget of this.widgets || []) {
        const name = String(widget.name || "").toLowerCase();
        if (name.includes("upload")) hideUploadWidget(widget);
      }

      if (folderWidget && imageWidget) {
        let refreshTimer = null;
        const originalFolderCallback = folderWidget.callback;
        folderWidget.callback = function () {
          originalFolderCallback?.apply(this, arguments);
          clearTimeout(refreshTimer);
          refreshTimer = setTimeout(
            () => refreshImageOptions(node, folderWidget, imageWidget),
            250
          );
        };

        const originalConfigured = node.onConfigure;
        node.onConfigure = function () {
          const value = originalConfigured?.apply(this, arguments);
          setTimeout(
            () => refreshImageOptions(node, folderWidget, imageWidget),
            0
          );
          return value;
        };
      }

      return result;
    };
  },
});

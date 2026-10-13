// =============================================================================
// File Name:    color_picker.js
// Description:  Attaches eyedropper / native color picker to color_in_HEX widget.
// Proposer:     MeguPaPa3
// Author:       Google Gemini
// Created:      2026-09-24
//
// Change Log:
// Date        Author         Description
// ----------  -------------  --------------------------------------------------
// 2026-09-24  Google Gemini  Initial release adhering to AGENTS.md guidelines.
// =============================================================================

import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ColorPickerExtension",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // Target both internal class name and display name variants
    const targetNodes = [
      "DensePosePartExtractor",
      "Denspose Part Extractor(TEST)",
    ];
    if (!targetNodes.includes(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      // Add pipette action button directly below widgets
      this.addWidget("button", "Pick Color (Pipette)", "pick_color", () => {
        const hexWidget = this.widgets?.find((w) => w.name === "color_in_HEX");

        // Primary: Chromium EyeDropper API (Samples any pixel on screen)
        if (window.EyeDropper) {
          const eyeDropper = new EyeDropper();
          eyeDropper
            .open()
            .then((result) => {
              if (hexWidget) hexWidget.value = result.sRGBHex;
              this.setDirtyCanvas(true, true);
            })
            .catch(() => {});
          return;
        }

        // Fallback: Invisible native HTML color input
        const picker = document.createElement("input");
        picker.type = "color";
        picker.value = hexWidget?.value?.startsWith("#") ? hexWidget.value : "#481f70";
        picker.style.position = "fixed";
        picker.style.left = "-9999px";
        picker.style.top = "-9999px";
        picker.style.opacity = "0";
        document.body.appendChild(picker);

        picker.addEventListener("input", () => {
          if (hexWidget) hexWidget.value = picker.value;
          this.setDirtyCanvas(true, true);
        });

        picker.addEventListener("change", () => {
          if (hexWidget) hexWidget.value = picker.value;
          this.setDirtyCanvas(true, true);
          picker.remove();
        });

        picker.addEventListener("blur", () => picker.remove());

        if (typeof picker.showPicker === "function") {
          picker.showPicker();
        } else {
          picker.click();
        }
      });

      return r;
    };
  },
});
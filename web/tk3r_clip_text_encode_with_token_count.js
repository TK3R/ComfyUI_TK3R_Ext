import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
	name: "TK3R.TextOutput",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "TK3RCLIPTextEncodeWithTokenCount" || nodeData.name === "TK3R CLIP Text Encode With Token Count") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (this.widgets) {
					const pos = this.widgets.findIndex((w) => w.name === "output_text");
					if (pos !== -1) {
						for (let i = pos; i < this.widgets.length; i++) {
							this.widgets[i].onRemove?.();
						}
						this.widgets.length = pos;
					}
				}

				const text = message.text;
				if (!text) { return; }

				// Create widget as multiline initially to get a textarea
				const w = ComfyWidgets["STRING"](this, "output_text", ["STRING", { multiline: true }], app).widget;
				w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.6;
                w.inputEl.rows = 2;
                w.inputEl.style.height = "auto";
				w.value = text[0];

                // Trick ComfyUI/LiteGraph into treating this as a fixed-size widget
                // so it doesn't try to expand it to fill remaining vertical space
                w.options = w.options || {};
                w.options.multiline = false;

                // Override computeSize to return fixed small height
                w.computeSize = function(width) {
                    return [width, 40]; // ~40px for 2 lines
                }
				
                // Refresh node appearance without resetting dimensions
				this.onResize?.(this.size);
			};
		}
	},
});

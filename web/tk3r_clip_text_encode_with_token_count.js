import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
	name: "TK3R.TextOutput",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "TK3RCLIPTextEncodeWithTokenCount" || nodeData.name === "TK3R CLIP Text Encode With Token Count") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				const text = message.text;
				if (!text) { return; }

                let w = this.widgets?.find((w) => w.name === "output_text");
                const isNew = !w;

                if (isNew) {
				    // Create widget as multiline initially to get a textarea
				    const res = ComfyWidgets["STRING"](this, "output_text", ["STRING", { multiline: true }], app);
                    w = res.widget;
                }

                w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.6;
				w.value = text[0];

                // Override computeSize to tell the layout engine this widget is flexible but small
                w.computeSize = function(width) {
                    return [width, 45]; // ~45px for good visibility
                }

                if (!this.tk3r_resized_patched) {
                    const oldOnResize = this.onResize;
                    this.onResize = function(size) {
                        const WIDGET_HEIGHT = 30;
                        const LAYOUT_HEIGHT = 15; // What LiteGraph thinks a single-line widget takes
                        
                        // 1. Manually protect our widget from being expanded by standard logic
                        const outW = this.widgets?.find(w => w.name === "output_text");
                        if (outW) {
                            // Lie to the node that it's NOT multiline so standard logic treats it as fixed small height
                            outW.options = outW.options || {};
                            outW.options.multiline = false;
                        }

                        // 2. Run standard resize (calculates input text field size based on 'small' output widget)
                        oldOnResize?.call(this, size);
                        
                        // 3. Fix the output widget height and extend the node to accommodate it
                        if (outW && outW.inputEl) {
                            outW.inputEl.style.height = WIDGET_HEIGHT + "px";
                            outW.inputEl.style.maxHeight = WIDGET_HEIGHT + "px";
                            
                            // We told LiteGraph it was ~20px (single line), but we rendered it as 45px.
                            // We need to extend the node visually to wrap around the larger widget.
                            this.size[1] += (WIDGET_HEIGHT - LAYOUT_HEIGHT) + 5; 
                        }
                    }
                    this.tk3r_resized_patched = true;
                }
				
                // Only trigger resize if we just added the widget or patched the method
                if (isNew) {
				    this.onResize?.(this.size);
                }
			};
		}
	},
});

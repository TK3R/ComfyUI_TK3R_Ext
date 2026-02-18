import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
	name: "TK3R.TextOutput",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "TK3RCLIPTextEncodeWithTokenCount" || nodeData.name === "TK3R CLIP Text Encode With Token Count") {
			
            function updateWidget(node, text, isRestoring) {
                let w = node.widgets?.find((w) => w.name === "output_text");
                const isNew = !w;

                if (isNew) {
				    // Create widget as multiline initially to get a textarea
				    const res = ComfyWidgets["STRING"](node, "output_text", ["STRING", { multiline: true }], app);
                    w = res.widget;
                }

                w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.6;
				w.value = text;

                const WIDGET_HEIGHT = 30;
                const LAYOUT_HEIGHT = 15; // What LiteGraph thinks a single-line widget takes
                const PADDING = 15; // Extra space to prevent cramped layout

                // Override computeSize to tell the layout engine this widget is flexible but small
                w.computeSize = function(width) {
                    return [width, WIDGET_HEIGHT + PADDING]; // WIDGET_HEIGHT + PADDING
                }

                if (!node.tk3r_resized_patched) {
                    const oldOnResize = node.onResize;
                    node.onResize = function(size) {                        
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
                        }
                    }
                    node.tk3r_resized_patched = true;
                }

                // Only trigger resize if we just added the widget AND it's not a restore
                // (Restores from JSON already have the correct size)
                if (isNew && !isRestoring) {
                     // We told LiteGraph it was ~15px (single line), but we rendered it as 30px.
                     // We need to extend the node visually to wrap around the larger widget just once.
                     node.size[1] += (WIDGET_HEIGHT - LAYOUT_HEIGHT) + PADDING; 
                     node.onResize?.(node.size);
                }
            }

            const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (w_values) {
				onConfigure?.apply(this, arguments);
                if (w_values?.widgets_values?.length) {
                    // Check if our widget value is saved at the end
                    // This assumes our widget is the last one (or one of the extra ones)
                    // Since onExecuted appends new widgets, and onConfigure loads standard widgets:
                    const standardCount = this.widgets ? this.widgets.length : 0;
                    const savedCount = w_values.widgets_values.length;
                    
                    if (savedCount > standardCount) {
                        // The saved widget value is likely at standardCount index
                        // (assuming no other dynamic widgets interfere)
                        const savedValue = w_values.widgets_values[standardCount];
                        if (typeof savedValue === 'string') {
                            updateWidget(this, savedValue, true);
                        }
                    }
                }
			};

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				const text = message.text;
				if (!text) { return; }

                updateWidget(this, text[0]);
			};
		}
	},
});

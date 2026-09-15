"""Build demo/index.html: inline the trained weights into demo.html as JSON.
The demo runs the exact NumPy-trained CNN in the browser (ported to JS)."""
import json, os
import numpy as np

HERE = os.path.dirname(__file__)
m = np.load(os.path.join(HERE, "..", "watcher-model", "model.npz"))

# Round to 5 sig-figs to keep the payload small; predictions are unaffected.
def enc(a):
    return [float(f"{v:.5g}") for v in a.ravel().tolist()]

weights = {k: enc(m[k]) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
blob = json.dumps(weights, separators=(",", ":"))

html = open(os.path.join(HERE, "demo.html")).read().replace("{{WEIGHTS}}", blob)
open(os.path.join(HERE, "index.html"), "w").write(html)
print(f"weights {len(blob)/1024:.0f} KB -> demo/index.html ({len(html)/1024:.0f} KB)")
for k in weights:
    print(f"  {k}: {len(weights[k])}")

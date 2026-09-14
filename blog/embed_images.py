"""Build blog/index.html: inline optimized figures into blog.html as data URIs."""
import base64, io, os
from PIL import Image

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "watcher-model", "data")

# name -> (source path, target width, format, quality)
IMGS = {
    "booking_loading": (f"{DATA}/real/booking_loading.png", 860, "JPEG", 86),
    "booking_loaded":  (f"{DATA}/real/booking_loaded.png",  860, "JPEG", 86),
    "preview":         (f"{DATA}/preview.png",              1120, "JPEG", 88),
    "results":         (f"{DATA}/results.png",              1000, "PNG",  None),
}


def data_uri(path, w, fmt, q):
    im = Image.open(path).convert("RGB")
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        im.save(buf, "JPEG", quality=q, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}", len(buf.getvalue())


html = open(os.path.join(HERE, "blog.html")).read()
total = 0
for name, (path, w, fmt, q) in IMGS.items():
    uri, size = data_uri(path, w, fmt, q)
    html = html.replace("{{IMG:" + name + "}}", uri)
    total += size
    print(f"  {name:16s} {size/1024:6.0f} KB  ({fmt} {w}px)")

out = os.path.join(HERE, "index.html")
open(out, "w").write(html)
print(f"total images {total/1024:.0f} KB -> wrote {out} ({len(html)/1024:.0f} KB)")

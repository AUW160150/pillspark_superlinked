"""Verify patient identifiers are redacted in the web demo image.

Positive control: the same regions in the original photo must be busy (high
std dev); otherwise this check could pass on any flat image.
"""
from pathlib import Path
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "landing/assets/demo-chart.jpg"
ORIG = ROOT / "baba's_chemo_image.jpeg"

# Regions in ORIGINAL pixel coords (inset inside the fill boxes): name, reg no.
REGIONS = [(275, 326, 663, 375), (1102, 331, 1321, 377)]


def stdev(img, box):
    return max(ImageStat.Stat(img.crop(box).convert("RGB")).stddev)


web = Image.open(WEB)
if not ORIG.exists():
    # Public clones don't ship the unredacted original: check the published image's regions are flat fills only
    # (no positive control is possible without the original).
    ok = all(max(ImageStat.Stat(web.crop(tuple(round(v * web.size[0] / 1537) + (3 if i < 2 else -3) for i, v in enumerate(b))).convert("RGB")).stddev) < 6
             for b in REGIONS) and not web.getexif()
    print("original not present: checked flat redaction fill only")
    print("REDACTION_OK" if ok else "REDACTION_FAILED")
    raise SystemExit(0 if ok else 1)
orig = Image.open(ORIG)
sx = web.size[0] / orig.size[0]
ok = True
for box in REGIONS:
    wbox = tuple(round(v * sx) + (3 if i < 2 else -3) for i, v in enumerate(box))
    w, o = stdev(web, wbox), stdev(orig, box)
    print(f"region {box}: web stddev={w:.1f} original stddev={o:.1f}")
    if not (w < 6 and o > 20):
        ok = False
if web.getexif():
    print("web image still has EXIF metadata")
    ok = False
print("REDACTION_OK" if ok else "REDACTION_FAILED")
raise SystemExit(0 if ok else 1)

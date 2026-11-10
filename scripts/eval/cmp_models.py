import sys, pathlib
from PIL import Image
from src.detector import Detector
from src.counting import count_crossers
from src.config import CLASS_NAMES
from ultralytics import YOLO
V = pathlib.Path(sys.argv[1]); yaml = sys.argv[2]
imgs = sorted((V/'images').iterdir()); gts = []
for p in imgs:
    g = dict.fromkeys(CLASS_NAMES, 0)
    for l in (V/'labels'/(p.stem+'.txt')).read_text().splitlines():
        if l.strip(): g[CLASS_NAMES[int(l.split()[0])]] += 1
    gts.append(g)
for w in sys.argv[3:]:
    d = Detector(w); exact = mae = 0; swap = 0
    for p, g in zip(imgs, gts):
        c, _ = count_crossers(d(Image.open(p).convert('RGB')), 0.40)
        e = sum(abs(c[k]-g[k]) for k in CLASS_NAMES); mae += e; exact += e == 0
    m = YOLO(w).val(data=yaml, split='val', imgsz=640, verbose=False, plots=False, workers=0)
    print(f'{pathlib.Path(w).stem:28s} mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  exact {100*exact/len(imgs):.1f}%  MAE {mae/len(imgs):.2f}', flush=True)

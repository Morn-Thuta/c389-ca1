"""Run v4 on valid at several inference sizes and cache raw predictions (conf>=0.01)."""
import pickle, sys
from pathlib import Path
from PIL import Image
from src.detector import Detector
V = Path(sys.argv[1]); W = sys.argv[2]; OUT = Path(sys.argv[3])
imgs = sorted((V/'images').iterdir())
out = {'images': [p.name for p in imgs], 'sizes': [Image.open(p).size for p in imgs], 'preds': {}}
for sz in [512, 640, 768, 896]:
    det = Detector(W, imgsz=sz)
    out['preds'][sz] = [det(Image.open(p).convert('RGB')) for p in imgs]
    print('done', sz, flush=True)
pickle.dump(out, open(OUT, 'wb'))

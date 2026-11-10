"""Compare tuned runs with v4 26m on VALID only: counts, direction errors, hidden/small people."""
import sys, copy
from pathlib import Path
from PIL import Image
from src.detector import Detector
from src.counting import count_crossers, iou
from src.config import CLASS_NAMES

V = Path(sys.argv[1]); WDIR = Path(sys.argv[2])
MODELS = [('v4 26m', 'final_4cls_ig_v4_m.pt', 640), ('A (800)', 'tune_v4_A.pt', 800),
          ('A @640', 'tune_v4_A.pt', 640), ('B', 'tune_v4_B.pt', 640)]
imgs = sorted((V / 'images').iterdir())


def gts(p, size):
    w, h = size; out = []
    for l in (V / 'labels' / (p.stem + '.txt')).read_text().splitlines():
        if l.strip():
            c, x, y, bw, bh = map(float, l.split()[:5])
            out.append((CLASS_NAMES[int(c)], [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def cover(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


data = [(p, Image.open(p).convert('RGB')) for p in imgs]
G = [gts(p, im.size) for p, im in data]
print(f"{'model':9s} {'thr':>4s} {'exact%':>6s} {'MAE':>5s} {'totErr':>6s} {'swaps':>5s} {'missed':>6s} {'extra':>5s} "
      f"{'hidden':>6s} {'small':>5s} {'clear':>5s}  bias[PL,PR,WL,WR]")
for name, wf, sz in MODELS:
    det = Detector(WDIR / wf, imgsz=sz)
    raw = [det(im) for _, im in data]
    for t in [0.3, 0.35, 0.4, 0.45]:
        exact = mae = tot = swaps = missed = extra = 0
        bias = dict.fromkeys(CLASS_NAMES, 0); grp = {'hidden': [0, 0], 'small': [0, 0], 'clear': [0, 0]}
        for (p, im), g, r in zip(data, G, raw):
            c, kept = count_crossers(copy.deepcopy(r), t, image_size=im.size)
            true = {k: sum(x[0] == k for x in g) for k in CLASS_NAMES}
            e = sum(abs(c[k] - true[k]) for k in CLASS_NAMES); mae += e; exact += e == 0
            tot += abs(sum(c[k] for k in CLASS_NAMES) - len(g))
            for k in CLASS_NAMES: bias[k] += c[k] - true[k]
            used = set()
            for i, (cls, b) in enumerate(g):
                best = max([(iou(b, d['box']), j) for j, d in enumerate(kept) if j not in used], default=(0, -1))
                ok = False
                if best[0] >= 0.45:
                    used.add(best[1])
                    if kept[best[1]]['class_name'] == cls: ok = True
                    else: swaps += 1
                else:
                    missed += 1
                occ = max([cover(b, o) for j, (_, o) in enumerate(g) if j != i], default=0)
                h = (b[3] - b[1]) / im.size[1]
                key = 'hidden' if occ >= 0.3 else ('small' if h < 0.1 else 'clear')
                grp[key][0] += ok; grp[key][1] += 1
            extra += len(kept) - len(used)
        n = len(data)
        pct = lambda k: f"{100 * grp[k][0] / grp[k][1]:4.0f}%"
        print(f"{name:9s} {t:4.2f} {100 * exact / n:6.1f} {mae / n:5.2f} {tot / n:6.2f} {swaps:5d} {missed:6d} {extra:5d} "
              f"{pct('hidden'):>6s} {pct('small'):>5s} {pct('clear'):>5s}  {list(bias.values())}", flush=True)

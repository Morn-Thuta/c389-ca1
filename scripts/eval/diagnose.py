"""Where do misses happen? GT crossers split by occlusion; best raw prediction confidence for each."""
import pickle, sys
from pathlib import Path
from src.config import CLASS_NAMES
V = Path(sys.argv[1]); D = pickle.load(open(sys.argv[2], 'rb')); SZ = int(sys.argv[3]) if len(sys.argv) > 3 else 640
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0
def cover(a, b):  # share of a covered by b
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max((a[2]-a[0])*(a[3]-a[1]), 1e-9)
def gt_boxes(name, size):
    w, h = size; out = []
    for l in (V/'labels'/(Path(name).stem+'.txt')).read_text().splitlines():
        if not l.strip(): continue
        c, x, y, bw, bh = l.split()[:5]; x, y, bw, bh = float(x)*w, float(y)*h, float(bw)*w, float(bh)*h
        out.append((int(c), [x-bw/2, y-bh/2, x+bw/2, y+bh/2]))
    return out
buckets = {}
rows = []
for name, size, preds in zip(D['images'], D['sizes'], D['preds'][SZ]):
    gts = gt_boxes(name, size)
    for i, (c, b) in enumerate(gts):
        occ = max([cover(b, o) for j, (_, o) in enumerate(gts) if j != i], default=0)
        grp = 'occluded(>=30% covered)' if occ >= 0.3 else 'partly(10-30%)' if occ >= 0.1 else 'clear(<10%)'
        best_any = max([p['confidence'] for p in preds if iou(b, p['box']) >= 0.5], default=0)
        best_cls = max([p['confidence'] for p in preds if iou(b, p['box']) >= 0.5 and p['class_name'] == CLASS_NAMES[c]], default=0)
        kind = 'wheeled' if c >= 2 else 'person'
        for key in [grp, f'{grp}|{kind}']:
            d = buckets.setdefault(key, {'n':0,'>=0.4':0,'0.2-0.4':0,'0.05-0.2':0,'none':0,'wrongcls>=0.4':0})
            d['n'] += 1
            if best_cls >= 0.4: d['>=0.4'] += 1
            elif best_any >= 0.4: d['wrongcls>=0.4'] += 1
            elif best_any >= 0.2: d['0.2-0.4'] += 1
            elif best_any >= 0.05: d['0.05-0.2'] += 1
            else: d['none'] += 1
        h = (b[3]-b[1]) / size[1]
        rows.append((grp, kind, h, best_cls, best_any))
print(f'imgsz {SZ}: best prediction (IoU>=0.5) per ground-truth crosser')
print(f"{'group':38s} {'n':>4s} {'found>=0.4':>10s} {'wrongcls':>8s} {'0.2-0.4':>8s} {'0.05-0.2':>8s} {'missed':>7s}")
for k in sorted(buckets):
    d = buckets[k]; n = d['n']
    print(f"{k:38s} {n:4d} {100*d['>=0.4']/n:9.0f}% {100*d['wrongcls>=0.4']/n:7.0f}% {100*d['0.2-0.4']/n:7.0f}% {100*d['0.05-0.2']/n:7.0f}% {100*d['none']/n:6.0f}%")
# by size
print('\nby box height (share of image), found >=0.4 with right class:')
for lo, hi in [(0,0.1),(0.1,0.2),(0.2,0.35),(0.35,1.01)]:
    sel = [r for r in rows if lo <= r[2] < hi]
    if sel: print(f'  {lo:.2f}-{hi:.2f}: n={len(sel):3d}  found {100*sum(r[3]>=0.4 for r in sel)/len(sel):3.0f}%   any-class>=0.2 {100*sum(r[4]>=0.2 for r in sel)/len(sel):3.0f}%')

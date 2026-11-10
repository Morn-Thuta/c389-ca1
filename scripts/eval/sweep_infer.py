"""Count accuracy on valid over inference size x threshold x duplicate IoU (cached predictions)."""
import pickle, sys
from pathlib import Path
import src.counting as counting
from src.config import CLASS_NAMES
V = Path(sys.argv[1]); D = pickle.load(open(sys.argv[2], 'rb'))
gts = []
for name in D['images']:
    g = dict.fromkeys(CLASS_NAMES, 0)
    for l in (V/'labels'/(Path(name).stem+'.txt')).read_text().splitlines():
        if l.strip(): g[CLASS_NAMES[int(l.split()[0])]] += 1
    gts.append(g)
res = []
for sz, preds in D['preds'].items():
    for dup in [0.5, 0.6, 0.7, 0.8, 0.9]:
        counting.DUPLICATE_IOU = dup
        for t in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            exact = mae = tot_err = 0; bias = dict.fromkeys(CLASS_NAMES, 0)
            for g, r in zip(gts, preds):
                c, _ = counting.count_crossers(r, t)
                e = sum(abs(c[k]-g[k]) for k in CLASS_NAMES); mae += e; exact += e == 0
                tot_err += abs(sum(c[k] for k in CLASS_NAMES) - sum(g.values()))
                for k in CLASS_NAMES: bias[k] += c[k]-g[k]
            n = len(gts); res.append((exact/n, -mae/n, sz, dup, t, tot_err/n, list(bias.values())))
res.sort(reverse=True)
print('rank  exact%  MAE/img  totalErr/img  imgsz  dupIoU  thr   bias[PL,PR,WL,WR]')
for i, r in enumerate(res[:15]):
    print(f'{i+1:3d}   {100*r[0]:5.1f}   {-r[1]:5.2f}    {r[5]:5.2f}       {r[2]:4d}   {r[3]:.1f}    {r[4]:.2f}  {r[6]}')
cur = [r for r in res if r[2] == 640 and r[3] == 0.7 and r[4] == 0.4][0]
print(f'\ncurrent app (640, 0.7, 0.40): exact {100*cur[0]:.1f}%  MAE {-cur[1]:.2f}  totalErr {cur[5]:.2f}  bias {cur[6]}')
print('\nbest per imgsz:')
for sz in D['preds']:
    b = max(r for r in res if r[2] == sz); print(f'  {sz}: exact {100*b[0]:.1f}%  MAE {-b[1]:.2f}  dup {b[3]}  thr {b[4]}')

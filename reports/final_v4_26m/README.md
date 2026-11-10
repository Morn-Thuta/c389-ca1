# Training evidence: final model `final_4cls_ig_v4_m` (YOLO26m)

The model the app and the report use. Everything here is copied from the Colab run and our evaluation
runs. Photos are left out on purpose: the sample batches (`train_batch*.jpg`, `val_batch*.jpg`) show real
people, so they stay on the team's private Drive. The weights (`.pt`, 44 MB) are not in git either.

## How it was trained

| | |
|---|---|
| Where | Google Colab, Tesla T4, 150 epochs in 2.8 h (`training/results.csv`, column `time`) |
| Base | `yolo26m.pt` (COCO weights) |
| Data | `final_ig_v4`: 1517 train (428 own + 1,089 cited outside photos, train only), 132 valid, 66 test |
| Settings | imgsz 640, batch 16, seed 0, `fliplr 0.0`, ultralytics 8.4.120 (full list: `training/args.yaml`) |
| Classes | person_left 0, person_right 1, wheeled_left 2, wheeled_right 3 |

## Files

| Folder | What it shows |
|---|---|
| `training/CA1_train_v4_26m_colab_log.ipynb` | The notebook that trained the model, with its original Colab output: install, the 150-epoch training log and the final valid table. The two photo outputs were removed. |
| `training/results.csv`, `results.png` | Loss and mAP for every epoch |
| `training/Box*_curve.png`, `confusion_matrix*.png` | Valid-set curves and confusion matrix at the end of training |
| `training/args.yaml` | Every training setting Ultralytics used |
| `test/test_results.txt` | The one test-set run (7 Nov), detection and counting |
| `test/*.png` | Test-set curves and confusion matrix |
| `model_selection/` | How the final model was chosen, on the **valid** set only (the test set was not used to choose) |

## Results

| | mAP50 | mAP50-95 | All four counts exact | Count error per photo |
|---|---|---|---|---|
| Valid (132 photos) | 0.931 | 0.624 | 59.8% | 0.79 |
| Test (66 photos, run once) | 0.858 | 0.561 | 53.0% | 1.21 |

On the test set no pedestrian was counted as wheeled and no wheeled crosser as a pedestrian.

## Models we compared (valid set, threshold 0.40)

| Run | mAP50 | Exact | Count error |
|---|---|---|---|
| **v4 26m (final)** | 0.931 | **59.8%** | **0.79** |
| v4 26s | 0.898 | 52.3% | 0.98 |
| v5 fine-tune | 0.932 | 50.8% | 1.04 |
| v5 mirror | 0.919 | 56.1% | 0.89 |
| Tuning A (imgsz 800) | | 53.8% | 1.04 |
| Tuning B (cls 1.0) | | 58.3% | 0.83 |

Sources: `model_selection/valid_model_map_and_counts.txt` and `valid_tuning_runs_AB.txt`. The scripts that
produced them are in [`scripts/eval/`](../../scripts/eval/).

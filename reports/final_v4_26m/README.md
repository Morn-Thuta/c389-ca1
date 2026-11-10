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

## Known limits

- **Hidden and small people are missed more often.** On valid, the model finds 89% of people in clear view
  but 67% of those at least 30% covered by someone else, and 59% of those under 10% of the photo's height
  (`model_selection/valid_hidden_small_v4.txt`).
- **Kick-scooter and skateboard riders are counted as pedestrians.** The training data has almost none.
  We saw this on extra campus photos that were not used for training or scoring.
- **Some people are counted in the wrong direction.** On test, 20 people moving left or right were given the
  opposite direction (`test/test_results.txt`). Bicycles do better because their shape shows the direction.
- **Flip augmentation is off** (`fliplr 0.0`), because a flipped photo would swap left and right.
- **Small test set:** 66 photos from the same campus crossings, so the scores may not hold at other places
  or in other light.

Sources: `model_selection/valid_model_map_and_counts.txt` and `valid_tuning_runs_AB.txt`. The scripts that
produced them are in [`scripts/eval/`](../../scripts/eval/).

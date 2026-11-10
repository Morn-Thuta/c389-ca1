# Evaluation scripts (valid set)

Used to compare models and app settings before the one test run. Run them from the repo root as modules,
for example `python -m scripts.eval.cmp_models VALID_DIR data.yaml a.pt b.pt`. `VALID_DIR` is a YOLO split folder with `images/` and `labels/`; the outputs are saved in
[`reports/final_v4_26m/model_selection/`](../../reports/final_v4_26m/model_selection/).

| Script | What it does | Output |
|---|---|---|
| `cmp_models.py VALID_DIR DATA_YAML WEIGHTS...` | mAP from Ultralytics val plus app counting at 0.40 for each model | `valid_model_map_and_counts.txt` |
| `compare_runs.py VALID_DIR WEIGHTS_DIR` | v4 26m vs tuning runs A and B: exact counts, direction swaps, misses, hidden and small people | `valid_tuning_runs_AB.txt` |
| `cache_preds.py VALID_DIR WEIGHTS OUT.pkl` | Saves raw predictions at several image sizes, so the next two run fast | (cache, not kept) |
| `sweep_infer.py VALID_DIR OUT.pkl` | Count accuracy over image size, threshold and duplicate IoU | `valid_inference_sweep_v4.txt` |
| `diagnose.py VALID_DIR OUT.pkl` | Which people are missed: clear, partly covered, half hidden or small | `valid_hidden_small_v4.txt` |

The dataset and weights are not in git; point the paths at the private copies.

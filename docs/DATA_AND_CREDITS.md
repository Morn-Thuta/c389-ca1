# Data and credits

## Our dataset (`final_ig_v4`)

| Split | Own photos | Outside photos | Total |
|---|---|---|---|
| Train | 428 | 1,089 | 1,517 |
| Valid | 132 | 0 | 132 |
| Test | 66 | 0 | 66 |

- **Own photos (626):** taken by the team at campus pedestrian crossings and labelled in Roboflow with the
  [labelling guide](labelling-guide.pdf). They are split 70/20/10 by burst group (`src/data/resplit.py`), so
  near-identical shots never sit in two splits (`src/data/dedupe_check.py`).
- **Outside photos (1,089), training set only:** valid and test hold only our own photos, so the scores
  measure our crossings.
- **Classes:** `person_left` 0, `person_right` 1, `wheeled_left` 2, `wheeled_right` 3. A rider and their bike
  are one `wheeled_*` box, and walking or carrying a bike also counts as wheeled.
- **Test set:** 66 photos, 262 boxes (person_left 106, person_right 110, wheeled_left 30, wheeled_right 16).
  It was evaluated once, after the model and the app settings were fixed.
- **Privacy:** the photos show real people, so they, the dataset and the weights stay on the team's private
  Drive and Roboflow project, never in this repo. Everything is deleted at the end of the module.

## Outside datasets (CC BY 4.0, attribution required)

1. chakrit. *Person-Direction-Counter* (Version 3) [Dataset]. Roboflow Universe. CC BY 4.0.
   https://universe.roboflow.com/chakrit-4e4gq/person-direction-counter
   - 932 photos used. `walking_left` and `walking_right` were renamed `person_left` and `person_right`.
   - Front-facing images were dropped, video frames were thinned (every 3rd kept), and images were resized from
     640 to 512.
2. Kates Workspace. *cyclists_database* [Dataset]. Roboflow Universe. CC BY 4.0.
   https://universe.roboflow.com/kates-workspace-l2am8/cyclists_database-8twra
   - 157 photos of side-on cyclists used. Front and back views were dropped.
   - The direction was suggested by our earlier model, and low-confidence photos were dropped. Frames were
     thinned (every 2nd kept).

## Software and pretrained weights

- Ultralytics YOLO, version 8.4.120. https://github.com/ultralytics/ultralytics. YOLO26m was trained from
  its COCO-pretrained weights (`yolo26m.pt`).
- Lin, T.-Y. et al. (2014). Microsoft COCO: Common objects in context. ECCV 2014.
  https://doi.org/10.1007/978-3-319-10602-1_48
- Gradio, the web page. Abid, A. et al. (2019). arXiv:1906.02569. https://arxiv.org/abs/1906.02569
- Roboflow, used for labelling and exporting the dataset. https://roboflow.com

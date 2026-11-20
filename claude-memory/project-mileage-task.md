---
name: project-mileage-task
description: "Current modelling task - predict mileage of 2010-2024 cars (not price); pilot done 2026-11-20, feature work next on the same test split"
metadata:
  node_type: memory
  type: project
  originSessionId: 863030b2-bf46-4af3-93bc-9361b0c03370
  modified: 2026-11-20T15:10:38.502Z
---

The user chose to predict **mileage, not price**, for model years **2010-2024**, all regions
(older cars: declared mileage is mostly missing or meaningless). Pilot (2026-11-20): Ridge with
base features + brand x model + age^2, CV MAE 44.5k km, test MAE 44.1k km; diagnosis is
underfitting, so the next step is feature work (age x segment interactions such as taxi models /
LPG, price relative to same model-year, target cleaning of placeholders like 55555 and VAZ
odometer roll-over).

**Why:** the user wants every later iteration measured against the pilot numbers on the same
held-out cars.

**How to apply:** never re-draw the split - it is stored in
`mileage/data/mileage_2010_2024.parquet`, which is gitignored (`data/`), and is regenerated
deterministically by `mileage/build_dataset.py` (random_state 42). Append new experiments to
`mileage/results/experiments.csv` instead of overwriting the pilot rows.
Related: [[feedback-linear-models-checklist]], [[reference-github-repo]].

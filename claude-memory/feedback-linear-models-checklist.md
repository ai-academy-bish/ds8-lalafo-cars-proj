---
name: feedback-linear-models-checklist
description: Modelling must follow linear_regression_checklist.md with linear models (LinearRegression/Ridge) - it is a study project
metadata:
  node_type: memory
  type: feedback
  originSessionId: 863030b2-bf46-4af3-93bc-9361b0c03370
  modified: 2026-11-20T15:10:33.964Z
---

Modelling work follows `linear_regression_checklist.md` (repo root) and uses linear models
(LinearRegression, Ridge; Lasso/ElasticNet acceptable). Work is framed as: hypotheses ->
one change per experiment -> 5-fold CV on train -> one final test run -> metrics report.

**Why:** the user said this is a study project and asked to stick to the checklist plan
(2026-11-20); first a pilot with measurements, then deeper feature work compared against it.

**How to apply:** map report sections to checklist stages and keep a coverage table; log every
experiment to `mileage/results/experiments.csv`; tree/boosting models only as an optional
upper-bound comparison, never as the main model, unless the user changes the plan.
Related: [[project-mileage-task]].

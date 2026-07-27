# Drift Monitoring Analysis

## Setup
The reference distribution is the training dataset (`data/heart.csv`, duplicates removed).
Production data is simulated by shifting three features to imitate an incoming
population that is older, with higher cholesterol and lower peak heart rate:
- `age` shifted up by 8 to 14 years
- `chol` scaled up by 20 percent
- `thalach` (max heart rate) shifted down by 10 to 19 bpm

Drift is measured with Evidently's `DataDriftPreset`. The script exits non-zero
if the overall drift share exceeds a configurable threshold (default 0.5).

## Which features showed drift, and why?
Three features drifted: `age`, `chol`, and `thalach`. This is expected, because
these are exactly the features the simulation shifted. The remaining features
were left unchanged and did not drift. The overall drift share was 0.23 (roughly
one quarter of features), which stayed under the alert threshold.

## Would this drift likely affect model performance?
Very likely, yes. Age and maximum heart rate are among the stronger predictors
of heart disease in this dataset, so a real shift of this size in those features
would push incoming records into regions the model saw less often during training.
Cholesterol is a weaker but still relevant signal. Because the drift lands on
influential features rather than trivial ones, degraded recall and accuracy would
be a real risk in production.

## What action would you recommend?
Investigate first, then retrain. A drift share of 0.23 is meaningful but not
catastrophic, so the immediate step is to confirm the shift is real and not a
data-pipeline artifact (for example, a changed unit or a new measurement device).
If the shift is genuine and validated against fresh labels, retrain the model on
recent data so it reflects the current population. If no labels are yet available,
continue monitoring at a tighter threshold while collecting them.
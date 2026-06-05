# Booking Archetype Clustering in Hotel Booking
Luís Serrano (60253) · Tiago Fonseca (72898) · Miguel Teixeira (72922)

## Problem
Do hotel bookings cluster into distinct profiles based on pre-arrival characteristics — and do those profiles show consistent patterns across other booking dimensions?

## Overview
This study asks a simple question: can we identify distinct booking behaviour profiles from hotel reservation data, using only pre-booking features — no cancellation labels, no revenue signals?
We test three clustering approaches (K-Means, iK-Means, and GMM) on the Hotel Booking Demand dataset (António et al., 2019), covering 119,192 records across a 47-dimensional mixed-type feature space. K-Means with k = 8 comes out as the strongest model (Silhouette = 0.145), while GMM adds nuance by capturing softer cluster boundaries. The resulting clusters map onto interpretable booking patterns across lead time, customer type, and distribution channel.
No ground-truth labels exist, so all evaluation is internal — results should be interpreted with that in mind.
 
## How to Run


From the project root:
 
| Goal | Linux / macOS | Windows |
|---|---|---|
| Full run (all tasks + extensions) | `python3 run_all.py` | `python run_all.py` |
| Core tasks only | `python3 run_all.py --core-only` | `python run_all.py --core-only` |
 
All results are saved to the `figures/` and `tables/` folders.
 
---
 
## Execution Order
 
| Step | Notebook | Content |
|---|---|---|
| 1 | `src/project.ipynb` | Task 1: K-Means baseline |
| 2 | `src/task2_gmm.ipynb` | Task 2 + 3.2: GMM + AIC/BIC |
| 3 | `src/task3_evaluation.ipynb` | Task 3.1 + 3.3 + 3.4 |
| 4 | `src/task4_repository.ipynb` | Task 4.2: repo deliverables |
| 5 | `src/extension_E2_spectral.ipynb` | Extension E2: Spectral clustering *(skipped with --core-only)* |
| 6 | `src/extension_E5_visualization.ipynb` | Extension E5: t-SNE *(skipped with --core-only)* |

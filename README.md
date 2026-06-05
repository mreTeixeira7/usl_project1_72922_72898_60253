# Booking Archetype Clustering in Hotel Booking
Luís Serrano (60253) · Tiago Fonseca (72898) · Miguel Teixeira (72922)

## Problem
Do hotel bookings cluster into distinct profiles based on pre-arrival characteristics — and do those profiles show consistent patterns across other booking dimensions?

## Overview
This study asks a simple question: can we identify distinct booking behaviour profiles from hotel reservation data, using only pre-booking features — no cancellation labels, no revenue signals?
We test three clustering approaches (K-Means, iK-Means, and GMM) on the Hotel Booking Demand dataset (António et al., 2019), covering 119,192 records across a 47-dimensional mixed-type feature space. K-Means with k = 8 comes out as the strongest model (Silhouette = 0.145), while GMM adds nuance by capturing softer cluster boundaries. The resulting clusters map onto interpretable booking patterns across lead time, customer type, and distribution channel.
No ground-truth labels exist, so all evaluation is internal — results should be interpreted with that in mind.

## How to Run
Open run_all.py and press run. All results will be saved to the figures/ and tables/ folders.
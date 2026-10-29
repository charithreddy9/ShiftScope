# Research notes

## Question
When the input distribution changes, can confidence calibration and prediction sets still be trusted?

## Hypothesis
Gaussian noise applied only to test images will lower classifier accuracy and may produce overconfident wrong predictions. Temperature scaling fitted on clean data may not generalize to corrupted images. Nominal split-conformal coverage should not be assumed under shift.

## Protocol
Use the built-in 1,797-image scikit-learn 8x8 digits set. For each seed, create disjoint stratified training (60%), temperature validation (10%), conformal calibration (10%), and test (20%) subsets. Fit logistic regression and 150-tree random forest models. Fit temperature by validation negative log likelihood and the conformal threshold on a separate clean calibration subset. Apply Gaussian noise only at test time at sigma 0, 1.5, 3.0 and 4.5, clip pixels to [0,16]. Measure accuracy, ECE, NLL, Brier, AURC and observed conformal coverage.

## Interpretation
Calibrated confidence is not the same as accuracy. Standard conformal coverage assumes exchangeable calibration and test samples; the synthetic shift violates that assumption. Changes in coverage are empirical observations, not a disproof of the guarantee.

## Limitations
Small toy dataset; synthetic noise is not all real-world shift; only two models and three seeds. No independent test cohort, hypothesis tests, confidence intervals, or published-paper claims. Use the reproducible CSV and plots for a scoped, honest assessment.

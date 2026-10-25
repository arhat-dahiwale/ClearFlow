ClearFlow Phase 06 - SHAP exports
================================
Everything here was computed once in Python; Tableau/Power BI should only read it.
No models were retrained. All 200 Phase 05 artifacts were explained on the SAME
1,200 test rows (100 per class) listed in shap_sample_manifest.json.

Files
-----
shap_configuration_summary.csv  one row per configuration: explainer, SHAP output
    scale, additivity error, sample-only metrics (explanation_sample_*), full-test
    metrics from Phase 05 (full_test_*), top feature.
shap_class_feature_summary.csv  one row per configuration x explained class x feature.
    *_all_rows   : over all 1,200 rows (global view).
    *_true_class_rows : only rows whose true class is the explained class.
    mean_signed_* > 0 pushes the output TOWARD the explained class, < 0 AWAY.
    mean_positive_/mean_negative_ split the two directions so they cannot cancel.
shap_sample_predictions.csv     prediction + confidence per configuration x sample row.
shap_sample_features.csv        the 1,200 sampled rows (features + true label + ids).
shap_sample_manifest.json       seed, positions and counts of the sample.
shap_manifest.json              versions, explainers, scales, artifact hashes, caveats.
shap_local_cases.csv            top contributions for each local-explanation plot.
local_explanations/             waterfall PNGs for correct predictions and errors.

How to read it
--------------
* Scales differ by model family (probability / log-odds / logit). Compare feature
  RANKS and SIGNS across families, not raw magnitudes.
* share_of_class_total_abs_shap is a within-configuration, within-class share and
  is the safest quantity to put on one chart across families.
* explanation_sample_* metrics come from a class-balanced 1,200-row sample. Use
  full_test_* columns (Phase 05) for model comparison.
* SHAP describes model behaviour, not causes of attacks.

ClearFlow — Phase 04 Feature Selection exports

Input
- data/processed/train.csv.gz (Phase 03 fixed training split)
- exports/phase03_split/split_manifest.json (feature list, target, class order)
- The test split is never read or used for feature selection.

Methods
1. Random Forest impurity-based feature importance
2. ANOVA F-test (chosen instead of chi-square because numeric features need not be non-negative)
3. RFE with a Decision Tree estimator

Sampling
The three methods use the same reproducible, class-capped reservoir sample from the training split only.
The default cap is 10,000 rows per class (up to 120,000 rows total). Smaller classes are retained in full.
This limits computation and reduces the dominance of the majority class in feature ranking. It is not
model-training class balancing; the training split itself is not changed.

Files
- feature_rankings_long.csv: one row per method-feature ranking (3 x 70 rows)
- feature_rankings_wide.csv: one row per feature with rank/score columns for all methods
- feature_sets_long.csv: feature membership of each of the 10 sets
- feature_set_summary.csv: one row per feature set
- feature_set_overlap.csv: pairwise intersections and Jaccard similarities for the 10 sets
- selection_sample_summary.csv: number of rows sampled from each training class
- feature_sets.json: ordered feature lists for subsequent model training
- feature_selection_manifest.json: parameters, provenance and caveats

Feature sets
- rf_top10, rf_top20, rf_top40
- anova_top10, anova_top20, anova_top40
- rfe_top10, rfe_top20, rfe_top40
- all_features: all 70 features in Phase 3 schema order (shared baseline)

Important
Feature-set score columns have method-specific meanings. Compare ranks and membership, not raw scores
across methods. The selection sample is capped by class; document this when interpreting rankings.

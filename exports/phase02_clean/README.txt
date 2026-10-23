ClearFlow — Phase 2 clean-data exports
====================================

Main cleaned model dataset (local, compressed; not committed to Git):
  data/processed/cleaned_flows.csv.gz

Rows include provenance columns 'dataset' and 'source_file', feature columns,
and 'Label'. Drop 'dataset' and 'source_file' before model fitting.
Destination Port is intentionally retained. The duplicate
'Fwd Header Length.1' is excluded. Heartbleed and Infiltration are excluded
from modeling, but retained in Phase 1 raw visualizations.

CSV exports in this folder:
  preprocessing_summary_by_file.csv
    Mutually exclusive row-removal counts per input file and final row counts.
  preprocessing_summary_totals.csv
    Stage totals by source dataset and overall.
  class_counts_before_after.csv
    Raw harmonized vs cleaned class counts and percentages by source file.
  clean_class_counts_by_dataset.csv
    Cleaned class counts for charting; use this rather than counting the sample.
  feature_preprocessing_audit.csv
    Feature min/max after ambiguous-label removal and zero-variance decision.
  column_policy.csv
    Documented metadata exclusions and the Destination Port decision.
  ambiguous_feature_summary.csv / ambiguous_label_pairs.csv
    Conflicting input-vector counts and label pairs after exact raw deduplication.
  clean_feature_profile_by_dataset.csv
    Min, max, mean, and zero frequency on the full cleaned data.
  clean_visualization_sample_wide.csv
    Deterministic, class-covered sample with all retained features.
  clean_feature_distribution_sample_wide.csv / _long.csv
    Sample for comparing selected feature distributions to Phase 1.

Sample note:
  The visualization sample is capped per class per input chunk to make
  minority classes visible. It is not prevalence-representative; do not use it
  to calculate class counts. Use the clean class-count table instead.

Methodology note:
  Exact duplicates are identified from all original row fields within each
  dataset before metadata is removed. After harmonization, complete numeric
  feature vectors associated with multiple labels are excluded as ambiguous.
  No balancing, train/test splitting, or scaling is performed in Phase 2.

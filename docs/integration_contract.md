# VeriMedia AI Integration Contract

## Purpose

Every forensic module must return a standardized `ModuleResult`.

## Required fields

- `analysis_id`: Unique analysis identifier.
- `media_type`: `image`, `video`, `audio`, or `multimodal`.
- `module`: Module name.
- `status`: `success`, `partial`, `failed`, or `unavailable`.
- `scores`: Numeric scores between `0.0` and `1.0`.
- `evidence`: Structured forensic evidence.
- `artifacts`: Generated files or visualizations.
- `timestamps`: Suspicious time intervals where applicable.
- `warnings`: Non-fatal limitations.
- `errors`: Failure details.
- `model_versions`: Detector and model versions.

## Score direction

Higher suspiciousness scores indicate stronger manipulation or synthetic-generation indicators.

The baseline common score is:

```text
suspicion_score
# Evaluation and reproducibility

## Historical project results

The January 29, 2026 final presentation is the source of the following results. They are reported project measurements, not results independently reproduced from this cleaned repository.

| Configuration | Classification accuracy | Mean pixel-level AUROC |
| --- | ---: | ---: |
| RegAD, raw client images | 93% | 69.15% |
| RegAD, preprocessed images | 100% | 87.42% |
| PatchCore, initial configuration | 87.6% | 63.57% |
| PatchCore, enhanced configuration | 100% | 83.02% |

Source: presentation pages 13–17. RegAD's reported localization AUROC exceeds enhanced PatchCore by 4.40 percentage points. Classification accuracy is tied. The raw/preprocessed RegAD difference is 18.27 percentage points; it is a pipeline comparison, not an isolated architectural ablation.

The slide inventory lists five categories, 14 OK images and nine not-OK images, with synthetic anomalies included. It does not establish the independent test count or support/test split. These inventory counts must not be presented as a confirmed test-set size. Per-category sample counts, real-versus-synthetic breakdowns, threshold selection, and averaging conventions need the original experiment records.

Runtime is reported as approximately 1.1 seconds for RegAD and 0.013 seconds for enhanced PatchCore. The RegAD slide uses “each class”; a common per-image timing protocol, hardware, warm-up, support construction, and preprocessing costs are not established. No speedup claim is made here.

## Current evaluation interface

`model_evaluation_AUROC.py` computes pooled pixel-level AUROC and image-level AUROC for one supplied test folder. The image score is the maximum pixel anomaly score. It does not compute thresholded classification accuracy or automatically average results across categories.

Every test image must have an explicit ground-truth mask, including an all-zero mask for normal images. Missing masks stop evaluation. Undefined AUROC is reported as `nan`, not zero. AUROC needs both labels at the relevant level: normal/anomalous images for image AUROC, and normal/anomalous pixels for pixel AUROC.

Run categories separately and document the aggregation method. Preserve source-image identity across splits: synthetic and augmented derivatives must remain with their source image. Calibrate thresholds on separate validation data, then hold them fixed for testing. Report real and synthetic defects separately, plus any combined result.

## Changes made during repository cleanup

- Consolidated custom inference into `inference.py`.
- Replaced test-batch min/max score normalization with raw scores and a required, externally calibrated classification threshold. Historical thresholds such as 0.05 cannot be transferred to this scale.
- Preserved per-image normalization for visual display only. The visualization's 0.66 mask cutoff is not a calibrated segmentation operating point.
- Corrected custom-path embedding device selection and single-image score-map dimensions.
- Required masks explicitly and preserved undefined AUROC as `nan`.

These changes improve the interface and evaluation safeguards. They do not retroactively validate or regenerate the presentation results. Original client images, ground-truth masks, final run logs, and checkpoints are not bundled. Full model execution and historical-result reproduction remain dependent on those inputs and a compatible environment.

## Recommended experiment record

For each run, record the commit, checkpoint provenance, support filenames and count, test filenames, masks, real/synthetic status, image resolution, seed, threshold calibration split, hardware, package versions, per-image scores, and timing boundaries. Report false negatives and false positives alongside accuracy. With a small dataset, retain failure examples and avoid generalizing perfect observed accuracy to unseen production data.

# Project scope and attribution

## Industrial problem

Foil-stamping defects can persist through a print run before an operator notices them. The project investigated reference-based anomaly detection with few normal examples, under reflections, shadows, varying image alignment, and limited real defects.

## Team workflow described in the presentation

1. Prepare normal reference images and annotate defect masks.
2. Remove background with BiRefNet and standardize geometry with perspective transformation, resizing, and centering.
3. Compare few-shot anomaly detection methods, including RegAD and PatchCore.
4. Produce anomaly maps and image-level decisions.
5. Explore ORB matching and homography for positional drift that the anomaly model missed.

The final presentation discusses missed small defects, incomplete background removal, and incorrect predictions. More real defects, consistent lighting, and controlled camera alignment are the principal next steps.

## What this repository contains

This repository focuses on the RegAD adaptation: custom support/test folder loading, feature-based anomaly scoring, visualizations, and mask-based evaluation. It also retains the inherited MVTec training/testing scripts and core model implementation. The full BiRefNet, perspective-correction, ORB, and PatchCore pipelines described by the team are not implemented here. Supply already preprocessed images to the custom inference interface.

Maintainer: [Tarunno360](https://github.com/Tarunno360). The maintainer adapted the RegAD implementation for the project. Detailed individual ownership of every team component is not assigned by the presentation, so shared components are credited to the project team.

Team members, as listed on the presentation: Md. Samsul Arefin, Md. Rumman Shahriar, SM Azmain Faisal, Shahriar Khandoker, and Sabuj Majumder.

## Prior work

RegAD was introduced by Chaoqin Huang, Haoyan Guan, Aofan Jiang, Ya Zhang, Michael Spratling, and Yan-Feng Wang in *Registration based Few-Shot Anomaly Detection*, ECCV 2022.

- [Original paper](https://arxiv.org/abs/2207.07361)
- [Original implementation](https://github.com/MediaBrain-SJTU/RegAD)

The original MIT copyright notice is retained in `LICENSE`. Registration-based feature learning and the inherited architecture are credited to the original authors. This project's contribution is the industrial adaptation and experimental workflow.

## Presentation

[Final presentation, January 29, 2026](final-presentation.pdf) — 26 slides. This web-optimized visual copy preserves the slide content and team credits; text is rasterized. The slides describe the wider team project and retain the historical results. For current evaluation behavior, see [evaluation notes](evaluation.md).

Client data and weights are not distributed in this repository. The presentation retains the client examples supplied for inclusion by the repository owner; repository code licensing does not imply a separate license to redistribute third-party images or client data.

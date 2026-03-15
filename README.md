# Trust Vision – Adversarial Robustness Lab

## Problem Statement
Deep vision models (medical + industrial) are brittle: tiny pixel perturbations or mislabeled features can flip diagnoses/inspections, undermining safety and trust.

## Possible Solutions
- Adversarial training (FGSM/PGD/TRADES) to harden classifiers.
- Certified robustness bounds to quantify worst-case perturbations.
- Runtime defenses (denoising, input filtering).
- Monitoring for drift and elevated attack-success rates.
- Model/attack benchmarking (FGSM, PGD, CW, AutoAttack) with visualizations.

## Our Solution
A unified FastAPI + static web UI that:
- Runs pixel-space attacks (FGSM/PGD/CW/AutoAttack) on a ResNet-18/CLIP classifier.
- Shows clean vs. adversarial vs. defended images, heatmaps, and certified L2 radius.
- Offers a multimodal demo (X-ray + vitals) to illustrate risk across modalities.
- Ships as a single Docker image for Hugging Face Spaces (free CPU tier).

## How Our Solution Differs
- **End-to-end demo in one container**: attacks, defenses, UI, and static assets ship together—no external notebooks or services required.
- **Visualization-first**: heatmaps, certified radii, and attack/defense comparisons surface failure modes quickly, not just metrics.
- **Multimodal lens**: a lightweight X-ray + vitals demo shows robustness issues beyond single-image classifiers.
- **No GPU dependency**: runs on free CPU tiers (HF Spaces) so teams can explore robustness without specialized hardware.

## Our View
- Robustness must be demonstrated, not assumed. We surface failure cases visually and with metrics.
- Lightweight, reproducible demos help teams reason about risk before deploying larger models.

## How It Solves the Problem
- **Attack surface**: Generate adversarial examples on-demand to see real misclassifications.
- **Defense surface**: Apply simple denoise/blur defense and compare outputs.
- **Evidence**: Heatmaps and certified radii show where and how the model is fragile.
- **Deployment**: One-click Space deploy makes sharing and testing easy.

## Working of the Model
1. Input image → preprocessing (resize/crop/normalize).
2. Classifier: ResNet-18 (ImageNet weights) or CLIP ViT-B/32 zero-shot (medical/industrial labels).
3. Optional detector: YOLOv8n for multi-object cases.
4. Attacks: FGSM / PGD / CW / AutoAttack in pixel space.
5. Defense: Median + Gaussian denoise, then reclassify.
6. Metrics: attack success, rolling window alert, certified L2 radius approximation.

### “para meter based attack” (multimodal) flow
- Static, client-side demo (no backend calls) that stitches a sample chest X-ray and synthetic vitals.
- JavaScript randomly picks one of four sample images in `para meter based attack/chest/files/*/0.jpg` and generates vitals.
- Renders Grad-CAM-style overlay and probability bars purely in the browser.

## How the Files Are Linked (diagram)


## Accuracy / Robustness Snapshot
- Clean top-1 (ImageNet weights) ~69–71% (reference ResNet-18).
- Certified L2 radii are small (~0.02–0.03 on demo samples): highlights fragility.
- Defense recovers many low-ε attacks but not all; this is a *demonstration*, not a production guarantee.

## Reference Papers
- Goodfellow et al., “Explaining and Harnessing Adversarial Examples” (FGSM).
- Madry et al., “Towards Deep Learning Models Resistant to Adversarial Attacks” (PGD/TRADES).
- Carlini & Wagner, “Towards Evaluating the Robustness of Neural Networks” (CW).
- Croce & Hein, “Reliable Evaluation of Adversarial Robustness with AutoAttack.”

## Similar/Public Models & Tooling
- TRADES-robust ResNet checkpoints (Madry Lab).
- RobustBench: curated robustness leaderboard and baselines.
- YOLOv8 (Ultralytics) for detection; not adversarially trained by default.

##Images :

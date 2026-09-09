# Mitigating Dataset Artifacts in NLI via Dataset Cartography

Final project for UT Austin's AI 388 (Natural Language Processing, Fall 2025). This project diagnoses and mitigates dataset artifacts in Natural Language Inference (NLI) by fine-tuning ELECTRA-small on SNLI. The pipeline covers artifact diagnosis, dataset cartography based filtering, hard negative mining, and back-translation augmentation. Models are evaluated on both in-domain accuracy (SNLI) and out-of-domain robustness (ANLI).

## Part I: Diagnosing the Artifact

This section implements two diagnostic tests to confirm the SNLI dataset contains exploitable artifacts.

* N-gram Artifact Analysis: Computes the conditional probability P(label | word) over hypothesis tokens. This identifies words (such as "nobody") that act as near deterministic shortcuts for the label, independent of the premise.
* Hypothesis-Only Ablation: Trains a model with the premise blanked out. This model reached 60.0% accuracy, well above the 33.3% random baseline, quantifying how much the standard model relies on hypothesis-only shortcuts rather than genuine inference.

## Part II: Mitigation via Dataset Cartography

This section implements a custom CartographyCallback (in run.py) that logs per-example logits and gold labels at the end of every training epoch. From these training dynamics I computed each example's confidence (the mean probability assigned to the gold label across epochs) and variability (the standard deviation of that probability), following Swayamdipta et al. (2020). The top 33% most variable examples were extracted as the Ambiguous Subset, the most informative training signal.

This subset was combined with Hard Negative Mining. Pairs sharing the same premise with high lexical overlap (Jaccard similarity above 0.5) but different hypotheses were selected, forcing the model to rely on meaning rather than surface word overlap.

## Results

| Model | SNLI (in-domain) | ANLI (OOD robustness) |
|---|---|---|
| Baseline (full SNLI) | 90.1% | 32.6% |
| Hypothesis-only (ablation) | 60.0% | 33.8% |
| Full Contrast (unfiltered hard negatives) | 84.5% | 31.1% |
| Ambiguous Contrast (cartography + hard negatives) | 70.0% | 34.5% |
| Augmented (En-Es back-translation) | 70.1% | 34.9% |
| Multilingual (Es/De/Fr back-translation) | 70.7% | 34.0% |

Filtering to the Ambiguous Subset and mining hard negatives trades in-domain accuracy for out-of-domain robustness (32.6% to 34.5% on ANLI). This is consistent with the hypothesis that the baseline model was relying on shortcuts rather than robust reasoning. Adding single-language back-translation (En-Es) improved robustness further to 34.9%. Extending to three pivot languages (Es, De, Fr) underperformed the single-language version (34.0%). This is likely due to semantic drift accumulating across additional round-trip translations, showing that more augmentation diversity does not always help.

## Attribution

Base training and evaluation code (run.py, helpers.py) is adapted from Prof. Greg Durrett's fp-dataset-artifacts repository (https://github.com/gregdurrett/fp-dataset-artifacts), built for UT Austin's AI 388. My contribution includes the artifact diagnosis (n-gram analysis and hypothesis-only ablation), the CartographyCallback and cartography-based filtering, hard negative mining, single and multi-language back-translation augmentation, and the full comparative evaluation.

## Running It

Train the baseline model and log cartography dynamics for each epoch.

    python3 run.py --do_train --do_eval --task nli --dataset snli --output_dir ./model_baseline/ --num_train_epochs 6

Evaluate on SNLI (in-domain) or ANLI (out-of-domain robustness).

    python3 run.py --do_eval --task nli --dataset snli --model ./model_baseline/ --output_dir ./eval_snli/
    python3 run.py --do_eval --task nli --dataset anli --model ./model_baseline/ --output_dir ./eval_anli/

See requirements.txt for dependencies.
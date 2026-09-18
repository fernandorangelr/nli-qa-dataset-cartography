# Mitigating Dataset Artifacts in NLI via Dataset Cartography

Final project for UT Austin's AI 388 (Natural Language Processing, Fall 2025). This project diagnoses and mitigates dataset artifacts in Natural Language Inference (NLI) by fine-tuning ELECTRA-small on SNLI. The pipeline covers artifact diagnosis, dataset cartography based filtering, hard negative mining, and back-translation augmentation. Models are evaluated on both in-domain accuracy (SNLI) and out-of-domain robustness (ANLI).

## Part I: Diagnosing the Artifact

This section implements two diagnostic tests to confirm the SNLI dataset contains exploitable artifacts.

* N-gram Artifact Analysis: Computes the conditional probability P(label | word) over hypothesis tokens. This identifies words (such as "nobody") that act as near deterministic shortcuts for the label, independent of the premise.
* Hypothesis-Only Ablation: Trains a model with the premise blanked out. This model reached 59.1% accuracy, well above the 33.3% random baseline, quantifying how much the standard model relies on hypothesis-only shortcuts rather than genuine inference.

## Part II: Mitigation via Dataset Cartography

This section implements a custom CartographyCallback (in run.py) that logs per-example logits and gold labels at the end of every training epoch. From these training dynamics I computed each example's confidence (the mean probability assigned to the gold label across epochs) and variability (the standard deviation of that probability), following Swayamdipta et al. (2020). The top 33% most variable examples were extracted as the Ambiguous Subset, the most informative training signal.

This subset was combined with Hard Negative Mining. Pairs sharing the same premise with high lexical overlap (Jaccard similarity above 0.5) but different hypotheses were selected, forcing the model to rely on meaning rather than surface word overlap.

## Results

| Model | Train Size | SNLI (in-domain) | ANLI (OOD robustness) |
|---|---|---|---|
| Baseline (full SNLI) | 549,367 | 89.9% | 32.7% |
| Hypothesis-only (ablation) | 549,367 | 59.1% | 31.8% |
| Full Contrast (unfiltered hard negatives) | 84,146 | 84.1% | 27.8% |
| Ambiguous Contrast (cartography + hard negatives) | 8,700 | 71.1% | 31.5% |
| Augmented (En-Es back-translation) | 16,200 | 69.6% | 33.8% |
| Multilingual (Es/De/Fr back-translation) | 31,200 | 71.3% | 32.8% |

*Note: ANLI Round 1's `test_r1` split is only 1,000 examples, so single-run deltas of a few points between models fall within normal sampling noise.*

Despite reaching 89.9% in-domain accuracy on SNLI, the Standard Baseline scored *below random chance* (33.3%) on ANLI, evidence that it relies on lexical shortcuts (e.g., negation words) rather than genuine entailment reasoning. This holds consistently across every run of this pipeline.

Filtering to the Ambiguous Subset and mining hard negatives (Ambiguous Contrast) trains on 8,700 examples, roughly 63x less data than the baseline, while remaining in the same range as the baseline on ANLI. In this run, filtering alone did not produce a clear robustness gain over the baseline; the difference falls within normal sampling noise. Adding single-language back-translation (Augmented, En-Es) showed a modest improvement over the baseline (32.7% to 33.8%), though this delta is also within that noise range and should not be read as conclusively established without repeated runs. Extending to three pivot languages (Multilingual) landed close to the baseline (32.8%), consistent with the qualitative finding (see below) that additional back-translation diversity introduces semantic drift that can offset any robustness gain from increased lexical diversity.

The clearest, most reproducible finding of this project is the baseline's below-chance ANLI performance despite near-90% in-domain accuracy: a concrete demonstration that high in-domain accuracy can mask reliance on dataset artifacts rather than genuine reasoning.

*This repo also includes `report.pdf`, the original write-up submitted for the course. Its Table 2 reflects the checkpoints and evaluation split used at submission time; the table above reflects the most recent reproducible run of this same pipeline.*

## Attribution

Base training and evaluation code (run.py, helpers.py) is adapted from Prof. Greg Durrett's fp-dataset-artifacts repository (https://github.com/gregdurrett/fp-dataset-artifacts), built for UT Austin's AI 388. My contribution includes the artifact diagnosis (n-gram analysis and hypothesis-only ablation), the CartographyCallback and cartography-based filtering, hard negative mining, single and multi-language back-translation augmentation, and the full comparative evaluation.

## Running It

Train the baseline model and log cartography dynamics for each epoch.

    python3 run.py --do_train --do_eval --task nli --dataset stanfordnlp/snli --output_dir ./model_baseline/ --num_train_epochs 6

Evaluate on SNLI (in-domain) or ANLI (out-of-domain robustness).

    python3 run.py --do_eval --task nli --dataset stanfordnlp/snli --model ./model_baseline/ --output_dir ./eval_snli/
    python3 run.py --do_eval --task nli --dataset facebook/anli --model ./model_baseline/ --output_dir ./eval_anli/

See requirements.txt for dependencies.
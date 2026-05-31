# Leakage-Safe Diagnostic Sentiment Analysis in Noisy Social Media Streams

**Multi-Task Sequence Modelling · Focal-Loss Reasoning · Transformer Benchmarking**

> **Thompson Ikechukwu, Chika Victory Innocent**  
> Centre for Inclusive Development Research and Analytics (CINDRAN)  
> *Correspondence: info@cindran.org*

---

## Overview

This repository is the official implementation for the paper:

> *Leakage-Safe Diagnostic Sentiment Analysis in Noisy Social Media Streams: Multi-Task Sequence Modelling, Focal-Loss Reasoning, and Transformer Benchmarking*

Airline tweet sentiment is a well-studied benchmark — but most published pipelines on this corpus contain a critical methodological flaw: **target leakage**. Negative-reason labels (e.g., "Cancelled Flight", "Customer Service Issue") are appended to the tweet text as input features, meaning the model is partially given the answer it is asked to predict. This inflates reported accuracy to implausible levels (up to 91.94% in prior work) and produces classifiers that cannot function at inference time when reason labels are unavailable.

This project corrects that flaw and reframes the problem as a **leakage-safe diagnostic learning task**:

- Reason labels are **never** used as input features
- A shared encoder is trained with a primary **three-class sentiment head** and a **conditional auxiliary reason head**, whose loss activates only for true negative tweets
- The result is an honest, reproducible benchmark with auditable diagnostic reasoning

---

## Key Contributions

| Contribution | Description |
|---|---|
| **Leakage correction** | Identifies and eliminates target leakage; establishes corrected baselines across six model classes |
| **Multi-task architecture** | Indicator-gated auxiliary reason head — activated only for true negatives, never at inference time |
| **Focal-loss diagnostic variant** | Alpha-balanced focal loss applied to the imbalanced reason head to suppress majority-class dominance |
| **Transformer benchmark** | Fine-tuned DistilBERT-base-uncased evaluated under the same leakage-safe protocol |
| **Computational efficiency analysis** | Measured CPU latency, throughput, and parameter counts across all model classes |
| **Statistical significance** | Paired McNemar test (p = 1.64 × 10⁻¹³) for the primary DistilBERT–M4-FL comparison |

---

## Results

All results are evaluated on a held-out test set of **2,196 tweets** under a strict leakage-safe protocol. Raw cleaned tweet text only — no reason labels appended at any stage.

### Sentiment Classification Performance

| Model | Accuracy | Macro F1 | Weighted F1 | Params |
|---|---|---|---|---|
| M1: TF-IDF + Random Forest | 0.7801 | 0.6992 | 0.7738 | — |
| M2: Word2Vec + LSTM | 0.7659 | 0.7239 | 0.7749 | ~180K |
| M3: Word2Vec + BiLSTM | 0.7577 | 0.7141 | 0.7675 | ~340K |
| M4: Multi-task Conv1D-BiLSTM | 0.7407 | 0.7017 | 0.7529 | ~490K |
| **M4-FL: Focal-loss Conv1D-BiLSTM** | **0.7650** | **0.6955** | **0.7628** | **1.25M** |
| M5: DistilBERT-base-uncased ⭐ | **0.8333** | **0.7820** | **0.8304** | **67.0M** |

> ⭐ DistilBERT is the classification ceiling. M4-FL is the recommended lightweight diagnostic option.

### Computational Efficiency (CPU, Intel Xeon, Colab)

| Model | Parameters | Latency (ms/tweet) | Throughput (tweets/sec) |
|---|---|---|---|
| DistilBERT | 66,955,779 | 85.59 | 11.68 |
| **M4-FL** | **1,251,818** | **0.61** | **1,628** |
| **Relative advantage (M4-FL)** | **53.5× fewer** | **139.4× faster** | **139.4× higher** |

### Auxiliary Diagnostic Reason Head (True Negatives Only, n = 1,377)

The reason head is evaluated only on true negative tweets, where reason labels are defined. This is an investigative component — macro F1 of 0.2904 reflects the difficulty of the 10-class imbalanced diagnostic task.

| Reason Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cancelled Flight | 0.7040 | 0.7154 | 0.7097 | 123 |
| Late Flight | 0.7500 | 0.2869 | 0.4150 | 251 |
| Customer Service Issue | 0.7235 | 0.3489 | 0.4708 | 450 |
| Lost Luggage | 0.5522 | 0.3978 | 0.4625 | 93 |
| Can't Tell | 0.2610 | 0.6012 | 0.3640 | 168 |
| *Other classes* | *low* | *low* | *low* | *<100* |
| **Macro average** | **0.3287** | **0.3130** | **0.2904** | **1,377** |

---

## Architecture

```
Tweet text (raw, cleaned)
        │
   [Embedding Layer]          ← 100-d trainable embeddings (vocab = 10,303)
        │
   [Conv1D, 64 filters, k=3]  ← local n-gram feature extraction
        │
   [BiLSTM, h=128/direction]  ← 256-d bidirectional context
        │
   [Shared representation z ∈ ℝ²⁵⁶]
        │              │
[Sentiment Head]  [Reason Head]         ← reason head activated only if y_true = NEGATIVE
   3-class softmax   10-class softmax   ← NEVER used as input; auxiliary supervision only
```

**Multi-task objective:**

```
L_total = L_sentiment + β · 𝟙[y_true = negative] · L_focal_reason

L_focal_reason = −αt (1 − pt)^γ log(pt)     [γ=1.0, β=0.25, selected by val macro F1]
```

---

## Repository Structure

```
Airline-Tweet-Sentiment-Analysis/
│
├── DISTILBERT.ipynb              # Main executable notebook (all experiments)
├── README.md                     # This file
│
├── data/
│   └── README.md                 # Dataset download instructions (CrowdFlower/Kaggle)
│
├── figures/
│   ├── architecture_diagram.png  # Fig 1 — leakage-safe multi-task architecture
│   ├── distilbert_cm.png         # Fig 2 — DistilBERT confusion matrix
│   ├── m4fl_cm.png               # Fig 3 — M4-FL confusion matrix
│   ├── performance_comparison.png# Fig 4 — macro F1 comparison bar chart
│   └── efficiency_frontier.png   # Fig 5 — accuracy vs latency scatter plot
│
└── paper/
    └── Leakage_Safe_Sentiment_arXiv_FINAL.docx   # Companion manuscript
```

---

## Quickstart

### Prerequisites

```bash
pip install torch>=2.10.0 transformers>=5.0.0 numpy pandas scikit-learn \
            seaborn matplotlib statsmodels
```

### Running the notebook

The entire pipeline — data loading, all six model trainings, evaluations, efficiency profiling, and statistical tests — is contained in a single executable notebook:

```bash
# Option A: Run in Google Colab (recommended — GPU access)
# Upload DISTILBERT.ipynb and run all cells in order

# Option B: Run locally
jupyter notebook DISTILBERT.ipynb
```

> **Important:** A CUDA-enabled GPU is required for DistilBERT fine-tuning in reasonable time (~15 minutes on Tesla T4). All other models train on CPU in under 5 minutes.

### Dataset

The Twitter US Airline Sentiment dataset is publicly available on Kaggle:

```
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
```

Download `Tweets.csv` and place it in the `data/` directory before running the notebook. The notebook handles all preprocessing internally.

---

## Reproducing the Paper Results

The notebook is structured to reproduce every result table and figure in the manuscript:

| Notebook section | Manuscript table/figure |
|---|---|
| §3 — DistilBERT fine-tuning | Table V, Fig. 2 |
| §6 — Focal-loss sweep | Table VI |
| §7 — M4-FL held-out evaluation | Tables VII, VIII, Fig. 3 |
| §8 — Comparative analysis | Tables IV-A, IV-B, Fig. 4 |
| §9 — Computational profiling | Table IX, Fig. 5 |
| §11.2 — McNemar test | Section IV-F |

All random seeds are fixed (`random.seed(42)`, `torch.manual_seed(42)`, `np.random.seed(42)`). Results may vary by ±0.2% across hardware/framework versions.

---

## Known Limitations

This codebase is a research implementation. Three limitations are explicitly acknowledged in the paper:

1. **Ablation confound.** M4-FL differs from M4 in hidden size, filter count, batch size, and sequence length — not only the loss function. The focal-loss effect cannot be isolated from the capacity increase. A capacity-matched comparison is the highest-priority future experiment.

2. **Single-split results.** All metrics rest on a single 70/15/15 stratified split. The M2–M3 inversion (LSTM outperforming BiLSTM) is likely a single-seed artefact. A five-seed extension with ±std and pairwise significance tests is planned.

3. **Reason head maturity.** Macro F1 of 0.2904 on the diagnostic reason head means this component is **investigative only** — not production-ready. Customer Service Issue recall (0.3489) is particularly weak due to focal-loss majority-class suppression.

---

## Citation

If you use this code or the corrected leakage-safe baselines in your work, please cite:

```bibtex
@article{ikechukwu2025leakage,
  title     = {Leakage-Safe Diagnostic Sentiment Analysis in Noisy Social Media Streams:
               Multi-Task Sequence Modelling, Focal-Loss Reasoning, and Transformer Benchmarking},
  author    = {Ikechukwu, Thompson and Innocent, Chika},
  year      = {2026},
}
```

---

## License

This project is released under the [MIT License](LICENSE). The Twitter US Airline Sentiment dataset is subject to its original CrowdFlower/Kaggle terms of use.

---

## Acknowledgements

This work was conducted under the auspices of the **Centre for Inclusive Development Research and Analytics (CINDRAN)**. The authors acknowledge the CrowdFlower community for the original dataset annotation, and the HuggingFace team for the `transformers` library.

# Genomic-Language-Models-Vs-Classical-Approaches-Promoter-Classification-Ecoli-Sequence-Length-Aware

A sequence-length-aware benchmark comparing pretrained genomic language models against classical machine learning baselines for binary promoter classification in *Escherichia coli* K-12.

> **Paper:** *Genomic Language Models vs. Classical Approaches for Promoter Classification in E. coli: A Sequence-Length-Aware Benchmark*
> Rebeka Maneva, Konstantin Lozhankoski
> Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University, Skopje, N. Macedonia
> *(paper in progress)*

---

## Overview

This study benchmarks five models for binary promoter classification across three sequence window sizes (100 bp, 200 bp, 500 bp):

| Model | Type |
|---|---|
| k-mer + TF-IDF + Logistic Regression | Classical baseline |
| 1D CNN | Neural baseline |
| DNABERT-2 | Pretrained genomic language model |
| HyenaDNA | Pretrained long-range genomic model |
| Nucleotide Transformer v2 | Pretrained genomic language model |

The main finding is that no single architecture dominates across all sequence lengths — performance depends on the interaction between model design and sequence context. The CNN baseline achieves the highest AUC at 100 bp (0.899), while HyenaDNA performs best at 500 bp (F1 = 0.801), and DNABERT-2 leads among pretrained models at 100 bp (F1 = 0.827).

---

## Data

Positive examples are derived from 2,122 experimentally validated transcription start sites (TSSs) from [RegulonDB](https://regulondb.ccg.unam.mx/), covering the complete *E. coli* K-12 genome. Negative examples are sampled from genomic regions with no TSS within a minimum distance of 300 bp (100/200 bp windows) or 500 bp (500 bp window).

The genome FASTA and raw RegulonDB TSS files can be downloaded directly from:

- **Genome:** [NCBI — E. coli K-12 MG1655 (NC_000913.3)](https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3)
- **TSSs:** [RegulonDB — Transcription Start Sites](https://regulondb.ccg.unam.mx/)

The processed CSV datasets (`promoter_binary_*.csv` and `dnabert2_*/`) are included.

---


## Attribution

The DNABERT-2 training script (`code/train_dnabert2.py`) is sourced from the official DNABERT-2 repository:

> Zhihan1996/DNABERT_2 — https://github.com/Zhihan1996/DNABERT_2
> Zhou et al., ICLR 2024

All credit for that script goes to the original authors.

---
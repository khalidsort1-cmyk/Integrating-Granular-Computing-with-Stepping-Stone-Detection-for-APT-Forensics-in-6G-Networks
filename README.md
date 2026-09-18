# Granular Computing for Multi-Stage APT Detection in 6G Networks
### A Difficulty-Aware Forensics Evaluation Framework

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE TIFS](https://img.shields.io/badge/Journal-IEEE%20TIFS-blueviolet)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10206)

This repository contains the reference implementation for the paper:

> **Granular Computing for Multi-Stage APT Detection in 6G Networks: A Difficulty-Aware Forensics Evaluation Framework**
> Khalid Abdulrazzaq Abdulnabi Alminshd, Pedram Salehpour, Jaber Karimpour, and Mohd Nizam Omar
> *IEEE Transactions on Information Forensics and Security* (submitted)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Datasets](#datasets)
6. [Quick Start](#quick-start)
7. [Forensic Benchmark Audit](#forensic-benchmark-audit)
8. [Synthetic 6G APT Dataset Generator](#synthetic-6g-apt-dataset-generator)
9. [Reproducing Paper Results](#reproducing-paper-results)
10. [Classifiers](#classifiers)
11. [Citation](#citation)
12. [License](#license)
13. [Contact](#contact)

---

## Overview

Advanced Persistent Threats (APTs) in 6G networks exhibit multi-stage, stealthy lifecycles
that defeat conventional signature-based intrusion detection. This work presents a unified,
**difficulty-aware forensic evaluation** of **13 classifiers** — **8 SSD-augmented granular
methods, 3 ML baselines, and 2 DL baselines** — across **5 real-world NIDS datasets** and a
**controlled synthetic 6G APT benchmark** at three difficulty levels.

Two pinned commits correspond to the two coordinated components of this artefact:

| Component | Commit | Lines | Purpose |
|---|---|---|---|
| **Full pipeline** | [`dee5786`](https://github.com/khalidsort1-cmyk/Granular-Computing-for-MultiStage-APT-Detection-in-6G-Networks-/commit/dee5786121b423c9029bede18c52f031c032141d) | 1729 | End-to-end experimental framework |
| **Standalone generator** | [`5c70a1b`](https://github.com/khalidsort1-cmyk/Granular-Computing-for-MultiStage-APT-Detection-in-6G-Networks-/commit/5c70a1b85e3b01c752dfc4e53c94ad0570b2f783) | 257 | Synthetic 6G APT generator (self-contained) |

---

## Key Contributions

1. **Forensic Benchmark Audit** — A single-feature decision-stump diagnostic that flags
   label-leaking features (`F1 ≥ 0.90`) and reports duplicate-row rates. Applied to five
   widely used NIDS datasets, it reveals that **three of five are forensically compromised**:
   KDD Cup 99 (62.2% duplicate rows, 7 leaks), MSCAD (37 leaks), and NIDS-Bench 2026
   (high-dimensional joint separability). Only **UNSW-NB15** and **NSL-KDD** remain
   discriminative.

2. **Unified SSD + Granular Framework** — Eight granular classifiers on SSD-augmented
   feature spaces, augmented by three ML baselines (Random Forest, XGBoost, SVM) and
   two DL baselines (MLP, 1D-CNN) under a common scikit-learn interface.

3. **Difficulty-Controlled Adversarial Benchmark** — The first synthetic 6G APT generator
   parameterised by **attack mimicry rate**, **distributional spread**, and **stage
   separation**, enabling reproducible robustness measurement at three difficulty levels
   (VERY HARD / HARD / EASY).

4. **Deployment-Oriented Pareto Analysis** — Granular methods dominate the
   accuracy–efficiency frontier: **SSD + Rough Set** and **SSD + Neighborhood Systems**
   train **14–26× faster** than tuned XGBoost at 0.1–2.2 F1-point cost, extending to
   **295×** on the synthetic benchmark.

---

## Repository Structure

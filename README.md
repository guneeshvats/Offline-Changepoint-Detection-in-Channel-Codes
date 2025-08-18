# Offline Changepoint Detection in Channel Codes

This repository tracks the progress of my research project on **offline changepoint detection in channel codes**. It consolidates literature, code experiments, results, slides, and submission targets.

---

## Table of Contents

* [Overview](#overview)
* [Literature Review](#literature-review)
* [Code Progress (Processed & Unprocessed Data)](#code-progress-processed--unprocessed-data)
  * [Processed Data: Bernoulli Data](#processed-data-bernoulli-data)
  * [Unprocessed Data: Neural Networks](#unprocessed-data-neural-networks)
    * [Type I (Distinct Neural Networks), Approach I](#type-i-distinct-neural-networks-approach-i)
    * [Type II (Diverse Neural Networks), Approach I](#type-ii-diverse-neural-networks-approach-i)
  * [Comparative Plots](#comparative-plots)
* [Target Conferences/Journals](#target-conferencesjournals)
* [Presentations / Progress](#presentations--progress)
  * [Results](#results)
* [Research Paper Drafts](#research-paper-drafts)
* [Code Progress (Components Summary)](#code-progress-components-summary)
* [Extra Important Resources and Links](#extra-important-resources-and-links)
* [Project Concept Sketches](#project-concept-sketches)

---

## Overview

Offline changepoint detection is critical in identifying shifts in data behavior, particularly within channel codes. This repository serves as a record of the research process, tracking progress through literature, code, and experimental results.

---

## Literature Review

**Master tracking sheet (continuously updated):** [Link](https://docs.google.com/spreadsheets/d/1NH3iFS4BFo1hTYis5-IdWirP7EWbnc5gohk_1H-MvWg/edit?gid=0#gid=0)

Below is a curated (partial) list of key papers and books. The full list is maintained in the sheet above.

| **S.No** | **Title**                                                                                                                   | **Link**                                                                                                            |
| -------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1        | **Paper - 1:** Selective Review of Offline Changepoint Detection Methods                                                    | [View Paper](https://www.sciencedirect.com/science/article/pii/S0165168419303494)                                   |
| 2        | **Paper - 2:** Automatic Change-Point Detection in Time Series via Deep Learning (Arthi)                                    | [View Paper](https://arxiv.org/abs/2211.03860)                                                                      |
| 3        | **Paper - 3:** Inference About the Change-Point in a Sequence of Binomial Variables (1970)                                  | [View Paper](https://www.jstor.org/stable/2334766)                                                                  |
| 4        | **Paper - 4:** A Test for a Change in a Parameter Occurring at an Unknown Point (1955)                                      | [View Paper](https://academic.oup.com/biomet/article-abstract/42/3-4/523/296358)                                    |
| 5        | **Paper - 5:** Nonparametric Changepoint Analysis for Bernoulli Random Variables Based on Neural Networks (Thesis)          | [View Paper](https://kluedo.ub.rptu.de/frontdoor/deliver/index/docId/2032/file/Final_Draft_October_14102008.pdf)    |
| 6        | **Paper - 6:** A Simple cumulative sum type statistic for the change-point problem with zero-one observations (Arthi, 1980) | [View Paper](https://www.jstor.org/stable/2335319)                                                                  |
| 7        | **Book - 1:** Detection of Abrupt Changes: Theory and Application                                                           | [Chapter 2, 3.1](https://people.irisa.fr/Michele.Basseville/kniga/kniga.pdf)                                        |
| 8        | **Book - 2:** Error Control Coding: Fundamentals and Applications                                                           | [View Book](https://pg024ec.wordpress.com/wp-content/uploads/2013/09/error-control-coding-by-shu-lin.pdf)           |
| 9        | **Book - 3:** Fundamentals of Statistical Signal Processing                                                                 | [View Book]()                                                                                                       |
| 10       | **Paper - 7:** Bayesian Approach to Inference About a Changepoint in a Sequence of Random Variables (1975)                  | [View Paper](https://www.jstor.org/stable/2335381?refreqid=fastly-default%3A81fef0d7415e79e63875176c864c8f65&seq=2) |
| 11       | **Book - 4:** An Introduction to Signal Detection and Estimation, Vincent Poor                                              | [View Book]()                                                                                                       |

---

## Code Progress (Processed & Unprocessed Data)

This section tracks the coding and experimentation journey for this project. It documents each method tried, comparative insights, and rationale for selecting or discarding algorithms.

### Processed Data: Bernoulli Data

* **When Bernoulli parameters are known**

  1. MLE
  2. CUSUM

* **When parameters are not known**

  1. BinSeg + Bernoulli CF
  2. PELT + Bernoulli CF

**For PMF1–PMF2:** *(On Hold for now)*

### Unprocessed Data: Neural Networks

#### Type I (Distinct Neural Networks), Approach I

```
Type I: Neural networks are trained on sequences with a single value of p (BSC parameter).
Approach I: 1D CNN with Residual Block and Adaptive Pooling
Approach II: (Only for n = 63 BCH Codes)
```

1. **T = 5**, p ∈ {0.01, 0.05, 0.1, 0.2}, Codeword Length = 15: [Link](https://www.kaggle.com/code/guneeshvats/cl-15-t-5-approach-1)
2. **T = 7**, p ∈ {0.01, 0.05, 0.1, 0.2}, Codeword Length = 15: [Link](https://www.kaggle.com/code/guneeshvats/cl-15-t-7-approach-1)
3. **T = 10**, p ∈ {0.01, 0.05, 0.1, 0.2}, Codeword Length = 15: [Link](https://www.kaggle.com/code/guneeshvats/cl-15-t-10-approach-1)
4. **T = 5**, p ∈ {0.001, 0.005, 0.01, 0.05}, Codeword Length = 31: [Link](https://www.kaggle.com/code/guneeshvats/cl-31-t-5-approach-1)
5. **T = 7**, p ∈ {0.001, 0.005, 0.01, 0.05}, Codeword Length = 31: [Link](https://www.kaggle.com/code/guneeshvats/cl-31-t-7-approach-1)
6. **T = 10**, p ∈ {0.001, 0.005, 0.01, 0.05}, Codeword Length = 31: [Link](https://www.kaggle.com/code/guneeshvats/cl-31-t-10-approach-1)
7. **T = 5**, p ∈ {0.00001, 0.005, 0.01, 0.05}, Codeword Length = 63: [Link](https://www.kaggle.com/code/guneeshvats/cl-63-t-5-approach-1)
8. **T = 7**, p ∈ {0.00001, 0.005, 0.01, 0.05}, Codeword Length = 63:
9. **T = 10**, p ∈ {0.00001, 0.005, 0.01, 0.05}, Codeword Length = 63:

#### Type II (Diverse Neural Networks), Approach I

```
Type II: Model is trained on sequences of a fixed length but with multiple BSC parameters p ∈ {0.001, 0.005, 0.01, 0.05, 0.1} (and intermediate values below).
Approach I: Still figuring out.
Purpose: Handle the case when p is unknown at inference time.
Approach III: CNN & transformer hybrid model
```

1. **T = 5**, p ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05}, Codeword Length = 15: [Link](https://www.kaggle.com/code/guneeshvats/type2-cl-15-t-5)
2. **T = 7**, p ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05}, Codeword Length = 15:
3. **T = 10**, p ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05}, Codeword Length = 15:
4. **T = 5**, p ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05}, Codeword Length = 30:
5. **T = 7**, p ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05}, Codeword Length = 30:
6. **T = 10**, p ∈ {0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05}, Codeword Length = 30:

### Comparative Plots

**Neural Networks (Type I & II):**

1. Accuracy vs N'
2. Accuracy vs N'
3. Accuracy vs p
4. Accuracy vs T

**Baselines vs Neural Networks:**

1. Known p (Type I NN) — comparison with MLE & CUSUM: Accuracy vs {fixed T/p} (whichever is not fixed forms the x-axis)
2. Unknown p (Type II NN) — comparison with BinSeg & PELT: Accuracy vs {fixed T/p}

---

## Target Conferences/Journals

1. ITW — [Link]() — April
2. Globecom — [Link]() — April
3. WCNC — [Link](https://wcnc2026.ieee-wcnc.org/group/21) — **15 September 2025**
4. IEEE ICC — [Link](https://icc2026.ieee-icc.org/) — **29 September 2025 (week)**
5. IEEE Communications Letters (Journal) — [Link](https://www.comsoc.org/publications/journals/ieee-comml/ieee-communications-letters-submit-manuscript)
6. NCC — December
7. COMSNETS — **15 October 2025** — [Link](https://www.comsnets.org/)

---

## Presentations / Progress

1. **Selective Review of Offline Changepoint Detection Methods and Derivation of Pd** — [Google Slides Draft](https://docs.google.com/presentation/d/1yzx00AFN8aDG7L4OdEDbvaQSgfRj37CbkmYR_34oxAI/edit#slide=id.p)
2. **Book:** Detection of Abrupt Changes — Part 1 — [Link](https://docs.google.com/presentation/d/1PnksHSrUnm4IxZZjZRDIiH2pTVHfskBcWiLQSv_T2x0/edit?usp=sharing)
3. **Book:** Detection of Abrupt Changes — Part 2 — [Link](https://docs.google.com/presentation/d/1iXaYZVFk-exzLrhFkULLsp9rYov2lKB-wqtYYZgmQAg/edit?slide=id.g365fba041ae_0_227#slide=id.g365fba041ae_0_227)
4. **Fundamentals Presentation (Steven M. Kay), Chapter 3:** *Fundamentals of Statistical Signal Processing, Vol. II — Detection Theory* — [Google Slides](https://docs.google.com/presentation/d/1lgZ_AjC37yOn1BTG8N3TQjbQZ_DMGhDRBksqX38kGpw/edit?slide=id.p#slide=id.p)
5. **David Hinkley’s Paper (1970)** — [Link](https://docs.google.com/presentation/d/10IOkuVqAIMPgPNODWeXjyfC6MruyAOy3SdZtoe14s8A/edit?slide=id.p#slide=id.p)
6. **Bayesian Approach to Inference about a Changepoint** — [Link](https://docs.google.com/presentation/d/1kav9KW3tlR1cqhAe5ZjEAPWfs7MjHPLMCOK41FYLdz4/edit?slide=id.p#slide=id.p)

### Results

1. **Preliminary Results — Processed Data** (Ruptures): [Link](https://docs.google.com/presentation/d/1wyRNPNR1VTmX5hlMlGAFHBnWoSIPuzE6qKK-2E9-rkI/edit?usp=sharing)
2. **Preliminary Results — Unprocessed Data** — Approach 1 (1D CNN with residual block and adaptive pooling): [Link](https://docs.google.com/presentation/d/1KcciWTHpWIijZlj-yCmJKgQ4REtyWrCr6Z5-QIe8WD8/edit?usp=sharing)
3. **Implementation of Hinkley’s Paper** — Code and results are in the repository folder: `Hunkley Implementation` (root directory).

---

## Research Paper Drafts

1. Draft Version 0 — [Link](https://www.overleaf.com/project/68833e7c90f485946f428a2a)
2. Other similar work's Draft — [Link](https://www.overleaf.com/project/67d121a950d4fef4b0e5c5ab)

---

## Code Progress (Components Summary)

1. **Data Generation Code** (MATLAB code) - BCH, LDPC
2. **Unprocessed Codewords Data:** Neural Networks (1D CNN), Transformers
3. **Processed Data:** Ruptures and Python libraries
4. **Implementation of Hinkley’s Recursive Formulation**

---

## Extra Important Resources and Links

1. **CPD Project Related Topics and Research Papers (to be done):** [Link](https://docs.google.com/spreadsheets/d/1NDgSKFA3LEqDTNt4PL07NPiguZQqH2xJB_vK9ANAU-A/edit?gid=0#gid=0)

---

## Project Concept Sketches

**Idea of the Project:**

<!-- You can adjust width/height as needed (only width shown below for responsiveness). -->

<p>
  <img src="images/page_1.jpeg" alt="Idea_1" width="500" />
</p>
<p>
  <img src="images/page_2.jpeg" alt="Idea_2" width="500" />
</p>

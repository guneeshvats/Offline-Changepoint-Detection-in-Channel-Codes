# Offline Changepoint Detection in Channel Codes

This repository contains the full implementation for **offline changepoint detection (CPD)** in **binary channel-coded communication streams** (BCH / LDPC) over BSC and AWGN channels.  
The core idea is a **two-stage pipeline**:

1. **Statistical Coarse MLE Stage**  
   Project each received codeword using a carefully selected vector \( h \in C_2^\perp \setminus C_1^\perp \).  
   This produces a Bernoulli sequence \( z_i = y_i \cdot h^T \).  
   A Hinkley-style Bernoulli MLE is used to get a coarse estimate \( \tilde{\tau} \) and a confidence window.

2. **Neural Refinement Stage**  
   A small 1D CNN scans raw codeword windows around the coarse region and refines the CP location.

This repository includes:
- MATLAB generation of channel-corrupted codewords  
- Python implementation of statistical CPD (MLE, CUSUM, PELT, BinSeg)  
- Deep learning notebooks for model training (known-p and unknown-p)  
- Final unified Python pipelines for **known-p** and **unknown-p** settings  

---

## Installation

```bash
git clone https://github.com/guneeshvats/Offline-Changepoint-Detection-in-Channel-Codes.git
cd Offline-Changepoint-Detection-in-Channel-Codes
pip install -r requirements.txt
```

## Directory Structure 
```
Offline-Changepoint-Detection-in-Channel-Codes/
│
├── Data_Generation/                         # MATLAB scripts for creating simulated codeword pools
│   ├── BCH+AWGN_Channel/                    # BCH over AWGN (generates CSV pools)
│   ├── BCH+BSC_Channel/                     # BCH over BSC (generates CSV pools)
│   ├── LDPC+AWGN_Channel/                   # LDPC over AWGN
│   └── LDPC+BSC_Channel/                    # LDPC over BSC
│
├── Deep_Learning_Approach/                  # Jupyter notebooks for training & evaluation
│   ├── BCH+BSC_known_p/                     # Known–p models
│   │   ├── n=15/                            # Contains T={5,7,10} trained models + experiments
│   │   └── n=31/                            # Contains T={5,7,10} trained models + experiments
│   └── BCH+BSC_unknown_p/                   # Unknown–p Type-II models (trained on multi-p data)
│       ├── n=15/
│       └── n=31/
│
├── Hinkley_Implementation/                  # Pure statistical CPD (Bernoulli MLE + πₙ recursion)
│   ├── code.py                              # Implements MLE, recursive πₙ, CUSUM, etc.
│   └── README.md
│
├── Processed_Data/
│   ├── Bernoulli_Data/                      # Ruptures baselines: BinSeg, PELT, CUSUM, MLE
│   └── PMF1_PMF2/                           # Probability mass visualizations and utilities
│
├── final_algorithm_known_p.py               # ***Main unified algorithm (known BSC-p)***
├── final_algorithm_unknown_p.py             # Unified pipeline for unknown-p inference
│
├── Find_min_weight_h_BCH_n15_n31.m          # MATLAB search for low-weight h (BCH)
├── Find_min_weight_h_LDPC_n648.m            # MATLAB search for low-weight h (LDPC)
│
├── requirements.txt
└── README.md
```

## Quick Guide

1. To run the final algorithm → use : final_algorithm_known_p.py or final_algorithm_unknown_p.py
2. To train CNN models → open notebooks under Deep_Learning_Approach
3. To regenerate data → run scripts under Data_Generation
4. To study the math → read Hinkley_Implementation/code.py



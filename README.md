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


### Running the Final Algorithm (Known-p)

The script final_algorithm_known_p.py implements the entire pipeline:

1. Load C1/C2 channel-corrupted codeword pools
2. Build a sequence of length M
3. Project using h → Bernoulli sequence z
4. Run coarse MLE + compute window using πₙ
5. Run a TorchScript CNN for refinement
6. Produce final estimate τ^


### 1. Configure Inputs (Inside the Script)

Open final_algorithm_known_p.py and edit the block:
```
# ========================= USER INPUTS ===========================

# 1) Channel codeword pools (CSV files with rows of 0/1)
CSV_C1 = "/path/to/bsc_p0.005_C1_n15_k11.csv"
CSV_C2 = "/path/to/bsc_p0.005_C2_n15_k7.csv"

# 2) Global sequence length and true CP (for synthetic experiments)
M = 1000
TAU_TRUE = 500

# 3) Binary test vector h (length n)
H = [1,1,0,0,0,0,0,0,0,1,0,0,0,1,0]   # Example for n=15

# 4) BSC parameters
p = 0.005
SWAP_THETA12 = True

# 5) Confidence level for selecting π_n window
alpha = 0.90

# 6) Neural model (TorchScript)
MODEL_PATH = "/path/to/models/n15_T5_p0005.pt"
WINDOW_T  = 5
STRIDE    = 1
DEVICE    = "cpu"

# 7) Random seed
SEED = 12345

# 8) π_n recursion hyperparameters
PIN_NGRID    = 20
PIN_NDISPLAY = 20
VIEW_HALF    = 10

# 9) Monte-Carlo experiments (optional)
MC_ENABLE = False
MC_NR = 3000
```
### 2. Run the Script

```bash
python final_algorithm_known_p.py
```

### Example output : 

```
================= RESULT =================
True changepoint τ (count) : 500
Coarse MLE τ~ (on z_i)    : 492
α-window S (even), half   : S=20, halfwidth=10
Refined NN τ̂ (count)     : 498
==========================================
[saved] result_final_algorithm.csv
```

### Output Files

`result_final_algorithm.csv` contains:

1. All input parameters
2. θ₁, θ₂, coarse estimate
3. Window width
4. Crop indices
5. Final estimate
6. πₙ distribution values


### Unknown-p Pipeline

```bash
python final_algorithm_unknown_p.py
```
This uses a Type-II CNN model trained across multiple p values.
No BSC parameter p is needed at inference.


## Hinkley Statistical Baselines

The folder: `Hinkley_Implementation/`
contains:
code.py → exact MLE recursion, πₙ computation, CUSUM, and baseline methods
README.md → notes explaining the maths
Useful for benchmarking MLE-only CPD vs. neural refinement.




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

# Installation

```bash
git clone https://github.com/guneeshvats/Offline-Changepoint-Detection-in-Channel-Codes.git
cd Offline-Changepoint-Detection-in-Channel-Codes
pip install -r requirements.txt


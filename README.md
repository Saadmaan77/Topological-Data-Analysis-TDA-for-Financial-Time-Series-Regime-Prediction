# Topological-Data-Analysis-TDA-for-Financial-Time-Series-Regime-Prediction

Topological Data Analysis (TDA) for Financial Time Series & Regime Prediction
A hybrid quantitative machine learning pipeline combining Topological Data Analysis (TDA) and Deep Sequence Modeling (LSTM) to predict market direction and detect structural volatility regime shifts on S&P 500 index ETF (SPY) data.
📌 Overview
Standard time-series models often struggle to capture non-linear attractor dynamics and geometric phase space collapses preceding market crashes. This repository implements:
1.	Takens' Phase Space Reconstruction using Average Mutual Information (AMI) and False Nearest Neighbors (FNN).
2.	Persistent Homology Feature Extraction (H0 and H1 persistence entropy and fixed-resolution persistence images).
3.	Multi-Scale Topological Feature Engineering across dynamic lookback windows.
4.	Gated Hybrid Neural Network Architecture featuring a learned data-dependent gating mechanism between sequential features and topological descriptors.
5.	Realistic Financial Backtesting Engine with transaction cost modeling and sensitivity stress testing.
🏗️ Architecture Pipeline
Raw SPY OHLCV Data (2005–2024)
       │
       ├──► Feature Preprocessing (RSI, Log Returns, Realized Vol, MACD, Volume Delta)
       │         │
       │         └──► Sliding Window Normalization (60-day, Local Z-Score)
       │                   │
       │                   ├──► [Branch 1] Sequential Stream (60 x 5) ──► PyTorch LSTM
       │                   │                                                      │
       └──► Takens Delay Embedding (τ=3, d=10)                                    │
                 │                                                                ▼
                 ├──► Point Clouds ──► Ripser (H₀, H₁) ──► Vectorization ──► Learned Gate Fusion
                 │    (Multi-Scale: 30, 45, 60 days)       (PI + Entropy)         │
                 │                                                                ▼
                 └──► Early Crash Warning Diagnostics (H₁ Entropy)         Prediction & Backtest
🔬 Mathematical & Topological Foundations
1. Phase Space Reconstruction (Takens' Theorem)
For a 1D log-return series s(t), the delay coordinate map reconstructs the underlying attractor:
v(t)=[s(t),s(t−τ),s(t−2τ),…,s(t−(d−1)τ)]∈Rd
•	Delay (τ): Selected at the first local minimum of Average Mutual Information (AMI):
AMI(τ)=x,y∑P(x,y;τ)log2P(x)P(y)P(x,y;τ)
•	Embedding Dimension (d): Optimized via False Nearest Neighbors (FNN) thresholding until false neighbor percentage drops below 1%.
2. Topological Feature Vectorization
•	Persistence Entropy: Measures geometric disorder across filtration intervals (bi,di):
E(D)=−i=1∑npilog(pi),pi=∑j(dj−bj)di−bi
•	Persistence Images: Transforms discrete persistence diagrams into fixed 10×10 Gaussian-smoothed surface representations via linear scaling and 2D bilinear interpolation.
3. Gated Hybrid Fusion Mechanism
Rather than simple concatenation, the model adaptively weights sequential and topological embeddings:
hraw=Wrhlstm,htda=Wthmlp
g=σ(Wg[hraw∥htda]+bg)
z=g⊙hraw+(1−g)⊙htda
y^=σ(MLP(z))
📂 Repository Structure
├── data/
│   └── preprocessed_data.csv          # Preprocessed SPY indicators (cached)
├── models/
│   ├── baseline_lstm.py               # Pure LSTM baseline network
│   └── gated_hybrid.py                # Gated LSTM + Multi-Scale TDA network
├── src/
│   ├── data_loader.py                 # yfinance ingestion & sliding window builder
│   ├── indicators.py                  # Technical indicator computation
│   ├── takens.py                      # AMI & FNN phase space reconstruction
│   ├── tda_features.py                # Ripser interface, persistence entropy & images
│   └── backtest.py                    # Vectorized backtesting & performance metrics
├── notebooks/
│   └── tda_financial_modeling.ipynb   # Complete interactive research notebook
├── requirements.txt                   # Python dependencies
└── README.md
⚡ Installation & Setup
Bash
# Clone the repository
git clone https://github.com/your-username/tda-financial-forecasting.git
cd tda-financial-forecasting

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
requirements.txt
Plaintext
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
torch>=2.0.0
yfinance>=0.2.20
ripser>=0.6.4
persim>=0.3.1
🚀 Quickstart
Run the Full Pipeline
Python
import yfinance as yf
from src.data_loader import generate_windows
from src.takens import find_optimal_tau, false_nearest_neighbors
from src.tda_features import compute_multiscale_tda_features
from models.gated_hybrid import GatedHybridModel

# 1. Ingest Data
df = yf.download("SPY", start="2005-01-01", end="2024-12-31", auto_adjust=True)

# 2. Phase Space Parameters
tau, _ = find_optimal_tau(df['log_return'].dropna().values[:3500])
d, _ = false_nearest_neighbors(df['log_return'].dropna().values[:3500], tau=tau)

# 3. Compute Multi-Scale TDA & Train Gated Model
# (Refer to notebooks/tda_financial_modeling.ipynb for complete workflow)
📊 Backtest & Empirical Results
Evaluated out-of-sample on unseen SPY test partitions (15% split hold-out, 1 bps friction):
Strategy	Accuracy	Precision	Recall	F1-Score	Sharpe	Sortino	Max Drawdown	Cumulative Return
Baseline LSTM	49.80%	0.5329	0.4315	0.4769	0.1736	0.1595	-23.41%	+6.64%
Gated Hybrid (LSTM + TDA)	49.93%	0.5217	0.6726	0.5876	0.2456	0.2839	-23.60%	+11.61%
🔍 Transaction Cost Sensitivity (Gated Hybrid)
Cost (bps)	Cumulative Return	Sharpe Ratio	Max Drawdown	Trade Count
0 bps	12.97%	0.2727	-23.42%	121
1 bps	11.61%	0.2456	-23.60%	121
5 bps	6.34%	0.1373	-24.69%	121
10 bps	0.10%	0.0021	-26.33%	121
20 bps	-11.31%	-0.2673	-29.50%	121
📉 Topological Crash Early Warning Signal
Tracking H1 persistence entropy over sliding windows reveals structural loop dissolution prior to major market liquidations:
•	2008 Global Financial Crisis: Sharp spikes in 1D topological entropy preceded the Lehman collapse and major equity drawdowns.
•	2020 COVID-19 Flash Crash: Abrupt topological phase shifts captured the compression of market state space weeks prior to peak equity decline.


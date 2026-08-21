# Topological Data Analysis for Financial Time-Series Prediction

A research project investigating whether **Topological Data Analysis (TDA)** can improve short-term financial direction prediction when combined with recurrent neural networks.

The project develops and evaluates two generations of hybrid models:

* **Version 1:** LSTM + TDA Hybrid Model
* **Version 2:** Improved Gated LSTM + TDA Hybrid Model

The experiments use historical **SPY (SPDR S&P 500 ETF)** market data and combine conventional financial indicators with topological features extracted from Takens-embedded financial time series.

---

## 📌 Project Overview

Financial time series contain nonlinear and complex temporal structures that may not be fully represented by conventional technical indicators.

This project explores whether **topological features** can provide additional information about the underlying structure of financial market dynamics.

The overall pipeline is:

```text
SPY Historical Data
        │
        ▼
Technical Indicators
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
60-Day Financial         Log-Return Sequence
Feature Windows                │
        │                      ▼
        │              Takens Phase Space
        │                  Embedding
        │                      │
        │                      ▼
        │               Point Clouds
        │                      │
        │                      ▼
        │             Persistent Homology
        │                      │
        │                      ▼
        │          H0 / H1 Persistence
        │                      │
        │                      ▼
        │       Entropy + Persistence Images
        │                      │
        │                      ▼
        │                TDA Features
        │                      │
        └──────────┬───────────┘
                   ▼
              Hybrid Model
                   │
                   ▼
       Next-Day Direction Prediction
                   │
                   ▼
          Financial Backtesting
```

---

# 🧪 Dataset

The experiments use historical market data for:

**Ticker:** `SPY`

**Period:**

```text
2005-01-01 → 2024-12-31
```

The data is obtained using `yfinance`.

The downloaded dataset contains:

* Open
* High
* Low
* Close
* Volume

The original dataset contains **5,032 observations** before feature preprocessing.

After preprocessing, **5,012 observations** remain.

---

# ⚙️ Feature Engineering

The conventional financial feature set consists of:

| Feature         | Description                                         |
| --------------- | --------------------------------------------------- |
| `log_return`    | Logarithmic price return                            |
| `RSI`           | 14-period Relative Strength Index                   |
| `volatility`    | 20-period rolling standard deviation of log returns |
| `MACD`          | Difference between 12-period and 26-period EMAs     |
| `volume_change` | Logarithmic change in trading volume                |

Each 60-day window is locally standardized using a z-score.

The resulting sequence has the shape:

```text
60 × 5
```

The prediction target is binary:

```text
1 → next-day price increases
0 → next-day price does not increase
```

The generated dataset contains:

```text
4,951 windows
```

with an approximately **55.1% positive-label rate**.

---

# 🌀 Topological Data Analysis Pipeline

## 1. Takens Phase-Space Reconstruction

The log-return sequence from each 60-day window is transformed into a phase-space representation using **Takens embedding**.

The embedding parameters are estimated from the training portion of the data.

### Time Delay

Average Mutual Information (AMI) is used to estimate the time delay:

```text
τ = 3
```

### Embedding Dimension

False Nearest Neighbors (FNN) is used to estimate the embedding dimension:

```text
d = 10
```

Therefore, every financial window is transformed into a topological point cloud using:

```text
Takens embedding
τ = 3
d = 10
```

---

# 🔬 Persistent Homology

Persistent homology is calculated using **Ripser**.

The experiments compute:

```text
H0 → connected-component topology
H1 → loop/cycle topology
```

Persistence diagrams are generated for all financial windows.

For example, the notebook reports persistence diagrams containing both H0 and H1 features.

---

# 📐 TDA Feature Extraction

The persistence diagrams are converted into machine-learning features using two complementary representations.

### Persistence Entropy

Persistence entropy summarizes the distribution of feature lifetimes in a persistence diagram.

It is calculated for the finite persistence intervals.

### Persistence Images

H1 persistence diagrams are also transformed into persistence images using `persim`.

The resulting TDA representations include:

* H0 persistence information
* H1 persistence information
* Persistence entropy
* H1 persistence-image features

The single-scale representation contains:

```text
102 TDA features
```

A multi-scale representation is also constructed:

```text
306 TDA features
```

The final Version 2 hybrid model uses the **306-dimensional multi-scale TDA representation**.

---

# 🤖 Model 1 — LSTM + TDA Hybrid

## Version 1

The first model combines two independent representations.

### Financial Sequence Branch

The conventional financial features are processed by an LSTM:

```text
60 × 5
   │
   ▼
 LSTM
   │
   ▼
32-dimensional temporal representation
```

### TDA Branch

The TDA vector is passed through an MLP:

```text
TDA features
    │
    ▼
Linear(32)
    │
   ReLU
    │
   Dropout
    │
    ▼
16-dimensional TDA representation
```

### Fusion

The two representations are concatenated:

```text
LSTM representation
        +
TDA representation
        │
        ▼
   Concatenation
        │
        ▼
    Fusion MLP
        │
        ▼
 Binary prediction
```

The Version 1 model therefore performs **late feature fusion through concatenation**.

---

# 🚀 Model 2 — Gated LSTM + TDA Hybrid

## Version 2

Version 2 improves the original architecture by introducing **gated fusion**.

Instead of simply concatenating the LSTM and TDA representations, the model learns how much information to take from each branch.

### LSTM Branch

```text
60 × 5
  │
  ▼
LSTM
  │
  ▼
Temporal representation
```

### TDA Branch

```text
306-dimensional TDA vector
          │
          ▼
       Linear(64)
          │
         ReLU
          │
       Dropout
          │
          ▼
       Linear(32)
          │
         ReLU
          │
          ▼
   Topological representation
```

### Projection

Both representations are projected into the same fusion dimension.

```text
LSTM representation ──► Raw Projection ──┐
                                         ├──► Gating Network
TDA representation ───► TDA Projection ──┘
```

The learned gate is:

```text
gate = sigmoid(G([raw_repr, tda_repr]))
```

The fused representation is:

```text
fused = gate × raw_repr
      + (1 - gate) × tda_repr
```

This allows the network to learn whether a particular sample should rely more heavily on:

* temporal financial information, or
* topological information.

---

# 📊 Experimental Setup

The data is divided chronologically:

```text
70% → Training
15% → Validation
15% → Testing
```

This avoids randomly mixing future observations into the training set.

Training uses:

* PyTorch
* AdamW optimizer
* Learning rate: `5e-4`
* Weight decay: `1e-4`
* Batch size: `64`
* Dropout: `0.2`
* Maximum epochs: `40`
* Early stopping patience: `7`
* Gradient clipping: `1.0`
* Binary Cross Entropy with logits
* Class weighting based on the training distribution

The experiments use fixed random seeds for reproducibility.

---

# 📈 Version 2 Results

The final Version 2 experiment compares the baseline LSTM with the improved gated hybrid model.

| Metric            | Baseline LSTM | Gated LSTM + TDA |
| ----------------- | ------------: | ---------------: |
| Accuracy          |        0.4980 |       **0.4993** |
| Precision         |    **0.5329** |           0.5217 |
| Recall            |        0.4315 |       **0.6726** |
| F1                |        0.4769 |       **0.5876** |
| ROC-AUC           |    **0.5015** |           0.4650 |
| Balanced Accuracy |    **0.5023** |           0.4882 |
| MCC               |    **0.0046** |          -0.0254 |
| Sharpe            |        0.1736 |       **0.2456** |
| Sortino           |        0.1595 |       **0.2839** |
| Annualized Return |          2.2% |         **3.8%** |
| Max Drawdown      |       -23.41% |          -23.60% |
| Calmar            |        0.0942 |       **0.1608** |
| Win Rate          |        22.88% |       **35.67%** |
| Number of Trades  |           147 |          **121** |
| Cumulative Return |         6.64% |       **11.61%** |

### Interpretation

The Version 2 hybrid model produced:

* higher recall,
* higher F1,
* higher Sharpe ratio,
* higher Sortino ratio,
* higher annualized return,
* higher cumulative return,
* fewer trades,

than the baseline LSTM in the reported single-seed test.

However, its ROC-AUC, balanced accuracy, and MCC were lower.

Therefore, the results **do not demonstrate that TDA universally improves predictive classification performance**. Instead, they suggest that the topological representation can affect the resulting trading strategy differently from conventional classification metrics.

---

# 💰 Transaction-Cost Sensitivity

Version 2 was also evaluated under different transaction-cost assumptions.

|   Cost | Cumulative Return |  Sharpe |
| -----: | ----------------: | ------: |
|  0 bps |            12.97% |  0.2727 |
|  1 bps |            11.61% |  0.2456 |
|  5 bps |             6.34% |  0.1373 |
| 10 bps |             0.10% |  0.0021 |
| 20 bps |           -11.31% | -0.2673 |

This shows that the strategy is sensitive to transaction costs.

At higher transaction costs, the advantage disappears.

---

# 🧩 Ablation Study

Version 2 includes an ablation study to investigate which components contribute to the model.

The evaluated configurations include:

* Raw-only LSTM
* TDA-only MLP
* LSTM + H0
* LSTM + H1 entropy
* LSTM + H1 persistence image + entropy
* LSTM + all single-scale TDA
* LSTM + all multi-scale TDA

### Selected Results

| Model                         |   Accuracy |         F1 |     Sharpe | Cumulative Return |
| ----------------------------- | ---------: | ---------: | ---------: | ----------------: |
| Raw-only                      |     0.4980 |     0.4769 |     0.1736 |             6.64% |
| TDA-only                      |     0.4805 |     0.3931 |     0.2776 |             7.05% |
| LSTM + H0                     | **0.5249** |     0.6257 |     0.5322 |            28.05% |
| LSTM + H1 entropy             |     0.5128 |     0.5896 | **0.5751** |            29.59% |
| LSTM + H1 image + entropy     | **0.5303** | **0.6931** |     0.5362 |        **31.98%** |
| LSTM + all TDA (single-scale) |     0.5303 |     0.6931 |     0.5362 |            31.98% |
| LSTM + all TDA (multi-scale)  |     0.4993 |     0.5876 |     0.2456 |            11.61% |

The ablation results indicate that individual/single-scale TDA components can behave differently from the final multi-scale representation.

---

# 🔁 Multi-Seed Stability

Version 2 also evaluates the baseline and improved hybrid models across eight random seeds:

```text
13
42
101
2024
777
3407
8
99
```

Mean ± standard deviation results include:

| Metric            |       Baseline LSTM |     Improved Hybrid |
| ----------------- | ------------------: | ------------------: |
| Accuracy          |     0.5017 ± 0.0179 |     0.4886 ± 0.0232 |
| F1                |     0.5389 ± 0.0883 | **0.5411 ± 0.1156** |
| ROC-AUC           | **0.4990 ± 0.0065** |     0.4571 ± 0.0180 |
| Balanced Accuracy | **0.4965 ± 0.0078** |     0.4808 ± 0.0122 |
| MCC               |    -0.0066 ± 0.0164 |    -0.0426 ± 0.0267 |
| Annualized Return |   **4.58% ± 4.05%** |       3.21% ± 5.61% |
| Max Drawdown      |     -22.81% ± 6.01% | **-22.57% ± 2.71%** |
| Calmar            | **0.2383 ± 0.2205** |     0.1611 ± 0.2580 |
| Cumulative Return | **14.53% ± 13.03%** |     10.57% ± 17.36% |

The multi-seed experiment therefore provides a more cautious interpretation than the single-seed experiment.

For example, the paired Wilcoxon test for ROC-AUC produced:

```text
p = 0.0078
```

while the tests for accuracy, F1, Sharpe, Sortino, and cumulative return were not statistically significant at the conventional 0.05 level in the reported experiment.

---

# 🆚 Version 1 vs Version 2

| Component             | Version 1     | Version 2                     |
| --------------------- | ------------- | ----------------------------- |
| Financial sequence    | LSTM          | LSTM                          |
| TDA representation    | TDA features  | Multi-scale TDA features      |
| TDA encoder           | MLP           | MLP                           |
| Fusion                | Concatenation | **Learned gated fusion**      |
| Fusion representation | Fixed         | **Data-dependent**            |
| Ablation study        | Yes           | Yes                           |
| Financial backtesting | Yes           | Yes                           |
| Multi-seed analysis   | Yes           | Yes                           |
| Main improvement      | Basic hybrid  | **Gated hybrid architecture** |

The main architectural improvement from Version 1 to Version 2 is the replacement of simple concatenation with a **learned gating mechanism**.

---

# 🏗️ Project Structure

A recommended GitHub repository structure is:

```text
tda-financial-prediction/
│
├── notebooks/
│   ├── tda-thesis-version-1.ipynb
│   └── tda-thesis-version-2.ipynb
│
├── data/
│   └── preprocessed_data.csv
│
├── results/
│   ├── figures/
│   ├── ablation_results.csv
│   └── model_comparison.csv
│
├── README.md
├── requirements.txt
└── LICENSE
```

The notebooks currently contain the complete experimental workflow.

---

# 🛠️ Technologies

* Python 3
* NumPy
* Pandas
* Matplotlib
* SciPy
* scikit-learn
* PyTorch
* yfinance
* Ripser
* Persim

### Main TDA Libraries

```text
ripser
persim
```

### Deep Learning

```text
PyTorch
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/tda-financial-prediction.git
cd tda-financial-prediction
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch yfinance ripser persim
```

Or create a `requirements.txt` containing:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
torch
yfinance
ripser
persim
```

Then:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Open either notebook using Jupyter:

```bash
jupyter notebook
```

Then run:

```text
notebooks/tda-thesis-version-1.ipynb
```

or:

```text
notebooks/tda-thesis-version-2.ipynb
```

The notebooks perform:

1. Data acquisition
2. Data preprocessing
3. Technical indicator calculation
4. Sliding-window generation
5. Takens embedding
6. Persistent homology
7. TDA feature extraction
8. LSTM training
9. Hybrid model training
10. Model evaluation
11. Financial backtesting
12. Ablation experiments
13. Multi-seed stability analysis

---

# 🔬 Research Questions

This project investigates several questions:

### RQ1

Can topological features extracted from financial time series provide useful information for next-day market-direction prediction?

### RQ2

Does combining TDA representations with an LSTM improve over an LSTM using conventional financial features?

### RQ3

Does learned gated fusion provide a better way to combine temporal and topological representations than simple concatenation?

### RQ4

Which topological representations contribute most strongly to the predictive and financial performance?

### RQ5

Are observed improvements stable across different random seeds and transaction-cost assumptions?

---

# ⚠️ Limitations

The current experiments have several important limitations.

### 1. Single Asset

The primary experiment uses only:

```text
SPY
```

Generalization to other assets has not been established.

### 2. Predictive Performance Is Close to Chance

The reported classification accuracy is close to 50%.

Therefore, the results should **not** be interpreted as evidence of a highly accurate market-direction predictor.

### 3. Transaction Costs Matter

The strategy's performance deteriorates considerably as transaction costs increase.

### 4. Multi-Seed Results Are Mixed

The multi-seed experiment does not consistently show that the improved hybrid model outperforms the baseline across all metrics.

### 5. No Claim of Investment Performance

This is a research/academic machine-learning experiment and **not financial advice or an investment strategy recommendation**.

---

# 🚀 Future Work

Potential extensions include:

* Testing multiple stocks and ETFs
* Testing different market regimes
* Using higher-frequency financial data
* Comparing additional TDA representations
* Exploring persistent landscapes and persistence diagrams directly
* Learning TDA representations using neural networks
* Attention-based temporal models
* Transformer + TDA architectures
* More sophisticated fusion mechanisms
* Cross-asset topological features
* Walk-forward validation
* Nested hyperparameter optimization
* More realistic transaction-cost and slippage models
* Statistical comparison against stronger financial baselines

---

# 📚 Key Concepts

This project combines three major areas:

```text
Financial Time-Series Analysis
              +
Deep Learning
              +
Topological Data Analysis
```

### Financial Time Series

Technical indicators and returns represent conventional temporal information.

### LSTM

The LSTM learns temporal dependencies in the sequence of financial features.

### Topological Data Analysis

TDA provides a representation of geometric/topological structure in reconstructed financial dynamics.

### Persistent Homology

Persistent homology identifies topological structures across scales.

### Gated Fusion

Version 2 learns a data-dependent balance between:

```text
Temporal Representation
        ↕
Topological Representation
```

---

# 📄 Notebooks

The repository contains two experimental versions:

### Version 1

`tda-thesis-version-1.ipynb`

The initial **LSTM + TDA hybrid** architecture using concatenation-based fusion.

### Version 2

`tda-thesis-version-2.ipynb`

The improved **Gated LSTM + TDA hybrid** architecture with multi-scale TDA features, ablation analysis, transaction-cost sensitivity, and multi-seed stability testing.

---

# 👨‍💻 Author

**Saadmaan Shohid**

Bachelor's Thesis / Research Project

---

# ⚖️ Disclaimer

This repository is intended for **academic and research purposes only**.

The models and backtesting results should not be considered financial advice, investment recommendations, or evidence of guaranteed future market performance.

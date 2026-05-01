# A Comparative Study of Probabilistic Electricity Load Forecasting Using Baseline, LSTM, and Transformer Models

**DATA 612: Deep Learning -- University of Maryland, College Park**


---

## Overview

This project applies a Transformer-based deep learning model to predict electricity consumption using the [UCI Electricity Load Diagrams 2011-2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset. The Transformer outputs probabilistic forecasts (10th, 50th, and 90th percentile predictions) for a 24-hour horizon, trained with Quantile Loss (Pinball Loss). Performance is compared against naive baselines, seasonal baselines, and LSTM models.

## Repository Structure

```
deep_learning_project/
└── experimentation/
    ├── data_utils.ipynb       # Data pipeline (download, clean, features, split, normalize)
    ├── baselines.ipynb        # Naive baselines + LSTM models
    ├── transformers.ipynb     # Probabilistic Transformer model
    └── evaluation.ipynb       # Evaluation metrics, comparison, and visualizations
```

### File Descriptions

**`data_utils.ipynb`** -- Data Engineering & Preprocessing
- Downloads the UCI Electricity dataset (370 customers, 15-min intervals, 2011-2014)
- Aggregates customer-level readings into a single hourly total load series
- Creates cyclical time features (hour_sin/cos, dayofweek_sin/cos, month_sin/cos, is_weekend)
- Splits data chronologically (70% train / 15% val / 15% test)
- Normalizes using StandardScaler fitted on training data only
- **Output variables:** `train_scaled`, `val_scaled`, `test_scaled`, `scaler`, `df_features`

**`baselines.ipynb`** -- Baseline Models
- Clones the repo and runs `data_utils.ipynb` via `%run` to load the preprocessed data
- Creates sliding window sequences (168h input, 24h forecast)
- Implements three naive baselines:
  - Naive Persistence (repeat last 24h)
  - Seasonal Naive Day (same hour yesterday, lag=24)
  - Seasonal Naive Week (same hour last week, lag=168)
- Trains two LSTM models (MSE loss, point forecast):
  - Univariate LSTM (load only, input_size=1)
  - Multivariate LSTM (7 features, input_size=7)
- **Output variables:** `naive_persistence_preds_scaled`, `seasonal_naive_day_preds_scaled`, `seasonal_naive_week_preds_scaled`, `univariate_lstm_preds_scaled`, `multivariate_lstm_preds_scaled`, `y_test_scaled`

**`transformers.ipynb`** -- Probabilistic Transformer
- Clones the repo and runs `data_utils.ipynb` via `%run` to load the preprocessed data
- Implements the Transformer encoder architecture:
  - Input projection (7 features to d_model=64)
  - Sinusoidal positional encoding
  - 2-layer Transformer encoder (4 heads, FFN=128)
  - Output head producing 24x3 quantile predictions (q10, q50, q90)
- Trains with Quantile Loss (Pinball Loss) for 30 epochs with LR scheduling
- **Output variables:** `model` (trained Transformer), `train_losses`, `val_losses`

**`evaluation.ipynb`** -- Evaluation & Comparison
- Runs `baselines.ipynb` and `transformers.ipynb` via `%run` to get all trained models and predictions
- Generates Transformer test predictions (extracts q50 median as point forecast)
- Inverse-transforms all predictions to original scale
- Implements evaluation metrics from scratch: MAE, RMSE, MAPE
- Produces comparison table, improvement analysis, and visualizations:
  - Lollipop chart (models ranked by metric)
  - 24-hour forecast overlay (all models vs actual)
  - Horizon-wise error analysis (h=1 to h=24)
  - Transformer training/validation loss curve

## How to Run

### Prerequisites

- Google Colab (recommended, GPU runtime)
- Python 3.10+
- PyTorch, NumPy, Pandas, Matplotlib, scikit-learn

### Step-by-Step

1.  Step-by-Step

 **Clone the repository**
```bash
   git clone https://github.com/samarthsingh1/deep_learning_project.git
   cd deep_learning_project/experimentation
```

   Or on Google Colab:
```python
   !git clone https://github.com/samarthsingh1/deep_learning_project.git
```
2.  **Open any notebook in Google Colab**

   Each notebook is self-contained. It clones this repo and runs its dependencies automatically via `%run`.

3. **Run `evaluation.ipynb` for the full pipeline**

   This is the main notebook that chains everything together:
   ```
   evaluation.ipynb
     └── %run baselines.py
           └── %run data_utils.py    (downloads data, preprocesses)
           └── trains LSTMs, generates baseline predictions
     └── %run transformers.py
           └── %run data_utils.py    (downloads data, preprocesses)
           └── trains Transformer
     └── generates Transformer test predictions
     └── computes all metrics and visualizations
   ```

   Simply open `evaluation.ipynb` in Colab, set runtime to GPU, and **Run All**.

4. **Or run individual notebooks**

   - Run `data_utils.ipynb` alone to explore the dataset and preprocessing
   - Run `baselines.ipynb` to train and evaluate only the baselines and LSTMs
   - Run `transformers.ipynb` to train only the Transformer model

### Runtime Estimates (Colab GPU)

| Notebook | Approximate Runtime |
|----------|-------------------|
| `data_utils.ipynb` | ~3 min (dataset download) |
| `baselines.ipynb` | ~10 min (download + LSTM training) |
| `transformers.ipynb` | ~15 min (download + Transformer training) |
| `evaluation.ipynb` | ~30 min (runs both baselines + transformer + evaluation) |

### Notes

- The dataset (~250 MB) is downloaded automatically from the UCI repository on each run. A stable internet connection is required.
- The first cells in `baselines.ipynb` and `transformers.ipynb` clone this repo and convert `data_utils.ipynb` to a `.py` script using `jupyter nbconvert`, then execute it with `%run`. This is how the shared data pipeline is reused across notebooks.
- All models use the same chronological 70/15/15 train/val/test split to ensure fair comparison.
- Random seeds are set (`set_seed(42)`) for reproducibility, though minor variations may occur across different GPU hardware.

## Results Summary

| Model | MAE | RMSE | MAPE (%) |
|-------|-----|------|----------|
| Naive Persistence | 35,515 | 58,203 | 3.82 |
| Seasonal Naive Day | 35,515 | 58,203 | 3.82 |
| Seasonal Naive Week | 59,839 | 95,700 | 5.79 |
| Univariate LSTM | 37,442 | 52,527 | 4.23 |
| Multivariate LSTM | 34,356 | 48,108 | 3.81 |
| **Transformer (q50)** | **33,356** | **48,018** | **3.66** |

## References

[1] L’Heureux A, Grolinger K, Capretz MAM. Transformer-Based Model for Electrical Load Forecasting. Energies. 2022; 15(14):4993. https://doi.org/10.3390/en15144993
[2] Abumohsen M, Owda AY, Owda M. Electrical Load Forecasting Using LSTM, GRU, and RNN Algorithms. Energies. 2023; 16(5):2283. https://doi.org/10.3390/en16052283
[3] UCI Machine Learning Repository. Electricity Load Diagrams 2011–2014. https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
[4] Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 9(8):1735–1780, 1997
[5] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI Conference on Artificial Intelligence, 2021.
[6] Yi Wang, Dahua Gan, Mingyang Sun, Ning Zhang, Zongxiang Lu, Chongqing Kang. Probabilistic individual load forecasting using pinball loss guided LSTM
[7] Kozodoi, N., Zinovyeva, E., Valentin, S., Pereira, J., & Agundez, R. (2024). Probabilistic demand forecasting with graph neural networks. arXiv preprint arXiv:2401.13096.
[8] Dahua Gan; Yi Wang; Shuo Yang; Chongqing Kang - Embedding based quantile regression neural network for probabilistic load forecasting
[9] Omar Bouhamed, Maher Dissem, Manar Amayri, Nizar Bouguila - Transformer-based deep probabilistic network for load forecasting
[10] Wei Zhang, Hongyi Zhan, Hang Sun, Mao Yang - Probabilistic load forecasting for integrated energy systems based on quantile regression patch time series Transformer
[11] Scheuerer, M., Switanek, M. B., Worsnop, R. P., & Hamill, T. M. (2020). Using artificial neural networks for generating probabilistic subseasonal precipitation forecasts over California. Monthly Weather Review, 148(8). https://journals.ametsoc.org/view/journals/mwre/148/8/mwrD200096.pdf
[12]  Tao Hong, Shu Fan - Probabilistic electric load forecasting: A tutorial review
[13] Guillermo Moraleda Conejo  - PROBABILISTIC RESIDENTIAL LOAD FORECASTING BASED ON LSTM RECURRENT NEURAL NETWORKS: IMPLEMENTATION AND ASSESSMENT
[14] Syed Afraz Hussain Shah, Ubaid Ahmed, Muhammad Bilal - Improved electric load forecasting using quantile long short-term memory network with dual attention mechanism


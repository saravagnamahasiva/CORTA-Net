# Delhi Air Pollution Forecasting - CORTA-Net Model

## Project Description

**Advanced Machine Learning Framework for Forecasting Air Pollution in Delhi (2025-2030) using Hybrid Deep Learning Architecture**

This research-grade project implements a sophisticated **CORTA-Net (Cross-Orthogonal Recurrent Temporal Attention Network)** model for predicting PM2.5 and PM10 concentrations in Delhi. The framework combines state-of-the-art deep learning techniques with comprehensive feature engineering to provide accurate air quality forecasts and policy recommendations.

##  Key Features

- **Advanced Hybrid Architecture**: CORTA-Net model integrating cross-attention mechanisms, orthogonal regularization, and temporal attention
- **CorrXGBoost Feature Selection**: Hybrid feature selection combining Pearson correlation and XGBoost importance
- **Comprehensive Analysis**: Seasonal trends, stubble burning impact, and regional variations
- **Future Predictions**: Detailed forecasts for 2025-2030 with uncertainty quantification
- **Research-Grade Implementation**: Complete with ablation studies, statistical validation, and publication-ready visualizations
- **Policy Recommendations**: Data-driven strategies for pollution mitigation

## Repository Structure
Delhi-Air-Pollution-Forecasting/
│
├── 📂 data/
│ ├── city_day.csv
│ ├── station_day.csv
│ └── stations.csv
│
├── 📂 models/
│ ├── best_corta_net_model.pth
├── 📂 results/
│ ├── predictions_2025_2030.csv
│ ├── sensitivity_analysis.png
│ └── hyperparameter_results.csv
│
├── 📂 src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── corta_net.py
│ └── visualization.py
│
├── 📄 requirements.txt
├── 📄 environment.yml
├── 📄 setup.py
└── 📄 README.md



##Algorithms & Models Implemented

### **Core Architecture: CORTA-Net**
Input → Cross-Attention Layer → Orthogonal LSTM Layers (100 units, 2 layers) →
Temporal Attention Heads (4 heads) → Dense Layers (64, 32 units) → Output



### **Feature Engineering Pipeline**
1. **Temporal Features**: Year, Month, DayOfYear, Season, Quarter
2. **Cyclical Encoding**: sin/cos transformations for temporal patterns
3. **Lag Features**: 1, 2, 3, 7-day lags for PM2.5 and PM10
4. **Rolling Statistics**: 3, 7, 14-day moving averages and standard deviations
5. **Seasonal Indicators**: Stubble burning season flags

### **Feature Selection: CorrXGBoost**
- **Step 1**: Pearson correlation filtering (|r| > 0.3)
- **Step 2**: XGBoost importance selection (threshold > 0.015)
- **Step 3**: Union of both selected feature sets

### **Model Variants (Ablation Studies)**
1. **LSTM**: Baseline LSTM model
2. **LSTM-Attention**: LSTM with temporal attention
3. **Ortho-LSTM**: Orthogonal regularized LSTM
4. **CORTA-Net**: Complete proposed architecture

##  Tech Stack

### **Core Technologies**
- **Python 3.8+**
- **PyTorch 2.0+**: Deep learning framework
- **Scikit-learn**: Traditional ML models and metrics
- **XGBoost**: Feature importance and baseline models
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization

### **Key Libraries**

# Deep Learning
torch, torch.nn, torch.optim

# Data Processing
pandas, numpy, scipy

# Machine Learning
sklearn, xgboost

# Visualization
matplotlib, seaborn, plotly
 Contact
For questions, collaborations, or feedback:

Email: sarvagnamahashiva410@gmail.com





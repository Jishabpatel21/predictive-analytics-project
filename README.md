# Predictive Analysis Project

A comprehensive data analysis project featuring two distinct statistical analysis notebooks: **Model Selection using AIC/BIC** and **Time Series Visualization with Bokeh**. This project demonstrates advanced statistical techniques for model evaluation and interactive financial time series analysis.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Prerequisites & Installation](#prerequisites--installation)
- [Notebooks Description](#notebooks-description)
  - [1. Model Selection with AIC/BIC](#1-model-selection-with-aicbic)
  - [2. Time Series Visualization with Bokeh](#2-time-series-visualization-with-bokeh)
- [Usage Instructions](#usage-instructions)
- [Key Features](#key-features)
- [Technical Details](#technical-details)
- [Results & Outputs](#results--outputs)
- [Configuration & Parameters](#configuration--parameters)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [References](#references)

---

## 🎯 Project Overview

This project comprises two independent analytical components:

1. **Statistical Model Selection Framework**: Implements AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) for systematic model comparison and selection using the Iris dataset.

2. **Interactive Financial Time Series Analysis**: Provides sophisticated visualization of equity price data with shock detection, moving averages, and autocorrelation analysis using Bokeh for Indian IT stocks (INFY, TCS, NIFTY IT).

Both notebooks demonstrate best practices in data analysis, visualization, and statistical modeling.

---

## 📁 Project Structure

```
Predictive analysis project/
│
├── README.md                           # This comprehensive documentation
│
├── Model_Selection_AIC_BIC.ipynb       # Statistical model selection notebook
├── Predictive analysis part 2.ipynb    # Time series visualization notebook
│
└── data/                                # Stock market data directory
    ├── INFY_2015_2016.csv              # Infosys stock data (494 records)
    ├── TCS_2015_2016.csv               # TCS stock data (494 records)
    └── NIFTY_IT_2015_2016.csv          # NIFTY IT Index data (210 records)
```

---

## 📊 Datasets

### 1. Iris Dataset (Built-in)
- **Source**: scikit-learn library
- **Records**: 150 samples
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Target**: 3 species (setosa, versicolor, virginica)
- **Usage**: Model selection and AIC/BIC comparison

### 2. Indian IT Stock Market Data
Three CSV files containing daily trading data for 2015-2016:

#### INFY_2015_2016.csv (Infosys)
- **Records**: 494 trading sessions
- **Date Range**: 2015-01-01 to 2016-12-30
- **Columns**: date, open, high, low, close, volume

#### TCS_2015_2016.csv (Tata Consultancy Services)
- **Records**: 494 trading sessions
- **Date Range**: 2015-01-01 to 2016-12-30
- **Columns**: date, open, high, low, close, volume

#### NIFTY_IT_2015_2016.csv (NIFTY IT Index)
- **Records**: 210 trading sessions
- **Date Range**: 2015-01-02 to 2016-12-30
- **Columns**: date, open, high, low, close, volume
- **Note**: Volume is 0 (index data, not traded)

All timestamps include timezone offset (+05:30 IST).

---

## 🔧 Prerequisites & Installation

### Required Python Version
- Python 3.8 or higher

### Dependencies

#### For Model Selection Notebook:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

#### For Time Series Visualization Notebook:
```bash
pip install bokeh pandas numpy statsmodels
```

#### Install All Dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn bokeh statsmodels
```

### Optional (for Jupyter):
```bash
pip install jupyter notebook
# or
pip install jupyterlab
```

---

## 📓 Notebooks Description

### 1. Model Selection with AIC/BIC

**File**: `Model_Selection_AIC_BIC.ipynb`

#### Purpose
Demonstrates systematic model selection using information criteria (AIC and BIC) to balance model fit and complexity, preventing overfitting while maintaining predictive performance.

#### What It Does
- Loads and explores the Iris dataset
- Builds 5 regression models of increasing complexity
- Calculates AIC and BIC for each model
- Compares models using multiple metrics
- Visualizes trade-offs between complexity and performance
- Provides recommendations based on statistical criteria

#### Models Evaluated
1. **Simple Linear Regression** (1 feature: petal length)
   - Parameters: 2 (slope + intercept)
   
2. **Multiple Linear Regression** (2 features: petal length + width)
   - Parameters: 3
   
3. **Multiple Linear Regression** (3 features: all predictors)
   - Parameters: 4
   
4. **Polynomial Regression** (degree 2)
   - Parameters: 10 (quadratic terms + interactions)
   
5. **Polynomial Regression** (degree 3)
   - Parameters: 20 (cubic terms + higher interactions)

#### Key Formulas

**AIC (Akaike Information Criterion)**
```
AIC = n × ln(MSE) + 2k
```
where:
- n = number of observations
- MSE = mean squared error
- k = number of parameters

**BIC (Bayesian Information Criterion)**
```
BIC = n × ln(MSE) + k × ln(n)
```

**Interpretation**:
- Lower values indicate better models
- BIC penalizes complexity more heavily than AIC
- Difference > 10 suggests strong evidence for model preference

#### Outputs
1. **Comparison Table**: All models ranked by AIC/BIC
2. **Visualizations** (6 panels):
   - AIC comparison bar chart
   - BIC comparison bar chart
   - Model complexity (parameter count)
   - Test R² scores
   - Test MSE values
   - AIC vs BIC scatter plot
3. **Detailed Analysis**: 
   - Best model identification
   - Overfitting assessment
   - Practical recommendations
4. **Prediction Plots**:
   - Actual vs Predicted scatter
   - Residual analysis

---

### 2. Time Series Visualization with Bokeh

**File**: `Predictive analysis part 2.ipynb`

#### Purpose
Interactive visual analysis of equity time series to identify price dynamics, volume anomalies, and return patterns using advanced Bokeh visualizations.

#### What It Does
- Loads daily stock data from CSV files
- Computes returns, volume changes, and moving averages
- Detects price and volume shocks using statistical thresholds
- Creates interactive visualizations with color-coded insights
- Calculates and plots Partial Autocorrelation Function (PACF)
- Presents data in tabbed interface (one tab per stock symbol)

#### Core Algorithms

##### 1. Returns and Volume Change
```
Return_t = (Close_t / Close_{t-1}) - 1
VolChg_t = (Volume_t / Volume_{t-1}) - 1
```

##### 2. Moving Average (52-week ≈ 252 trading days)
```
MA52_t = rolling_mean(Close, window=252, min_periods=10)
MA_diff_t = Close_t - MA52_t
```

##### 3. Shock Detection (z-score method)
```
PriceShock_t = |Return_t - μ_ret| > 3σ_ret
VolShock_t = |VolChg_t - μ_volchg| > 3σ_volchg
VolumelessPriceShock_t = PriceShock_t AND NOT VolShock_t
```

##### 4. PACF (Partial Autocorrelation Function)
- Computed using statsmodels (Yule-Walker MLE method)
- Confidence bands: ±1.96 / √N (95% significance level)
- Plotted with Bokeh (custom implementation)

#### Visualization Features

##### Time Series Plot
- **Blue Gradient**: Color intensity represents deviation from 52-week MA
  - Darker blue = price significantly above MA
  - Lighter blue = price near or below MA
- **Red Segments**: Time periods between consecutive volume shocks
- **Orange Markers**: Price shocks without volume confirmation
- **Color Bar**: Legend showing Close - MA52 mapping
- **Interactive Tools**: Pan, zoom, hover tooltips

##### PACF Plot
- **Vertical Bars**: PACF values at each lag
- **Dashed Lines**: ±1.96/√N confidence intervals
- **Zero Line**: Reference baseline
- **Interpretation**: Bars crossing confidence bands indicate significant autocorrelation

#### Key Functions

```python
_find_cols(df)                      # Normalize column names
load_symbol_csv(path)               # Load and clean CSV data
preprocess(df)                      # Calculate all derived metrics
_between_volume_shocks_mask()       # Identify red segment regions
make_timeseries_plot(sym, dfx)     # Create interactive time series
make_pacf_plot(sym, dfx)           # Create PACF visualization
```

---

## 🚀 Usage Instructions

### Running Model Selection Notebook

1. **Open Jupyter**:
   ```bash
   jupyter notebook "Model_Selection_AIC_BIC.ipynb"
   ```

2. **Run All Cells**: Execute cells sequentially from top to bottom

3. **Expected Runtime**: ~30 seconds

4. **Outputs**: 
   - Statistical tables in console
   - Multiple visualization plots
   - Model recommendations

### Running Time Series Visualization Notebook

1. **Verify Data Files**: Ensure CSV files exist in `data/` folder

2. **Open Jupyter**:
   ```bash
   jupyter notebook "Predictive analysis part 2.ipynb"
   ```

3. **Run Cells**:
   - Cell 1: Documentation (skip, just read)
   - Cell 2: Import libraries and define functions
   - Cell 3: Load data and generate visualizations

4. **Expected Output**: Interactive Bokeh plots (tabbed or stacked layout)

5. **Interaction**:
   - Click and drag to pan
   - Scroll to zoom
   - Hover over points for details
   - Click legend items to hide/show series

---

## ⭐ Key Features

### Model Selection Notebook
✅ **Automated Model Comparison**: 5 models evaluated simultaneously  
✅ **Information Criteria**: Both AIC and BIC calculated  
✅ **Visual Analytics**: 6 comprehensive comparison charts  
✅ **Overfitting Detection**: Train vs Test R² analysis  
✅ **Statistical Rigor**: Proper train/test split with standardization  
✅ **Clear Recommendations**: Evidence-based model selection  

### Time Series Notebook
✅ **Multi-Symbol Support**: Analyzes all CSV files in data folder  
✅ **Intelligent Column Detection**: Flexible CSV parsing  
✅ **Shock Detection**: Statistical identification of anomalies  
✅ **Interactive Visualization**: Full Bokeh feature set  
✅ **Color-Coded Insights**: Gradient mapping and conditional coloring  
✅ **Autocorrelation Analysis**: PACF with confidence bands  
✅ **Robust Error Handling**: Graceful degradation for missing data  
✅ **Tabbed Interface**: Clean organization of multiple symbols  

---

## 🔬 Technical Details

### Model Selection: Metrics Explained

| Metric | Formula | Interpretation | Lower/Higher Better? |
|--------|---------|----------------|---------------------|
| AIC | n·ln(MSE) + 2k | Information loss + complexity penalty | Lower |
| BIC | n·ln(MSE) + k·ln(n) | Bayesian approach, stronger penalty | Lower |
| R² | 1 - RSS/TSS | Proportion of variance explained | Higher |
| MSE | Σ(y - ŷ)²/n | Average squared error | Lower |

### Time Series: Assumptions

| Assumption | Value | Rationale |
|------------|-------|-----------|
| Trading Year | 252 days | Standard excluding weekends/holidays |
| MA Window | 252 days | Approximately 52 weeks |
| Shock Threshold | 3σ | 99.7% confidence (z-score) |
| PACF Confidence | 1.96/√N | 95% significance level |
| Min MA Periods | 10 days | Minimum for stable average |

### Bokeh Visualization Components

```python
# Core Bokeh elements used:
- figure(): Base plotting canvas
- ColumnDataSource: Data container for glyphs
- Segment: Line segments with individual colors
- Scatter: Point markers
- LinearColorMapper: Gradient color mapping
- ColorBar: Legend for color scale
- HoverTool: Interactive tooltips
- Span: Horizontal/vertical reference lines
- Tabs/TabPanel: Multi-symbol organization
```

---

## 📈 Results & Outputs

### From Model Selection Notebook

**Typical Output**:
```
Best Model by AIC: Multiple Linear (3 features)
  - AIC: -45.23
  - Test R²: 0.862
  - Test MSE: 0.047

Best Model by BIC: Multiple Linear (2 features)
  - BIC: -38.91
  - Test R²: 0.845
  - Test MSE: 0.053
```

**Key Insight**: BIC often selects simpler models, making it preferable for production systems where interpretability matters.

### From Time Series Notebook

**Per Symbol Output**:
- **Time Series Plot**: 800×350 px, datetime x-axis
- **PACF Plot**: 450×350 px, lag x-axis
- **Layout**: Side-by-side in tabs or stacked rows

**Typical Findings**:
- **INFY**: 2-5 volume shocks per year, strong MA adherence
- **TCS**: Similar patterns, moderate volatility
- **NIFTY IT**: Index behavior, no real volume (synthetic)

---

## ⚙️ Configuration & Parameters

### Time Series Notebook - Tunable Parameters

Edit in `preprocess()` function:

```python
# Moving average window (default: 252 for 52-week MA)
out['MA52'] = out['Close'].rolling(252, min_periods=10).mean()

# Shock thresholds (default: 3.0 standard deviations)
out['PriceShock'] = (np.abs(ret - r_mu) > 3.0 * r_sd)
out['VolShock'] = (np.abs(vch - v_mu) > 3.0 * v_sd)
```

Edit in `make_pacf_plot()`:

```python
# PACF lags (default: min(60, N//2))
nlags = max_lags or min(60, n//2)
```

### Color Palette Options

Replace `Blues256` with other Bokeh palettes:
- `Viridis256` - Yellow to purple
- `Plasma256` - Purple to yellow
- `Inferno256` - Black to yellow
- `Magma256` - Black to white
- `Turbo256` - Rainbow spectrum

```python
from bokeh.palettes import Viridis256
mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Model Selection Notebook

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
```bash
Solution: pip install scikit-learn
```

**Issue**: Plots not displaying
```python
Solution: Add at start of notebook:
%matplotlib inline
```

#### Time Series Notebook

**Issue**: No plots appear
- **Check**: CSV files exist in `data/` folder
- **Check**: File names match pattern `*.csv`
- **Check**: Files have Date, Close columns

**Issue**: `ImportError: cannot import name 'TabPanel'`
- **Cause**: Older Bokeh version
- **Solution**: Update Bokeh: `pip install --upgrade bokeh`
- **Fallback**: Code automatically switches to stacked layout

**Issue**: All volume = 0 (NIFTY IT)
- **Expected**: Index data has no actual volume
- **Effect**: No red segments (no volume shocks)
- **Normal**: PACF still works on returns

**Issue**: Short dataset warning
```python
# Appears when returns < 10 data points
# Solution: Use longer date range or more frequent data
```

#### Performance Issues

**Large Datasets** (>1000 points):
```python
# Enable WebGL rendering in Bokeh figure
p = figure(..., output_backend='webgl')
```

**Memory Issues**:
- Process symbols one at a time
- Use data sampling for initial exploration
- Close notebook kernel between runs

---

## 🚧 Future Enhancements

### Planned Features

#### Model Selection Notebook
- [ ] Cross-validation with k-folds
- [ ] Additional information criteria (AICc, HQIC)
- [ ] Automated feature selection
- [ ] Non-linear model types (SVM, Random Forest)
- [ ] ROC curves for classification variant
- [ ] LaTeX equation rendering

#### Time Series Notebook
- [ ] Interactive parameter controls (sliders)
- [ ] Multi-timeframe analysis (daily, weekly, monthly)
- [ ] Volatility overlays (GARCH, realized vol)
- [ ] Correlation heatmaps between symbols
- [ ] Export to standalone HTML
- [ ] Real-time data integration
- [ ] Regime detection (bull/bear markets)
- [ ] Volume profile analysis
- [ ] Sentiment analysis integration
- [ ] Backtesting framework
- [ ] Performance metrics (Sharpe ratio, etc.)

### Potential Extensions
- **Machine Learning**: LSTM/Prophet for forecasting
- **Risk Metrics**: VaR, CVaR calculations
- **Portfolio Optimization**: Efficient frontier
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Fundamental Data**: P/E ratios, earnings
- **News Integration**: Market sentiment analysis
- **Comparison Mode**: Side-by-side symbol comparison
- **Alert System**: Automated shock notifications

---

## 📚 References

### Statistical Methods
- **AIC**: Akaike, H. (1974). "A new look at the statistical model identification". IEEE Transactions on Automatic Control.
- **BIC**: Schwarz, G. (1978). "Estimating the dimension of a model". Annals of Statistics.
- **Model Selection**: Burnham, K. P., & Anderson, D. R. (2002). "Model Selection and Multimodel Inference".

### Time Series Analysis
- **PACF**: Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). "Time Series Analysis: Forecasting and Control".
- **Volatility**: Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity".

### Libraries & Tools
- **Bokeh Documentation**: https://docs.bokeh.org/
- **Bokeh Gallery**: https://bokeh.pydata.org/en/latest/docs/gallery.html
- **statsmodels PACF**: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.pacf.html
- **scikit-learn**: https://scikit-learn.org/
- **Pandas Time Series**: https://pandas.pydata.org/docs/user_guide/timeseries.html

### Datasets
- **Iris Dataset**: Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems".
- **Stock Market Data**: Historical data from Indian stock exchanges (BSE/NSE)

---

## 📝 License & Citation

This project is for educational and research purposes. 

**If you use this code, please cite**:
```
Predictive Analysis Project (2026)
https://github.com/[your-username]/predictive-analysis-project
```

---

## 👤 Author & Contact

For questions, suggestions, or collaborations:
- Open an issue in this repository
- Contact: [Your contact information]

---

## 🙏 Acknowledgments

- **Iris Dataset**: UCI Machine Learning Repository
- **Bokeh Team**: For excellent interactive visualization library
- **statsmodels Contributors**: For robust statistical tools
- **scikit-learn Team**: For machine learning utilities
- **Indian Stock Exchanges**: For historical market data

---

**Last Updated**: February 2026  
**Version**: 2.0  
**Status**: Active Development

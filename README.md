# Stock Price Prediction using Deep Learning Ensemble

A comprehensive stock price prediction system that combines multiple deep learning models (LSTM with Attention, GRU, and Transformer) with sentiment analysis and financial event data to forecast stock prices.

## 📋 Overview

This project implements an ensemble-based stock price prediction system that leverages:
- **Deep Learning Models**: LSTM with Attention, GRU, and Transformer architectures
- **Sentiment Analysis**: FinBERT-based sentiment scoring from news articles
- **Financial Events**: Earnings dates, FOMC meetings, and company-specific events
- **Technical Indicators**: RSI, MACD, Moving Averages, and Volatility metrics

## ✨ Features

- **Multi-Model Ensemble**: Combines predictions from LSTM, GRU, and Transformer models
- **Sentiment Integration**: Uses FinBERT to analyze news sentiment and incorporate it into predictions
- **Event-Based Features**: Considers earnings dates, FOMC meetings, and company events
- **Technical Analysis**: Includes 10+ technical indicators (RSI, MACD, MA10, MA50, Volatility)
- **Multiple Stock Support**: Analyzes both US and Indian stocks
- **Comprehensive Evaluation**: Provides MAE, MSE, and R² metrics along with visualization

## 🏗️ Project Structure

```
RR/
├── FINAL.ipynb          # Main prediction system (AAPL & RELIANCE.NS)
├── AllStock.ipynb       # Batch analysis for 5 Indian stocks
└── README.md            # This file
```

### Files Description

- **FINAL.ipynb**: Contains the finalized prediction system with analyses for:
  - AAPL (Apple Inc.) - US Stock
  - RELIANCE.NS (Reliance Industries) - Indian Stock

- **AllStock.ipynb**: Contains batch analysis for 5 Indian stocks:
  - RELIANCE.NS (Reliance Industries)
  - TVSMOTOR.NS (TVS Motor)
  - ASHOKLEY.NS (Ashok Leyland)
  - INDIACEM.NS (India Cements)
  - MRF.NS (MRF Limited)

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework for neural network implementation
- **Transformers (Hugging Face)**: FinBERT model for financial sentiment analysis
- **scikit-learn**: Data preprocessing and evaluation metrics
- **yfinance**: Stock data fetching
- **pandas/numpy**: Data manipulation and processing
- **matplotlib**: Visualization of predictions
- **NewsAPI**: News article fetching for sentiment analysis
- **tqdm**: Progress bars for training

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Install Dependencies

```bash
pip install yfinance transformers torch scikit-learn requests tqdm matplotlib pandas numpy
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### Required Libraries

- `yfinance` - Stock data fetching
- `transformers` - FinBERT sentiment model
- `torch` - PyTorch deep learning framework
- `scikit-learn` - Preprocessing and metrics
- `requests` - API calls
- `tqdm` - Progress bars
- `matplotlib` - Visualization
- `pandas` - Data manipulation
- `numpy` - Numerical operations

## 🔑 API Keys Setup

### NewsAPI Key

1. Sign up for a free API key at [NewsAPI.org](https://newsapi.org/)
2. In the notebook, replace the API key:
   ```python
   NEWS_API_KEY = 'YOUR_NEWS_API_KEY_HERE'
   ```

**Note**: The current code contains a placeholder API key. Replace it with your own key before use.

## 🚀 Usage

### Running FINAL.ipynb

1. Open `FINAL.ipynb` in Jupyter Notebook or Google Colab
2. Set your NewsAPI key (see API Keys Setup above)
3. Run all cells sequentially
4. The notebook will:
   - Fetch stock data for AAPL and RELIANCE.NS
   - Download and process news articles
   - Calculate sentiment scores using FinBERT
   - Train three deep learning models
   - Generate ensemble predictions
   - Display evaluation metrics and visualizations

### Running AllStock.ipynb

1. Open `AllStock.ipynb` in Jupyter Notebook or Google Colab
2. Set your NewsAPI key
3. Run all cells to analyze all 5 Indian stocks sequentially

## 🧠 Model Architecture

### Ensemble Models

1. **LSTM_Attention** (LSTM with Attention)
   - Multi-layer LSTM with attention mechanism
   - Captures long-term dependencies in stock price sequences

2. **GRU_Model**
   - Gated Recurrent Unit architecture
   - Efficient alternative to LSTM with similar performance

3. **TransformerBlock**
   - Self-attention mechanism
   - Parallel processing capability

### Ensemble Method

The final prediction is an average of all three model predictions:
```python
final_prediction = (lstm_pred + gru_pred + transformer_pred) / 3.0
```

### Training Configuration

- **Sequence Length**: 30 days (lookback period)
- **Batch Size**: 32
- **Epochs**: 30
- **Learning Rate**: 0.0001
- **Train/Test Split**: 85/15
- **Optimizer**: Adam
- **Loss Function**: MSE (Mean Squared Error)

## 📊 Feature Engineering

### Technical Indicators

- **Moving Averages**: MA10, MA50
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **Volatility**: 21-day rolling standard deviation
- **Returns**: Daily percentage returns

### External Features

- **News Sentiment**: FinBERT-based sentiment scores (-1 to +1)
- **Financial Events**: Earnings dates, FOMC meetings, company events
- **Price Features**: Open, High, Low, Close, Volume

## 📈 Evaluation Metrics

The model evaluation includes:

- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual prices
- **MSE** (Mean Squared Error): Average squared difference
- **R² Score**: Coefficient of determination (goodness of fit)

### Sample Output

```
--- Final Ensemble Model Evaluation ---
Mean Absolute Error (MAE): $X.XX
Mean Squared Error (MSE): X.XX
R-squared (R²): X.XXXX
```

## 🎯 Supported Stocks

### US Stocks
- AAPL (Apple Inc.)

### Indian Stocks (NSE)
- RELIANCE.NS (Reliance Industries)
- TVSMOTOR.NS (TVS Motor)
- ASHOKLEY.NS (Ashok Leyland)
- INDIACEM.NS (India Cements)
- MRF.NS (MRF Limited)

**Note**: You can modify the `TICKER` variable in the notebooks to analyze other stocks supported by Yahoo Finance.

## 📝 Data Sources

- **Stock Data**: Yahoo Finance (via `yfinance`)
- **News Articles**: NewsAPI
- **Sentiment Model**: FinBERT (`yiyanghkust/finbert-tone` from Hugging Face)
- **Financial Events**: 
  - Earnings dates from Yahoo Finance
  - FOMC meeting dates (hardcoded for reliability)
  - Company-specific events (e.g., Apple Events, Reliance AGMs)

## ⚙️ Configuration

Key parameters can be adjusted in the notebooks:

```python
SEQ_LEN = 30          # Sequence length (lookback days)
BATCH_SIZE = 32       # Training batch size
EPOCHS = 30           # Number of training epochs
LEARNING_RATE = 0.0001 # Learning rate for optimizer
TICKER = "AAPL"       # Stock ticker symbol
```

## 🔍 Key Features Implementation

### Sentiment Analysis
- Uses FinBERT model specifically trained on financial texts
- Processes news headlines from the last 30 days
- Calculates sentiment scores ranging from -1 (negative) to +1 (positive)

### Event Integration
- Earnings dates automatically fetched from Yahoo Finance
- FOMC meeting dates included for market-wide impact
- Company-specific events (e.g., product launches, AGMs)

### Data Preprocessing
- Min-Max scaling for feature normalization
- Handling of missing values and infinite values
- Feature alignment and merging

## 📊 Visualization

The notebooks generate visualizations showing:
- Actual vs Predicted stock prices
- Training loss curves (during model training)
- Time series plots with date labels

## ⚠️ Important Notes

1. **API Keys**: Replace the NewsAPI key with your own before running
2. **GPU Usage**: Models will automatically use GPU if available (CUDA)
3. **Training Time**: Training can take 30+ minutes depending on hardware
4. **Data Availability**: Requires internet connection for data fetching
5. **Market Hours**: Stock data is only available during market hours

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational and research purposes.

## ⚠️ Disclaimer

This project is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Do not use this system for actual trading without proper risk management and validation.

## 👤 Author

Created as part of stock price prediction research project.

## 🙏 Acknowledgments

- **FinBERT**: [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) for sentiment analysis
- **yfinance**: Yahoo Finance API wrapper
- **PyTorch**: Deep learning framework
- **Hugging Face**: Transformers library

---

**Last Updated**: 2025

For questions or issues, please open an issue on GitHub.


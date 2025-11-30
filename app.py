# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import time

# Set page config
st.set_page_config(
    page_title="Stock Prediction LSTM-PSO",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Configuration constants
BASE_UNITS = 16
BASE_DROPOUT = 0.01
BASE_BATCH = 16
BASE_EPOCHS = 10
BASE_LR = 1e-3

def set_seed(s=42):
    np.random.seed(s)
    tf.random.set_seed(s)

def compute_mape(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    mask = y_true != 0
    if np.any(mask):
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        return np.nan

def build_lstm_model(input_shape, units=16, dropout=0.01, lr=1e-3):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def make_sequences(data, window=1):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def main():
    st.title("📈 Stock Price Prediction with LSTM-PSO")
    
    # Data loading
    st.sidebar.header("Data Configuration")
    data_source = st.sidebar.radio("Data Source", ["Yahoo Finance Ticker", "Use Sample Data"])
    
    df = None
    if data_source == "Yahoo Finance Ticker":
        ticker = st.sidebar.text_input("Stock Ticker", "TLKM.JK")
        if st.sidebar.button("Download Data"):
            with st.spinner("Downloading data..."):
                try:
                    df = yf.download(ticker, period="2y")
                    st.sidebar.success("Data downloaded!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    else:
        if st.sidebar.button("Load Sample Data"):
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            prices = 4000 + np.cumsum(np.random.randn(len(dates)) * 50)
            df = pd.DataFrame({'Close': prices}, index=dates)
            st.sidebar.success("Sample data loaded!")

    if df is not None and not df.empty:
        # Display data
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.tail())
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.index, df['Close'], linewidth=1)
            ax.set_title("Stock Price")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Data Info")
            st.write(f"Records: {len(df)}")
            st.write(f"Mean: {df['Close'].mean():.2f}")

        # Training section
        st.markdown("---")
        st.header("Model Training")
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training..."):
                try:
                    # Preprocessing - AUTOMATIC
                    data = df[['Close']].values
                    n = len(data)
                    n_train = int(n * 0.8)  # 80:20 split
                    
                    scaler = MinMaxScaler().fit(data[:n_train])
                    data_scaled = scaler.transform(data)
                    
                    # Create sequences with window=1
                    X, y = make_sequences(data_scaled, window=1)
                    
                    X_train, y_train = X[:n_train-1], y[:n_train-1]
                    X_test, y_test = X[n_train-1:], y[n_train-1:]
                    
                    X_train = X_train.reshape(X_train.shape[0], 1, 1)
                    X_test = X_test.reshape(X_test.shape[0], 1, 1)
                    
                    # Baseline model
                    set_seed(42)
                    model_base = build_lstm_model((1, 1), BASE_UNITS, BASE_DROPOUT, BASE_LR)
                    history_base = model_base.fit(X_train, y_train, epochs=BASE_EPOCHS, batch_size=BASE_BATCH, verbose=0)
                    
                    # Predictions
                    y_pred_base = model_base.predict(X_test, verbose=0)
                    y_pred_base = scaler.inverse_transform(y_pred_base).flatten()
                    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    
                    mape_base = compute_mape(y_true, y_pred_base)
                    
                    # PSO model (using optimized parameters)
                    best_units = 96
                    best_lr = 0.0847
                    best_batch = 44
                    best_dropout = 0.3
                    
                    set_seed(42)
                    model_pso = build_lstm_model((1, 1), best_units, best_dropout, best_lr)
                    model_pso.fit(X_train, y_train, epochs=100, batch_size=best_batch, verbose=0)
                    
                    y_pred_pso = model_pso.predict(X_test, verbose=0)
                    y_pred_pso = scaler.inverse_transform(y_pred_pso).flatten()
                    mape_pso = compute_mape(y_true, y_pred_pso)
                    
                    # Store results
                    st.session_state.baseline_results = {'mape': mape_base, 'predictions': y_pred_base, 'actual': y_true}
                    st.session_state.final_results = {'mape': mape_pso, 'predictions': y_pred_pso, 'model': model_pso}
                    st.session_state.scaler = scaler
                    st.session_state.model_trained = True
                    
                    st.success("Training completed!")
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")

        # Results
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.header("📊 Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Baseline MAPE", f"{st.session_state.baseline_results['mape']:.4f}%")
            with col2:
                st.metric("PSO-LSTM MAPE", f"{st.session_state.final_results['mape']:.4f}%")
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(st.session_state.baseline_results['actual'], label='Actual', color='blue')
            ax.plot(st.session_state.baseline_results['predictions'], label='Baseline LSTM', color='red', linestyle='--')
            ax.plot(st.session_state.final_results['predictions'], label='PSO-LSTM', color='orange')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    else:
        st.info("👈 Please load data first")

if __name__ == "__main__":
    main()

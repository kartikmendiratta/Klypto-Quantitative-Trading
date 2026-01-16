import pandas as pd
import numpy as np
from .greeks import batch_calculate_greeks

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods)

def calculate_log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(series / series.shift(periods))

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_5'] = calculate_ema(df['close'], 5)
    df['ema_15'] = calculate_ema(df['close'], 15)
    df['ema_50'] = calculate_ema(df['close'], 50)
    
    # Distance from moving averages (stationary)
    df['distance_from_ema5'] = (df['close'] - df['ema_5']) / df['ema_5'] * 100
    df['distance_from_ema15'] = (df['close'] - df['ema_15']) / df['ema_15'] * 100
    df['distance_from_ema50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
    
    # EMA crossover (stationary)
    df['ema_diff'] = df['ema_5'] - df['ema_15']
    df['ema_diff_pct'] = df['ema_diff'] / df['close'] * 100
    
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['avg_iv'] = (df['atm_call_iv'] + df['atm_put_iv']) / 2
    df['iv_spread'] = df['atm_call_iv'] - df['atm_put_iv']
    
    df['pcr_oi'] = df['total_put_oi'] / df['total_call_oi'].replace(0, 1)
    df['pcr_volume'] = df['total_put_volume'] / df['total_call_volume'].replace(0, 1)
    
    df['futures_basis'] = (df['futures_close'] - df['close']) / df['close']
    
    # Log returns (stationary)
    df['log_returns'] = calculate_log_returns(df['close'])
    df['futures_log_returns'] = calculate_log_returns(df['futures_close'])
    
    # Regular returns (for compatibility)
    df['spot_returns'] = calculate_returns(df['close'])
    df['futures_returns'] = calculate_returns(df['futures_close'])
    
    # RSI (stationary)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # ATR normalized (stationary)
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Volatility
    df['volatility_20'] = df['spot_returns'].rolling(20).std() * np.sqrt(252 * 75)
    
    # Price momentum (stationary)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    return df

def add_greeks_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    greeks = batch_calculate_greeks(
        df, 
        spot_col='close',
        strike_col='atm_strike',
        dte_col='days_to_expiry',
        call_iv_col='atm_call_iv',
        put_iv_col='atm_put_iv'
    )
    
    for key, values in greeks.items():
        df[key] = values
    
    df['delta_neutral_ratio'] = abs(df['call_delta']) / abs(df['put_delta']).replace(0, 1)
    df['gamma_exposure'] = df['close'] * df['call_gamma'] * df['atm_call_oi']
    df['net_delta'] = df['call_delta'] * df['atm_call_oi'] + df['put_delta'] * df['atm_put_oi']
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_morning'] = (df['hour'] < 12).astype(int)
    df['is_last_hour'] = (df['hour'] >= 14).astype(int)
    df['time_of_day'] = df['hour'] + df['minute'] / 60
    return df

def add_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 5, 10]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_feature_set(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
    
    df = add_ema_features(df)
    df = add_derived_features(df)
    df = add_greeks_features(df)
    df = add_time_features(df)
    
    # Use stationary features for lags
    lag_cols = ['log_returns', 'rsi', 'avg_iv', 'pcr_oi', 'distance_from_ema15']
    df = add_lag_features(df, lag_cols)
    
    df = df.dropna().reset_index(drop=True)
    
    return df

def get_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        'avg_iv', 'iv_spread', 'pcr_oi', 
        'call_delta', 'call_gamma', 'call_vega',
        'futures_basis', 'spot_returns'
    ]
    return df[feature_cols].copy()

def get_ml_features(df: pd.DataFrame) -> list:
    # Exclude non-stationary raw prices and non-feature columns
    exclude_cols = [
        'timestamp', 'contract_month', 'is_atm', 'strike_offset',
        'open', 'high', 'low', 'close', 'volume',  # Raw spot prices
        'futures_open', 'futures_high', 'futures_low', 'futures_close', 'futures_volume',  # Raw futures prices
        'atm_strike', 'days_to_expiry',  # Non-predictive identifiers
        'ema_5', 'ema_15', 'ema_50',  # Keep distance from MA, exclude raw MA values
        'atm_call_ltp', 'atm_put_ltp',  # Raw option prices
        'signal', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'is_profitable'  # Strategy/target columns
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    return feature_cols

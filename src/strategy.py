import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Position(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position: Position
    regime_at_entry: int
    pnl: Optional[float] = None
    duration_minutes: Optional[int] = None
    features_at_entry: Optional[dict] = None

class EMARegimeStrategy:
    def __init__(self, fast_period: int = 5, slow_period: int = 15):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trades = []
        self.current_position = Position.FLAT
        self.current_trade = None
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        
        df['signal'] = 0
        df.loc[df['ema_cross_up'] & (df['regime'] == 1), 'signal'] = 1
        df.loc[df['ema_cross_down'] & (df['regime'] == -1), 'signal'] = -1
        
        df['exit_long'] = df['ema_cross_down']
        df['exit_short'] = df['ema_cross_up']
        
        return df
    
    def run(self, df: pd.DataFrame, regime_col: str = 'regime') -> list:
        self.trades = []
        self.current_position = Position.FLAT
        self.current_trade = None
        
        df = df.copy()
        if 'ema_fast' not in df.columns:
            df = self.generate_signals(df)
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if self.current_position == Position.FLAT:
                if prev_row['signal'] == 1:
                    self._open_trade(row, Position.LONG, prev_row[regime_col], prev_row)
                elif prev_row['signal'] == -1:
                    self._open_trade(row, Position.SHORT, prev_row[regime_col], prev_row)
                    
            elif self.current_position == Position.LONG:
                if prev_row['exit_long']:
                    self._close_trade(row)
                    
            elif self.current_position == Position.SHORT:
                if prev_row['exit_short']:
                    self._close_trade(row)
        
        if self.current_trade is not None:
            self._close_trade(df.iloc[-1])
        
        return self.trades
    
    def _open_trade(self, row: pd.Series, position: Position, regime: int, feature_row: pd.Series):
        feature_cols = ['avg_iv', 'iv_spread', 'pcr_oi', 'call_delta', 'call_gamma', 
                       'futures_basis', 'atr', 'ema_diff_pct']
        features = {col: feature_row.get(col, 0) for col in feature_cols if col in feature_row.index}
        
        self.current_trade = Trade(
            entry_time=row['timestamp'],
            exit_time=None,
            entry_price=row['open'],
            exit_price=None,
            position=position,
            regime_at_entry=regime,
            features_at_entry=features
        )
        self.current_position = position
    
    def _close_trade(self, row: pd.Series):
        if self.current_trade is None:
            return
            
        self.current_trade.exit_time = row['timestamp']
        self.current_trade.exit_price = row['open']
        
        if self.current_trade.position == Position.LONG:
            self.current_trade.pnl = self.current_trade.exit_price - self.current_trade.entry_price
        else:
            self.current_trade.pnl = self.current_trade.entry_price - self.current_trade.exit_price
        
        duration = (self.current_trade.exit_time - self.current_trade.entry_time).total_seconds() / 60
        self.current_trade.duration_minutes = int(duration)
        
        self.trades.append(self.current_trade)
        self.current_trade = None
        self.current_position = Position.FLAT

def trades_to_dataframe(trades: list) -> pd.DataFrame:
    records = []
    for t in trades:
        records.append({
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'position': t.position.name,
            'regime': t.regime_at_entry,
            'pnl': t.pnl,
            'pnl_pct': (t.pnl / t.entry_price * 100) if t.entry_price else 0,
            'duration_minutes': t.duration_minutes,
            **t.features_at_entry
        })
    return pd.DataFrame(records)

class MLFilteredStrategy(EMARegimeStrategy):
    def __init__(self, model, confidence_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.feature_columns = None
        
    def set_feature_columns(self, columns: list):
        self.feature_columns = columns
        
    def run_with_ml(self, df: pd.DataFrame, regime_col: str = 'regime') -> list:
        self.trades = []
        self.current_position = Position.FLAT
        self.current_trade = None
        
        df = df.copy()
        if 'ema_fast' not in df.columns:
            df = self.generate_signals(df)
        
        if self.feature_columns is not None:
            X = df[self.feature_columns]  # Pass DataFrame, not .values
            predictions = self.model.predict_proba(X)
            if predictions.ndim > 1:
                df['ml_confidence'] = predictions[:, 1]
            else:
                df['ml_confidence'] = predictions
        else:
            df['ml_confidence'] = 1.0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if self.current_position == Position.FLAT:
                if prev_row['signal'] == 1 and prev_row['ml_confidence'] >= self.confidence_threshold:
                    self._open_trade(row, Position.LONG, prev_row[regime_col], prev_row)
                elif prev_row['signal'] == -1 and prev_row['ml_confidence'] >= self.confidence_threshold:
                    self._open_trade(row, Position.SHORT, prev_row[regime_col], prev_row)
                    
            elif self.current_position == Position.LONG:
                if prev_row['exit_long']:
                    self._close_trade(row)
                    
            elif self.current_position == Position.SHORT:
                if prev_row['exit_short']:
                    self._close_trade(row)
        
        if self.current_trade is not None:
            self._close_trade(df.iloc[-1])
        
        return self.trades

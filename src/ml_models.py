import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TradeProfitabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

class XGBoostModel(TradeProfitabilityModel):
    def __init__(self, params: dict = None):
        super().__init__()
        self.params = params or {
            'objective': 'binary:logistic',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'base_score': 0.5
        }
        self.model = xgb.XGBClassifier(**self.params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = self.model.predict(X_val)
            cv_scores.append(accuracy_score(y_val, y_pred))
        
        self.model.fit(X_scaled, y, verbose=False)
        self.is_fitted = True
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance
    
    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_columns = data['feature_columns']
        instance.is_fitted = True
        return instance

class LSTMModel(TradeProfitabilityModel):
    def __init__(self, sequence_length: int = 10, lstm_units: int = 64, dropout: float = 0.2):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        
    def _build_model(self, n_features: int):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout, Input
            from keras.optimizers import Adam
        
        model = Sequential([
            Input(shape=(self.sequence_length, n_features)),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dropout(self.dropout),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])
        
        if y is not None:
            return np.array(sequences), np.array(targets)
        return np.array(sequences), None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> dict:
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)
        
        self.model = self._build_model(X.shape[1])
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        self.is_fitted = True
        
        return {
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history.get('val_accuracy', [0])[-1]
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_seq, _ = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.array([])
        
        proba = self.model.predict(X_seq, verbose=0)
        predictions = (proba > 0.5).astype(int).flatten()
        
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length:] = predictions
        return full_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_seq, _ = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.zeros((len(X), 2))
        
        proba = self.model.predict(X_seq, verbose=0).flatten()
        
        full_proba = np.zeros((len(X), 2))
        full_proba[:, 0] = 1.0
        full_proba[self.sequence_length:, 1] = proba
        full_proba[self.sequence_length:, 0] = 1 - proba
        return full_proba
    
    def save(self, path: str):
        self.model.save(path.replace('.joblib', '_lstm.h5'))
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            from keras.models import load_model
            
        data = joblib.load(path)
        instance = cls(sequence_length=data['sequence_length'])
        instance.scaler = data['scaler']
        instance.feature_columns = data['feature_columns']
        instance.model = load_model(path.replace('.joblib', '_lstm.h5'))
        instance.is_fitted = True
        return instance

def prepare_ml_dataset(feature_df: pd.DataFrame, trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    from .features import get_ml_features
    
    feature_df = feature_df.copy()
    
    # Ensure timestamp columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(feature_df['timestamp']):
        feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])
    
    # Get signal rows (where signal != 0)
    signal_mask = feature_df['signal'] != 0
    signal_rows = feature_df[signal_mask].copy()
    signal_rows['is_profitable'] = 0
    
    # Convert trades entry_time to datetime for comparison
    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    
    # Create a mapping: for each entry_time, find the signal that caused it
    # The signal is generated at time T, trade enters at time T+1 (next bar)
    # So signal_time < entry_time (the bar just before)
    
    for trade_idx, trade in trades_df.iterrows():
        entry_time = trade['entry_time']
        is_profitable = 1 if trade['pnl'] > 0 else 0
        
        # Find the most recent signal before this entry
        # Signal rows that are before entry_time
        candidates = signal_rows[signal_rows['timestamp'] < entry_time]
        
        if len(candidates) > 0:
            # Get the last signal before entry (closest in time)
            signal_idx = candidates.index[-1]
            signal_rows.loc[signal_idx, 'is_profitable'] = is_profitable
    
    # Use get_ml_features to automatically exclude raw prices and non-stationary features
    feature_cols = get_ml_features(signal_rows)
    
    X = signal_rows[feature_cols].fillna(0)
    y = signal_rows['is_profitable']
    
    return X, y

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.5
    
    return metrics

import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeHMM:
    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None
        self.scaler = StandardScaler()
        self.state_mapping = {}
        
    def fit(self, features: pd.DataFrame) -> 'MarketRegimeHMM':
        X = self.scaler.fit_transform(features.values)
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=42
        )
        self.model.fit(X)
        
        states = self.model.predict(X)
        self._map_states_to_regimes(features, states)
        
        return self
    
    def _map_states_to_regimes(self, features: pd.DataFrame, states: np.ndarray):
        state_returns = {}
        returns = features['spot_returns'].values if 'spot_returns' in features.columns else np.zeros(len(states))
        
        for state in range(self.n_states):
            mask = states == state
            state_returns[state] = returns[mask].mean() if mask.sum() > 0 else 0
        
        sorted_states = sorted(state_returns.keys(), key=lambda x: state_returns[x])
        
        self.state_mapping = {
            sorted_states[0]: -1,
            sorted_states[1]: 0,
            sorted_states[2]: 1
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(features.values)
        raw_states = self.model.predict(X)
        return np.array([self.state_mapping.get(s, 0) for s in raw_states])
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(features.values)
        return self.model.predict_proba(X)
    
    def get_transition_matrix(self) -> np.ndarray:
        return self.model.transmat_
    
    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'state_mapping': self.state_mapping
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'MarketRegimeHMM':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.state_mapping = data['state_mapping']
        return instance

def get_regime_statistics(df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
    df_temp = df.copy()
    df_temp['regime'] = regimes
    
    stats = df_temp.groupby('regime').agg({
        'close': ['mean', 'std'],
        'spot_returns': ['mean', 'std', 'count'],
        'avg_iv': 'mean',
        'pcr_oi': 'mean',
        'call_delta': 'mean',
        'call_gamma': 'mean'
    }).round(4)
    
    return stats

def get_regime_durations(regimes: np.ndarray) -> dict:
    durations = {-1: [], 0: [], 1: []}
    clean_regimes = pd.Series(regimes).fillna(0).astype(int).values
    current_regime = clean_regimes[0]
    current_duration = 1
    
    for i in range(1, len(clean_regimes)):
        if clean_regimes[i] == current_regime:
            current_duration += 1
        else:
            if current_regime in durations:
                durations[current_regime].append(current_duration)
            current_regime = clean_regimes[i]
            current_duration = 1
    if current_regime in durations:
        durations[current_regime].append(current_duration)
    
    return durations

def analyze_regimes(df: pd.DataFrame, regimes: np.ndarray) -> dict:
    clean_regimes = pd.Series(regimes).fillna(0).astype(int).values
    regime_counts = pd.Series(clean_regimes).value_counts().to_dict()
    total = len(clean_regimes)
    
    regime_pct = {k: v/total*100 for k, v in regime_counts.items()}
    durations = get_regime_durations(clean_regimes)
    
    avg_durations = {}
    for regime, dur_list in durations.items():
        if len(dur_list) > 0:
            avg_durations[regime] = np.mean(dur_list)
        else:
            avg_durations[regime] = 0
    
    return {
        'counts': regime_counts,
        'percentages': regime_pct,
        'avg_duration': avg_durations,
        'statistics': get_regime_statistics(df, regimes)
    }

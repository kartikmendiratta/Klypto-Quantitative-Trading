import pandas as pd
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class BacktestMetrics:
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    
    def to_dict(self) -> dict:
        return {
            'Total Return': self.total_return,
            'Total Return (%)': self.total_return_pct,
            'Sharpe Ratio': self.sharpe_ratio,
            'Sortino Ratio': self.sortino_ratio,
            'Calmar Ratio': self.calmar_ratio,
            'Max Drawdown': self.max_drawdown,
            'Max Drawdown (%)': self.max_drawdown_pct,
            'Win Rate (%)': self.win_rate,
            'Profit Factor': self.profit_factor,
            'Avg Trade Duration (min)': self.avg_trade_duration,
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Avg Win': self.avg_win,
            'Avg Loss': self.avg_loss
        }

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065, periods_per_year: int = 252*75) -> float:
    if returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.065, periods_per_year: int = 252*75) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside

def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()
    max_dd_pct = (drawdown / running_max).min() * 100
    return abs(max_dd), abs(max_dd_pct)

def calculate_calmar_ratio(total_return: float, max_drawdown: float, periods: int, periods_per_year: int = 252*75) -> float:
    if max_drawdown == 0:
        return 0.0
    annual_return = total_return * (periods_per_year / periods)
    return annual_return / max_drawdown

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 100000) -> BacktestMetrics:
    if len(trades_df) == 0:
        return BacktestMetrics(
            total_return=0, total_return_pct=0, sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, max_drawdown=0, max_drawdown_pct=0, win_rate=0,
            profit_factor=0, avg_trade_duration=0, total_trades=0,
            winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0
        )
    
    pnl = trades_df['pnl'].values
    total_return = pnl.sum()
    total_return_pct = total_return / initial_capital * 100
    
    equity = initial_capital + np.cumsum(pnl)
    equity_series = pd.Series(equity)
    
    returns = pd.Series(pnl) / initial_capital
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    
    max_dd, max_dd_pct = calculate_max_drawdown(equity_series)
    calmar = calculate_calmar_ratio(total_return, max_dd, len(trades_df))
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    avg_duration = trades_df['duration_minutes'].mean()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    
    return BacktestMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_duration=avg_duration,
        total_trades=len(trades_df),
        winning_trades=len(wins),
        losing_trades=len(losses),
        avg_win=avg_win,
        avg_loss=avg_loss
    )

def split_data(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df

def run_backtest(strategy, df: pd.DataFrame, initial_capital: float = 100000) -> Tuple[pd.DataFrame, BacktestMetrics]:
    from .strategy import trades_to_dataframe
    
    trades = strategy.run(df)
    trades_df = trades_to_dataframe(trades)
    metrics = calculate_metrics(trades_df, initial_capital)
    
    return trades_df, metrics

def compare_strategies(results: dict) -> pd.DataFrame:
    comparison = {}
    for name, (trades_df, metrics) in results.items():
        comparison[name] = metrics.to_dict()
    return pd.DataFrame(comparison).T

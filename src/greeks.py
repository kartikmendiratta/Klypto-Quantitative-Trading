import numpy as np
import mibian

RISK_FREE_RATE = 6.5

def calculate_greeks_call(spot: float, strike: float, dte: int, iv: float) -> dict:
    if dte <= 0:
        dte = 1
    if iv <= 0:
        iv = 0.1
    
    try:
        bs = mibian.BS([spot, strike, RISK_FREE_RATE, dte], volatility=iv*100)
        return {
            'delta': bs.callDelta,
            'gamma': bs.gamma,
            'theta': bs.callTheta,
            'vega': bs.vega,
            'rho': bs.callRho
        }
    except:
        return {'delta': 0.5, 'gamma': 0.01, 'theta': -5, 'vega': 10, 'rho': 5}

def calculate_greeks_put(spot: float, strike: float, dte: int, iv: float) -> dict:
    if dte <= 0:
        dte = 1
    if iv <= 0:
        iv = 0.1
    
    try:
        bs = mibian.BS([spot, strike, RISK_FREE_RATE, dte], volatility=iv*100)
        return {
            'delta': bs.putDelta,
            'gamma': bs.gamma,
            'theta': bs.putTheta,
            'vega': bs.vega,
            'rho': bs.putRho
        }
    except:
        return {'delta': -0.5, 'gamma': 0.01, 'theta': -5, 'vega': 10, 'rho': -5}

def calculate_iv_from_price(spot: float, strike: float, dte: int, price: float, option_type: str = 'call') -> float:
    if dte <= 0:
        dte = 1
    if price <= 0:
        return 0.15
    
    try:
        if option_type.lower() == 'call':
            bs = mibian.BS([spot, strike, RISK_FREE_RATE, dte], callPrice=price)
        else:
            bs = mibian.BS([spot, strike, RISK_FREE_RATE, dte], putPrice=price)
        iv = bs.impliedVolatility / 100
        return max(0.05, min(1.0, iv))
    except:
        return 0.15

def batch_calculate_greeks(df, spot_col: str, strike_col: str, dte_col: str, 
                           call_iv_col: str, put_iv_col: str) -> dict:
    n = len(df)
    result = {
        'call_delta': np.zeros(n),
        'call_gamma': np.zeros(n),
        'call_theta': np.zeros(n),
        'call_vega': np.zeros(n),
        'call_rho': np.zeros(n),
        'put_delta': np.zeros(n),
        'put_gamma': np.zeros(n),
        'put_theta': np.zeros(n),
        'put_vega': np.zeros(n),
        'put_rho': np.zeros(n)
    }
    
    for i in range(n):
        spot = df.iloc[i][spot_col]
        strike = df.iloc[i][strike_col]
        dte = df.iloc[i][dte_col]
        call_iv = df.iloc[i][call_iv_col]
        put_iv = df.iloc[i][put_iv_col]
        
        call_greeks = calculate_greeks_call(spot, strike, dte, call_iv)
        put_greeks = calculate_greeks_put(spot, strike, dte, put_iv)
        
        result['call_delta'][i] = call_greeks['delta']
        result['call_gamma'][i] = call_greeks['gamma']
        result['call_theta'][i] = call_greeks['theta']
        result['call_vega'][i] = call_greeks['vega']
        result['call_rho'][i] = call_greeks['rho']
        
        result['put_delta'][i] = put_greeks['delta']
        result['put_gamma'][i] = put_greeks['gamma']
        result['put_theta'][i] = put_greeks['theta']
        result['put_vega'][i] = put_greeks['vega']
        result['put_rho'][i] = put_greeks['rho']
    
    return result

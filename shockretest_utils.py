import numpy as np

def calc_atr(close, high, low, window):
    tr = np.maximum(high - low, np.maximum(abs(close - close.shift(1)), abs(high - close.shift(1))))
    return tr.rolling(window).mean()

def calc_adx(high, low, close, window):
    # Placeholder ADX (implementación real recomendada)
    return np.random.uniform(10, 45, len(close))

def calc_avwap(prices, volumes):
    return np.average(prices, weights=volumes)

def detect_fvg(high, low, prev_high, prev_low, min_gap):
    gap_up = low - prev_high
    gap_down = prev_low - high
    if gap_up > min_gap:
        return gap_up
    if gap_down > min_gap:
        return gap_down
    return None

def calc_halfback(open_, close):
    return open_ + np.sign(close - open_) * abs(close - open_) / 2

def swing_detection(close, window=3):
    arr = np.array(close)
    if len(arr) < window + 1:
        return [False]
    diff = np.sign(arr[-1] - arr[-window - 1])
    prev_diff = np.sign(arr[-window - 1] - arr[-window - 2])
    return [diff != prev_diff]

def is_shock(open_, close, body, atr, abs_ret, adx=None):
    from shockretest_config import SHOCK_ABS_RET_THRESHOLD, SHOCK_BODY_MULT_ATR, ADX_HIGH_THRESHOLD
    if abs_ret >= SHOCK_ABS_RET_THRESHOLD or body >= SHOCK_BODY_MULT_ATR * atr:
        if adx is not None and adx > ADX_HIGH_THRESHOLD:
            return False
        return True
    return False

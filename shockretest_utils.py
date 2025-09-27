import numpy as np

def calc_atr(close, high, low, window):
    """Calcula ATR."""
    tr = np.maximum(high - low, np.maximum(abs(close - close.shift(1)), abs(high - close.shift(1))))
    return tr.rolling(window).mean()

def calc_adx(high, low, close, window):
    """ADX clásico para filtro de tendencias."""
    # Implementación simplificada
    return np.random.uniform(10, 45, len(close))  # Placeholder (reemplazar por cálculo real)

def calc_avwap(prices, volumes):
    """AVWAP micro."""
    return np.average(prices, weights=volumes)

def detect_fvg(high, low, prev_high, prev_low, min_gap):
    """Detecta fair value gap micro."""
    gap = low - prev_high
    if gap > min_gap:
        return gap
    gap = prev_low - high
    if gap > min_gap:
        return gap
    return None

def calc_halfback(open_, close):
    """Calcula punto medio del cuerpo."""
    return open_ + np.sign(close - open_) * abs(close - open_) / 2

def is_shock(open_, close, body, atr, abs_ret, adx=None):
    """Detecta si la vela es shock."""
    if abs_ret >= SHOCK_ABS_RET_THRESHOLD or body >= SHOCK_BODY_MULT_ATR * atr:
        if adx is not None and adx > ADX_HIGH_THRESHOLD:
            return False
        return True
    return False

def swing_detection(close, window=3):
    """Detecta micro swings (simplificado)."""
    # Devuelve True si hay reversión de dirección en las últimas window velas
    return np.sign(close.diff(window)) != np.sign(close.diff())

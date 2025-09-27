import pandas as pd
from shockretest_config import *
from shockretest_utils import *

class ShockRetestStrategy:
    def __init__(self):
        self.last_shock = None
        self.shock_window = []  # Guarda shocks recientes para retest
        self.retest_events = []
        self.signals = []

    def on_new_candle(self, candle, df_1m, df_15m=None):
        """Llama cada vez que hay una nueva vela 1m. candle = dict con 'open','high','low','close','volume','timestamp'"""
        open_, high, low, close, volume = candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
        idx = candle['timestamp']

        # ATR y ADX
        atr = calc_atr(df_1m['close'], df_1m['high'], df_1m['low'], SHOCK_ATR_WINDOW)[-1]
        adx = None
        if df_15m is not None:
            adx = calc_adx(df_15m['high'], df_15m['low'], df_15m['close'], ADX_FILTER_WINDOW)[-1]

        # Detectar shock
        abs_ret = abs(close - open_) / open_
        body = abs(close - open_)
        is_up = close > open_
        is_down = close < open_
        if is_shock(open_, close, body, atr, abs_ret, adx):
            shock_event = {
                'timestamp': idx,
                'open': open_,
                'close': close,
                'high': high,
                'low': low,
                'volume': volume,
                'body': body,
                'atr': atr,
                'abs_ret': abs_ret,
                'direction': 'up' if is_up else 'down'
            }
            self.last_shock = shock_event
            self.shock_window.append(shock_event)
            # Calcula zonas de retest
            shock_event['halfback'] = calc_halfback(open_, close)
            shock_event['avwap'] = calc_avwap(df_1m['close'][-AVWAP_WINDOW:], df_1m['volume'][-AVWAP_WINDOW:])
            shock_event['open'] = open_
            # FVG (opcional)
            prev_candle = df_1m.iloc[-2]
            fvg = detect_fvg(high, low, prev_candle['high'], prev_candle['low'], FVG_MIN_GAP)
            if fvg:
                shock_event['fvg'] = fvg

        # Proceso de retest
        self.check_retest(candle, df_1m)

    def check_retest(self, candle, df_1m):
        """Chequea si hay retest de zonas dentro de la ventana T"""
        for shock_event in list(self.shock_window):
            # Retest window
            if candle['timestamp'] - shock_event['timestamp'] > RETEST_WINDOW * 60:
                self.shock_window.remove(shock_event)
                continue

            # Check zones
            touched_zones = []
            if self.zone_touched(candle['low'], candle['high'], shock_event['halfback']):
                touched_zones.append('halfback')
            if self.zone_touched(candle['low'], candle['high'], shock_event['open']):
                touched_zones.append('open')
            if self.zone_touched(candle['low'], candle['high'], shock_event['avwap']):
                touched_zones.append('avwap')
            if 'fvg' in shock_event and self.zone_touched(candle['low'], candle['high'], shock_event['fvg']):
                touched_zones.append('fvg')

            if touched_zones:
                # Trigger fade o continuación
                setup = self.setup_decision(candle, shock_event, touched_zones, df_1m)
                if setup:
                    self.send_signal(setup, shock_event, candle)
                self.shock_window.remove(shock_event)

    def zone_touched(self, low, high, zone):
        """Chequea si la zona fue tocada"""
        return low <= zone <= high

    def setup_decision(self, candle, shock_event, touched_zones, df_1m):
        """Decide fade o continuación según trigger"""
        # Simplificación: Si se toca HB y hay reversión de swing, fade. Si se toca HB y retoma dirección, continuación.
        close = candle['close']
        # Swing detection
        fade_trigger = swing_detection(df_1m['close'], window=3)[-1]
        if 'halfback' in touched_zones and fade_trigger:
            # Fade: reversión a HB con swing contrario al shock
            signal = {
                "tipo": "FADE_SHOCK",
                "timestamp": candle['timestamp'],
                "precio": shock_event['halfback'],
                "cancel_all_limits": True,
                "adjust_tp": True,
                "adjust_sl": True,
                "nuevo_grid": True,
                "override": {
                    "min_grid_spacing": 0.0008,
                    "max_grid_spacing": 0.0015,
                    "grid_range_min": 0.002,
                    "grid_range_max": 0.005
                },
                "tp": [shock_event['halfback'], shock_event['open']],
                "sl": shock_event['high'] + SL_ATR_BUFFER * shock_event['atr'] if shock_event['direction'] == 'up' else shock_event['low'] - SL_ATR_BUFFER * shock_event['atr']
            }
            return signal

        # Continuación: retest HB y retoma dirección original
        if 'halfback' in touched_zones and not fade_trigger:
            signal = {
                "tipo": "CONTINUATION_SHOCK",
                "timestamp": candle['timestamp'],
                "precio": shock_event['close'],
                "cancel_all_limits": True,
                "adjust_tp": True,
                "adjust_sl": True,
                "nuevo_grid": True,
                "override": {
                    "min_grid_spacing": 0.0015,
                    "max_grid_spacing": 0.0025,
                    "grid_range_min": 0.003,
                    "grid_range_max": 0.009
                },
                "tp": [shock_event['close'] + TP_BODY_MULT * abs(shock_event['close'] - shock_event['open'])],
                "sl": candle['low'] - SL_ATR_BUFFER * shock_event['atr'] if shock_event['direction'] == 'up' else candle['high'] + SL_ATR_BUFFER * shock_event['atr']
            }
            return signal
        return None

    def send_signal(self, signal, shock_event, candle):
        """Envía la señal estructurada al GridManager"""
        self.signals.append(signal)
        # Aquí deberías llamar a grid_manager.handle_signal(signal, candle['close'])
        print(f"[ShockRetestStrategy] Enviando señal: {signal['tipo']} a precio {signal['precio']} (timestamp {signal['timestamp']})")

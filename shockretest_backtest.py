import pandas as pd
from strategy_shockretest import ShockRetestStrategy

def run_backtest(df_1m, df_15m=None):
    """Corre el módulo sobre históricos y reporta estadísticas clave"""
    strat = ShockRetestStrategy()
    for i in range(len(df_1m)):
        candle = {
            'open': df_1m.iloc[i]['open'],
            'high': df_1m.iloc[i]['high'],
            'low': df_1m.iloc[i]['low'],
            'close': df_1m.iloc[i]['close'],
            'volume': df_1m.iloc[i]['volume'],
            'timestamp': int(df_1m.index[i].timestamp())
        }
        # Opcional: proveer df_15m para ADX/trend
        strat.on_new_candle(candle, df_1m.iloc[:i+1], df_15m)

    # Resultados
    print(f"Total shocks detectados: {len(strat.signals)}")
    fade_count = sum(1 for s in strat.signals if s['tipo'] == 'FADE_SHOCK')
    cont_count = sum(1 for s in strat.signals if s['tipo'] == 'CONTINUATION_SHOCK')
    print(f"Fade setups: {fade_count}, Continuation setups: {cont_count}")

    # Puedes agregar más estadísticas aquí

import matplotlib.pyplot as plt

def plot_shock_event(df_1m, shock_event, signal=None):
    idx = shock_event['timestamp']
    df = df_1m.set_index('timestamp')
    start_idx = idx - 30*60
    end_idx = idx + 30*60
    plot_df = df.loc[start_idx:end_idx]
    plt.figure(figsize=(10,4))
    plt.plot(plot_df['close'], label='Close')
    plt.axvline(x=idx, color='red', linestyle='--', label='Shock')
    plt.axhline(y=shock_event['halfback'], color='blue', linestyle='-', label='Halfback')
    plt.axhline(y=shock_event['open'], color='green', linestyle='--', label='Open')
    if signal:
        plt.axhline(y=signal['precio'], color='purple', linestyle='-.', label='Signal Price')
    plt.legend()
    plt.show()

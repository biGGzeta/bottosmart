import matplotlib.pyplot as plt

def plot_shock_event(df_1m, shock_event, signal=None):
    idx = shock_event['timestamp']
    start_idx = max(0, idx - 30)
    end_idx = min(len(df_1m)-1, idx + 30)
    plt.figure(figsize=(10,4))
    plt.plot(df_1m['close'][start_idx:end_idx], label='Close')
    plt.axvline(x=idx, color='red', linestyle='--', label='Shock')
    plt.axhline(y=shock_event['halfback'], color='blue', linestyle='-', label='Halfback')
    plt.axhline(y=shock_event['open'], color='green', linestyle='--', label='Open')
    if signal:
        plt.axhline(y=signal['precio'], color='purple', linestyle='-.', label='Signal Price')
    plt.legend()
    plt.show()

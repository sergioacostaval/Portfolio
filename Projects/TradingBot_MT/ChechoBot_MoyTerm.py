import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Descargar datos ---
def get_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Télécharge les données historiques depuis Yahoo Finance"""
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    data = data[['Close']].copy()
    return data

# --- Generar señales de compra con bloqueo hasta tocar el último máximo usado ---
def generate_buy_signals_blocked(df: pd.DataFrame, window: int = 365, dip_pct: float = 0.2) -> pd.DataFrame:
    """
    Genera señales de compra cuando Close <=  (rolling_max * (1 - dip_pct)),
    pero una vez que se genera una compra, no genera otra nueva hasta que el precio
    toque o supere el máximo (rolling_max) que se usó para esa compra.
    """
    df = df.copy()
    close = df['Close'].squeeze()  # Serie unidimensional

    rolling_max = close.rolling(window).max()
    buy_level = rolling_max * (1 - dip_pct)

    signals = [0] * len(df)  # array de 0/1 para señales de compra
    last_max_used = None     # máximo que bloquea nuevas compras hasta que se toque

    for i in range(len(df)):
        rm = rolling_max.iloc[i]
        bl = buy_level.iloc[i]
        cp = close.iloc[i]

        # saltar filas sin history suficiente
        if pd.isna(rm) or pd.isna(bl) or pd.isna(cp):
            continue

        # desbloquear si el precio toca o supera el máximo que estaba bloqueando
        if last_max_used is not None and cp >= last_max_used:
            last_max_used = None

        # si no está bloqueado y estamos por debajo del nivel de compra -> generar señal
        if last_max_used is None and cp <= bl:
            signals[i] = 1
            last_max_used = rm  # fijamos el máximo que debe tocarse para permitir la próxima compra

    df_out = pd.DataFrame({
        'Close': close,
        'rolling_max': rolling_max,
        'buy_level': buy_level,
        'signal': pd.Series(signals, index=df.index)
    }, index=df.index)

    # position igual a signal para mantener compatibilidad con tu backtest/vista
    df_out['position'] = df_out['signal']

    return df_out

# --- Backtest sencillo (sólo compra) ---
def backtest(df: pd.DataFrame, initial_capital: float = 10000.0):
    capital = initial_capital
    position = 0
    trade_log = []

    for row in df.itertuples():
        price = row.Close
        position_signal = row.position

        if position_signal == 1 and position == 0:  # Comprar
            position = capital // price
            capital -= position * price
            trade_log.append((row.Index, 'BUY', position, price, capital))

    final_value = capital + position * df['Close'].iloc[-1]
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'profit': final_value - initial_capital,
        'trades': trade_log
    }

# --- Graficar ---
def plot_trades(df: pd.DataFrame):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['rolling_max'], label='Rolling Max 365')
    plt.plot(df['buy_level'], linestyle='--', label='Buy Level (20% dip)')
    buys = df.index[df['signal'] == 1]
    plt.scatter(buys, df.loc[buys, 'Close'], marker='^', color='g', label='Buy')
    plt.title("Dip Strategy: buy 20% below 365-day high (blocked until last max touched)")
    plt.legend()
    plt.show()

# --- Ejecutar ---
if __name__ == "__main__":
    ticker = "AAPL"
    df = get_data(ticker, period="5y")
    df = generate_buy_signals_blocked(df, window=365, dip_pct=0.2)
    results = backtest(df, initial_capital=10000)

    print("Initial:", results['initial_capital'])
    print("Final:", round(results['final_value'], 2))
    print("Profit:", round(results['profit'], 2))
    print("Trades:")
    for t in results['trades']:
        print(t)

    plot_trades(df)



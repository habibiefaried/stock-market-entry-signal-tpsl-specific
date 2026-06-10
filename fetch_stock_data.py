import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse


def fetch_ohlcv(ticker: str, months: int = 3, warmup_days: int = 300) -> pd.DataFrame:
    """Fetch OHLCV data with extra warmup days so indicators can fully populate."""
    obj = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30 + warmup_days)
    df = obj.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving Averages
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()

    # RSI (14 and 7)
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20
    df["BB_Pct"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # Stochastic Oscillator
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ADX (Average Directional Index)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr14 = compute_atr(df, 14)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    df["ADX"] = dx.rolling(14).mean()
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di

    # CCI (Commodity Channel Index)
    tp = (high + low + close) / 3
    df["CCI"] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())

    # OBV (On Balance Volume)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_EMA"] = obv.ewm(span=20, adjust=False).mean()

    # Williams %R
    df["Williams_R"] = -100 * (high14 - close) / (high14 - low14)

    # MFI (Money Flow Index)
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf))
    df["MFI"] = mfi

    # Momentum
    df["Momentum_5"] = close / close.shift(5) - 1
    df["Momentum_10"] = close / close.shift(10) - 1

    # Volatility measures
    df["Volatility_5d"] = close.pct_change().rolling(5).std() * 100
    df["Volatility_10d"] = close.pct_change().rolling(10).std() * 100
    df["Volatility_20d"] = close.pct_change().rolling(20).std() * 100

    # Price position relative to MAs
    df["Close_SMA20_Ratio"] = (close - df["SMA_20"]) / df["SMA_20"]
    df["Close_SMA50_Ratio"] = (close - df["SMA_50"]) / df["SMA_50"]

    # Candle body and wick ratios
    body = (close - df["Open"]).abs()
    full_range = high - low
    df["Body_Ratio"] = body / full_range
    df["Upper_Wick_Ratio"] = (high - pd.concat([close, df["Open"]], axis=1).max(axis=1)) / full_range
    df["Lower_Wick_Ratio"] = (pd.concat([close, df["Open"]], axis=1).min(axis=1) - low) / full_range

    # VWAP approximation (daily reset not applicable, use rolling)
    df["VWAP_10"] = (volume * (high + low + close) / 3).rolling(10).sum() / volume.rolling(10).sum()

    # Choppiness Index
    atr_sum = compute_atr(df, 1).rolling(14).sum()
    highest = high.rolling(14).max()
    lowest = low.rolling(14).min()
    df["Choppiness"] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(14)

    # Rate of Change
    df["ROC_5"] = (close - close.shift(5)) / close.shift(5) * 100
    df["ROC_10"] = (close - close.shift(10)) / close.shift(10) * 100

    # === Additional Moving Averages ===
    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_200"] = close.rolling(200).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # Hull Moving Average (HMA) - faster MA with less lag
    wma_half = close.rolling(9).apply(lambda x: np.sum(x * np.arange(1, 10)) / 45, raw=True)
    wma_full = close.rolling(18).apply(lambda x: np.sum(x * np.arange(1, 19)) / 171, raw=True)
    hma_raw = 2 * wma_half - wma_full
    df["HMA_18"] = hma_raw.rolling(4).apply(lambda x: np.sum(x * np.arange(1, 5)) / 10, raw=True)

    # MA Crossover signals (distance between fast/slow)
    df["EMA9_EMA21_Diff"] = df["EMA_9"] - df["EMA_21"]
    df["SMA20_SMA50_Diff"] = df["SMA_20"] - df["SMA_50"]
    df["SMA50_SMA200_Diff"] = df["SMA_50"] - df["SMA_200"]

    # === Ichimoku Cloud ===
    high9 = high.rolling(9).max()
    low9 = low.rolling(9).min()
    df["Ichimoku_Tenkan"] = (high9 + low9) / 2
    high26 = high.rolling(26).max()
    low26 = low.rolling(26).min()
    df["Ichimoku_Kijun"] = (high26 + low26) / 2
    df["Ichimoku_Senkou_A"] = ((df["Ichimoku_Tenkan"] + df["Ichimoku_Kijun"]) / 2).shift(26)
    high52 = high.rolling(52).max()
    low52 = low.rolling(52).min()
    df["Ichimoku_Senkou_B"] = ((high52 + low52) / 2).shift(26)
    df["Ichimoku_Cloud_Width"] = df["Ichimoku_Senkou_A"] - df["Ichimoku_Senkou_B"]
    df["Price_vs_Cloud"] = close - pd.concat([df["Ichimoku_Senkou_A"], df["Ichimoku_Senkou_B"]], axis=1).max(axis=1)

    # === Parabolic SAR (simplified) ===
    psar = close.copy()
    af = 0.02
    af_step = 0.02
    af_max = 0.2
    bull = True
    ep = low.iloc[0]
    psar.iloc[0] = high.iloc[0]
    for i in range(1, len(close)):
        if bull:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
    df["PSAR"] = psar
    df["PSAR_Dist"] = (close - psar) / close * 100

    # === Keltner Channel ===
    keltner_mid = close.ewm(span=20, adjust=False).mean()
    keltner_atr = compute_atr(df, 10)
    df["Keltner_Upper"] = keltner_mid + 2 * keltner_atr
    df["Keltner_Lower"] = keltner_mid - 2 * keltner_atr
    df["Keltner_Pct"] = (close - df["Keltner_Lower"]) / (df["Keltner_Upper"] - df["Keltner_Lower"])

    # === Donchian Channel ===
    df["Donchian_Upper"] = high.rolling(20).max()
    df["Donchian_Lower"] = low.rolling(20).min()
    df["Donchian_Mid"] = (df["Donchian_Upper"] + df["Donchian_Lower"]) / 2
    df["Donchian_Pct"] = (close - df["Donchian_Lower"]) / (df["Donchian_Upper"] - df["Donchian_Lower"])

    # === Awesome Oscillator ===
    midprice = (high + low) / 2
    df["AO"] = midprice.rolling(5).mean() - midprice.rolling(34).mean()
    df["AO_Signal"] = df["AO"].rolling(5).mean()

    # === Detrended Price Oscillator (DPO) ===
    shift_period = 11
    df["DPO"] = close - close.rolling(20).mean().shift(shift_period)

    # === Coppock Curve ===
    roc14 = (close - close.shift(14)) / close.shift(14) * 100
    roc11 = (close - close.shift(11)) / close.shift(11) * 100
    df["Coppock"] = (roc14 + roc11).ewm(span=10, adjust=False).mean()

    # === TRIX (Triple Exponential Average) ===
    ema1 = close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    df["TRIX"] = ema3.pct_change() * 100
    df["TRIX_Signal"] = df["TRIX"].ewm(span=9, adjust=False).mean()

    # === Elder Ray (Bull/Bear Power) ===
    ema13 = close.ewm(span=13, adjust=False).mean()
    df["Bull_Power"] = high - ema13
    df["Bear_Power"] = low - ema13

    # === Mass Index ===
    hl_diff = high - low
    ema_hl = hl_diff.ewm(span=9, adjust=False).mean()
    ema_ema_hl = ema_hl.ewm(span=9, adjust=False).mean()
    ratio = ema_hl / ema_ema_hl
    df["Mass_Index"] = ratio.rolling(25).sum()

    # === Vortex Indicator ===
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["Vortex_Pos"] = vm_plus.rolling(14).sum() / tr.rolling(14).sum()
    df["Vortex_Neg"] = vm_minus.rolling(14).sum() / tr.rolling(14).sum()
    df["Vortex_Diff"] = df["Vortex_Pos"] - df["Vortex_Neg"]

    # === Ultimate Oscillator ===
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr_uo = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum()
    df["Ultimate_Osc"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    # === Know Sure Thing (KST) ===
    roc10 = close.pct_change(10) * 100
    roc15 = close.pct_change(15) * 100
    roc20 = close.pct_change(20) * 100
    roc30 = close.pct_change(30) * 100
    df["KST"] = (roc10.rolling(10).mean() * 1 + roc15.rolling(10).mean() * 2 +
                 roc20.rolling(10).mean() * 3 + roc30.rolling(15).mean() * 4)
    df["KST_Signal"] = df["KST"].rolling(9).mean()

    # === Squeeze Momentum (BB inside Keltner = squeeze) ===
    df["BB_Squeeze"] = ((df["BB_Lower"] > df["Keltner_Lower"]) & (df["BB_Upper"] < df["Keltner_Upper"])).astype(int)

    # === Volume indicators ===
    df["Volume_SMA_20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"]
    df["VROC"] = (volume - volume.shift(5)) / volume.shift(5)
    # Accumulation/Distribution Line
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    df["AD_Line"] = (clv * volume).cumsum()
    df["AD_Line_EMA"] = df["AD_Line"].ewm(span=20, adjust=False).mean()
    # Chaikin Money Flow
    df["CMF"] = (clv * volume).rolling(20).sum() / volume.rolling(20).sum()
    # Force Index
    df["Force_Index"] = close.diff() * volume
    df["Force_Index_EMA"] = df["Force_Index"].ewm(span=13, adjust=False).mean()

    # === Price action derived ===
    df["Price_Change_1d"] = close.pct_change(1) * 100
    df["Price_Change_3d"] = close.pct_change(3) * 100
    df["Price_Change_5d"] = close.pct_change(5) * 100
    df["HL_Range_Pct"] = (high - low) / close * 100
    df["Gap"] = (df["Open"] - close.shift(1)) / close.shift(1) * 100

    # === Candlestick Patterns (binary 0/1) ===
    open_price = df["Open"]
    body = close - open_price
    body_abs = body.abs()
    upper_wick = high - pd.concat([close, open_price], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_price], axis=1).min(axis=1) - low

    # Doji: body is very small relative to range
    df["Doji"] = (body_abs <= full_range * 0.1).astype(int)

    # Hammer: small body at top, long lower wick
    df["Hammer"] = ((lower_wick >= body_abs * 2) & (upper_wick <= body_abs * 0.5) & (body_abs > 0)).astype(int)

    # Inverted Hammer: small body at bottom, long upper wick
    df["Inverted_Hammer"] = ((upper_wick >= body_abs * 2) & (lower_wick <= body_abs * 0.5) & (body_abs > 0)).astype(int)

    # Shooting Star: like inverted hammer but after uptrend
    uptrend = close.shift(1) > close.shift(3)
    df["Shooting_Star"] = ((upper_wick >= body_abs * 2) & (lower_wick <= body_abs * 0.5) & uptrend & (body_abs > 0)).astype(int)

    # Hanging Man: like hammer but after uptrend
    df["Hanging_Man"] = ((lower_wick >= body_abs * 2) & (upper_wick <= body_abs * 0.5) & uptrend & (body_abs > 0)).astype(int)

    # Bullish Engulfing
    prev_body = body.shift(1)
    df["Bullish_Engulfing"] = ((body > 0) & (prev_body < 0) & (body_abs > body_abs.shift(1)) &
                               (open_price <= close.shift(1)) & (close >= open_price.shift(1))).astype(int)

    # Bearish Engulfing
    df["Bearish_Engulfing"] = ((body < 0) & (prev_body > 0) & (body_abs > body_abs.shift(1)) &
                               (open_price >= close.shift(1)) & (close <= open_price.shift(1))).astype(int)

    # Morning Star (3-candle bullish reversal)
    small_body_mid = body_abs.shift(1) <= full_range.shift(1) * 0.3
    df["Morning_Star"] = ((body.shift(2) < 0) & small_body_mid & (body > 0) &
                          (close > (open_price.shift(2) + close.shift(2)) / 2)).astype(int)

    # Evening Star (3-candle bearish reversal)
    df["Evening_Star"] = ((body.shift(2) > 0) & small_body_mid & (body < 0) &
                          (close < (open_price.shift(2) + close.shift(2)) / 2)).astype(int)

    # Three White Soldiers
    bullish_candle = body > 0
    df["Three_White_Soldiers"] = (bullish_candle & bullish_candle.shift(1) & bullish_candle.shift(2) &
                                  (close > close.shift(1)) & (close.shift(1) > close.shift(2))).astype(int)

    # Three Black Crows
    bearish_candle = body < 0
    df["Three_Black_Crows"] = (bearish_candle & bearish_candle.shift(1) & bearish_candle.shift(2) &
                               (close < close.shift(1)) & (close.shift(1) < close.shift(2))).astype(int)

    # Spinning Top: small body, similar wicks on both sides
    df["Spinning_Top"] = ((body_abs <= full_range * 0.3) & (upper_wick >= full_range * 0.25) &
                          (lower_wick >= full_range * 0.25)).astype(int)

    # Marubozu: full body, no/tiny wicks
    df["Marubozu"] = ((body_abs >= full_range * 0.9) & (full_range > 0)).astype(int)

    # Dragonfly Doji: doji with long lower wick
    df["Dragonfly_Doji"] = ((body_abs <= full_range * 0.1) & (lower_wick >= full_range * 0.6) &
                            (full_range > 0)).astype(int)

    # Gravestone Doji: doji with long upper wick
    df["Gravestone_Doji"] = ((body_abs <= full_range * 0.1) & (upper_wick >= full_range * 0.6) &
                             (full_range > 0)).astype(int)

    # Piercing Line
    df["Piercing_Line"] = ((body.shift(1) < 0) & (body > 0) & (open_price < low.shift(1)) &
                           (close > (open_price.shift(1) + close.shift(1)) / 2) &
                           (close < open_price.shift(1))).astype(int)

    # Dark Cloud Cover
    df["Dark_Cloud"] = ((body.shift(1) > 0) & (body < 0) & (open_price > high.shift(1)) &
                        (close < (open_price.shift(1) + close.shift(1)) / 2) &
                        (close > open_price.shift(1))).astype(int)

    # === Trend strength features ===
    df["Trend_5d"] = np.sign(close - close.shift(5))
    df["Trend_10d"] = np.sign(close - close.shift(10))
    df["Trend_20d"] = np.sign(close - close.shift(20))
    df["Higher_High"] = (high > high.shift(1)).astype(int)
    df["Lower_Low"] = (low < low.shift(1)).astype(int)
    df["Consecutive_Up"] = bullish_candle.astype(int).groupby((~bullish_candle).cumsum()).cumsum()
    df["Consecutive_Down"] = bearish_candle.astype(int).groupby((~bearish_candle).cumsum()).cumsum()

    # === Lag features (for model memory) ===
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = close.shift(lag)
        df[f"Volume_Lag_{lag}"] = volume.shift(lag)
        df[f"RSI14_Lag_{lag}"] = df["RSI_14"].shift(lag)

    # === Slope / acceleration features ===
    df["RSI14_Slope_3d"] = df["RSI_14"] - df["RSI_14"].shift(3)
    df["MACD_Accel"] = df["MACD_Hist"] - df["MACD_Hist"].shift(1)
    df["ADX_Slope_3d"] = df["ADX"] - df["ADX"].shift(3)
    df["ATR_Change"] = (df["ATR"] - df["ATR"].shift(5)) / df["ATR"].shift(5)

    # === Support/Resistance proximity ===
    df["Dist_High_20"] = (high.rolling(20).max() - close) / close * 100
    df["Dist_Low_20"] = (close - low.rolling(20).min()) / close * 100
    df["Dist_High_50"] = (high.rolling(50).max() - close) / close * 100
    df["Dist_Low_50"] = (close - low.rolling(50).min()) / close * 100

    # === Chande Momentum Oscillator (CMO) ===
    delta = close.diff()
    gain_sum = delta.where(delta > 0, 0.0).rolling(14).sum()
    loss_sum = (-delta).where(delta < 0, 0.0).rolling(14).sum()
    df["CMO"] = 100 * (gain_sum - loss_sum) / (gain_sum + loss_sum)

    # === Stochastic RSI ===
    rsi14 = df["RSI_14"]
    rsi_min = rsi14.rolling(14).min()
    rsi_max = rsi14.rolling(14).max()
    df["Stoch_RSI"] = (rsi14 - rsi_min) / (rsi_max - rsi_min) * 100
    df["Stoch_RSI_K"] = df["Stoch_RSI"].rolling(3).mean()
    df["Stoch_RSI_D"] = df["Stoch_RSI_K"].rolling(3).mean()

    # === Smoothed Rate of Change (SROC) ===
    ema13_close = close.ewm(span=13, adjust=False).mean()
    df["SROC"] = (ema13_close - ema13_close.shift(21)) / ema13_close.shift(21) * 100

    # === MACD Percentage Price Oscillator (PPO) ===
    df["PPO"] = (df["MACD"] / ema26) * 100
    df["PPO_Signal"] = df["PPO"].ewm(span=9, adjust=False).mean()
    df["PPO_Hist"] = df["PPO"] - df["PPO_Signal"]

    # === Linear Regression Indicator ===
    def linreg_slope(series, period=20):
        x = np.arange(period)
        slopes = series.rolling(period).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == period else np.nan, raw=True)
        return slopes
    df["LinReg_Slope_20"] = linreg_slope(close, 20) / close * 100
    df["LinReg_Slope_50"] = linreg_slope(close, 50) / close * 100
    # R-squared (how well price fits the trend)
    def linreg_r2(series, period=20):
        def calc_r2(y):
            if len(y) != period:
                return np.nan
            x = np.arange(period)
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return series.rolling(period).apply(calc_r2, raw=True)
    df["LinReg_R2_20"] = linreg_r2(close, 20)

    # === Wilder Moving Average (same as EMA with period 2*n-1) ===
    df["Wilder_MA_14"] = close.ewm(alpha=1/14, adjust=False).mean()
    df["Price_vs_Wilder"] = (close - df["Wilder_MA_14"]) / df["Wilder_MA_14"] * 100

    # === Chaikin Oscillator (AD Line EMA3 - EMA10) ===
    clv2 = ((close - low) - (high - close)) / (high - low)
    clv2 = clv2.fillna(0)
    ad = (clv2 * volume).cumsum()
    df["Chaikin_Osc"] = ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean()
    df["Chaikin_Osc_Norm"] = df["Chaikin_Osc"] / volume.rolling(20).mean()

    # === Price Volume Trend (PVT) ===
    pvt = (close.pct_change() * volume).fillna(0).cumsum()
    df["PVT_Norm"] = pvt / volume.rolling(20).mean()
    df["PVT_Signal"] = (pvt.ewm(span=20, adjust=False).mean()) / volume.rolling(20).mean()

    # === Twiggs Money Flow ===
    tr_twg = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    adv = ((close - low.clip(upper=close.shift(1).values)) / tr_twg) * volume
    dec = ((high.clip(lower=close.shift(1).values) - close) / tr_twg) * volume
    tmf_vol = volume.ewm(span=21, adjust=False).mean()
    df["Twiggs_MF"] = (adv.ewm(span=21, adjust=False).mean() - dec.ewm(span=21, adjust=False).mean()) / tmf_vol

    # === Chaikin Volatility ===
    hl_ema = (high - low).ewm(span=10, adjust=False).mean()
    df["Chaikin_Vol"] = (hl_ema - hl_ema.shift(10)) / hl_ema.shift(10) * 100

    # === Volatility Ratio ===
    tr_today = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["Volatility_Ratio"] = tr_today / tr_today.rolling(14).mean()

    # === Negative Volume Index / Positive Volume Index ===
    nvi = pd.Series(1000.0, index=df.index, dtype=float)
    pvi = pd.Series(1000.0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        if volume.iloc[i] < volume.iloc[i-1]:
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + close.pct_change().iloc[i])
            pvi.iloc[i] = pvi.iloc[i-1]
        else:
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + close.pct_change().iloc[i])
            nvi.iloc[i] = nvi.iloc[i-1]
    df["NVI_Ratio"] = nvi / nvi.rolling(50).mean()
    df["PVI_Ratio"] = pvi / pvi.rolling(50).mean()

    # === Volume Oscillator ===
    df["Vol_Osc"] = (volume.rolling(5).mean() - volume.rolling(20).mean()) / volume.rolling(20).mean() * 100

    # === Chandelier Exit (distance from high) ===
    highest_22 = high.rolling(22).max()
    df["Chandelier_Dist"] = (highest_22 - close) / close * 100

    # === Aroon Oscillator ===
    def aroon_up(s, period=25):
        return s.rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    def aroon_down(s, period=25):
        return s.rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    df["Aroon_Up"] = aroon_up(high, 25)
    df["Aroon_Down"] = aroon_down(low, 25)
    df["Aroon_Osc"] = df["Aroon_Up"] - df["Aroon_Down"]

    # === Ease of Movement (EMV) ===
    dm_emv = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (volume / 1e6) / (high - low)
    emv = dm_emv / box_ratio
    df["EMV"] = emv.rolling(14).mean()
    df["EMV_Signal"] = df["EMV"].rolling(9).mean()

    # === Vertical Horizontal Filter (VHF) ===
    highest_close = close.rolling(28).max()
    lowest_close = close.rolling(28).min()
    sum_abs_change = close.diff().abs().rolling(28).sum()
    df["VHF"] = (highest_close - lowest_close) / sum_abs_change

    # === Heikin Ashi signals ===
    ha_close = (df["Open"] + high + low + close) / 4
    ha_open = pd.Series(dtype=float, index=df.index)
    ha_open.iloc[0] = (df["Open"].iloc[0] + close.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    ha_body = ha_close - ha_open
    ha_range = ha_high - ha_low
    df["HA_Body_Ratio"] = ha_body / ha_range
    df["HA_Bullish"] = (ha_body > 0).astype(int)
    df["HA_Consecutive_Bull"] = df["HA_Bullish"].groupby((~df["HA_Bullish"].astype(bool)).cumsum()).cumsum()
    df["HA_Consecutive_Bear"] = (1 - df["HA_Bullish"]).groupby(df["HA_Bullish"].astype(bool).cumsum()).cumsum()

    # === Pivot Points (classic) ===
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    df["Pivot_Dist"] = (close - pivot) / close * 100
    r1 = 2 * pivot - low.shift(1)
    s1 = 2 * pivot - high.shift(1)
    df["Pivot_R1_Dist"] = (r1 - close) / close * 100
    df["Pivot_S1_Dist"] = (close - s1) / close * 100

    # === Displaced Moving Average (DMA) ===
    df["DMA_20_5"] = close.rolling(20).mean().shift(5)
    df["Price_vs_DMA"] = (close - df["DMA_20_5"]) / df["DMA_20_5"] * 100

    # === MA Oscillator ===
    df["MA_Osc_10_30"] = (close.rolling(10).mean() - close.rolling(30).mean()) / close.rolling(30).mean() * 100

    df = df.copy()
    return df


def label_tp_sl(df: pd.DataFrame, atr_col: str = "ATR", tp_mult: float = 1.5, sl_mult: float = 1.0, max_lookahead: int = 10) -> pd.DataFrame:
    """
    For each candle, look forward to determine TP/SL outcome for BOTH directions.

    LONG position:
      TP = Entry + tp_mult * ATR (price goes UP to profit)
      SL = Entry - sl_mult * ATR (price goes DOWN to loss)

    SHORT position:
      TP = Entry - tp_mult * ATR (price goes DOWN to profit)
      SL = Entry + sl_mult * ATR (price goes UP to loss)
    """
    long_tp_levels = []
    long_sl_levels = []
    long_verdicts = []
    long_days = []
    short_tp_levels = []
    short_sl_levels = []
    short_verdicts = []
    short_days = []

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df[atr_col].values

    for i in range(len(df)):
        atr = atrs[i]
        if np.isnan(atr):
            for lst in [long_tp_levels, long_sl_levels, long_verdicts, long_days,
                        short_tp_levels, short_sl_levels, short_verdicts, short_days]:
                lst.append(np.nan)
            continue

        entry = closes[i]

        # --- LONG ---
        l_tp = entry + tp_mult * atr
        l_sl = entry - sl_mult * atr
        long_tp_levels.append(l_tp)
        long_sl_levels.append(l_sl)

        l_verdict = None
        l_days_needed = np.nan
        for j in range(i + 1, min(i + 1 + max_lookahead, len(df))):
            hit_tp = highs[j] >= l_tp
            hit_sl = lows[j] <= l_sl
            if hit_tp and hit_sl:
                prev_close = closes[j - 1] if j > 0 else entry
                if prev_close - l_sl < l_tp - prev_close:
                    l_verdict = "SL"
                else:
                    l_verdict = "TP"
                l_days_needed = j - i
                break
            elif hit_tp:
                l_verdict = "TP"
                l_days_needed = j - i
                break
            elif hit_sl:
                l_verdict = "SL"
                l_days_needed = j - i
                break
        long_verdicts.append(l_verdict)
        long_days.append(l_days_needed)

        # --- SHORT ---
        s_tp = entry - tp_mult * atr
        s_sl = entry + sl_mult * atr
        short_tp_levels.append(s_tp)
        short_sl_levels.append(s_sl)

        s_verdict = None
        s_days_needed = np.nan
        for j in range(i + 1, min(i + 1 + max_lookahead, len(df))):
            hit_tp = lows[j] <= s_tp
            hit_sl = highs[j] >= s_sl
            if hit_tp and hit_sl:
                prev_close = closes[j - 1] if j > 0 else entry
                if s_sl - prev_close < prev_close - s_tp:
                    s_verdict = "SL"
                else:
                    s_verdict = "TP"
                s_days_needed = j - i
                break
            elif hit_tp:
                s_verdict = "TP"
                s_days_needed = j - i
                break
            elif hit_sl:
                s_verdict = "SL"
                s_days_needed = j - i
                break
        short_verdicts.append(s_verdict)
        short_days.append(s_days_needed)

    df["LONG_TP_Level"] = long_tp_levels
    df["LONG_SL_Level"] = long_sl_levels
    df["VERDICT_LONG"] = long_verdicts
    df["DAY_PASS_LONG"] = long_days
    df["SHORT_TP_Level"] = short_tp_levels
    df["SHORT_SL_Level"] = short_sl_levels
    df["VERDICT_SHORT"] = short_verdicts
    df["DAY_PASS_SHORT"] = short_days

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data with TP/SL labels")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--months", type=int, default=108, help="Number of months of data")
    parser.add_argument("--tp-mult", type=float, default=1.5, help="TP multiplier of ATR")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="SL multiplier of ATR")
    parser.add_argument("--lookahead", type=int, default=10, help="Max days to look ahead for TP/SL")
    args = parser.parse_args()

    print(f"Fetching {args.months} months of data for {args.ticker} (with warmup buffer)...")
    df = fetch_ohlcv(args.ticker, args.months)
    print(f"Got {len(df)} raw data points (includes warmup)")

    print("Computing ATR...")
    df["ATR"] = compute_atr(df, 14)

    print("Computing technical indicators...")
    df = compute_technical_indicators(df)

    print(f"Labeling TP/SL (TP={args.tp_mult}xATR, SL={args.sl_mult}xATR, lookahead={args.lookahead} days)...")
    df = label_tp_sl(df, "ATR", args.tp_mult, args.sl_mult, args.lookahead)

    # Truncate: only keep rows within the target date range (last N months)
    cutoff_date = datetime.now() - timedelta(days=args.months * 30)
    df = df[df["Date"] >= cutoff_date].reset_index(drop=True)

    # Remove rows where either direction has no verdict (not enough future data)
    df = df[df["VERDICT_LONG"].notna() & df["VERDICT_SHORT"].notna()].reset_index(drop=True)

    # Drop any row that still has NaN in any column
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {before_drop - len(df)} rows with NaN values")

    # Convert DAY_PASS columns to int
    df["DAY_PASS_LONG"] = df["DAY_PASS_LONG"].astype(int)
    df["DAY_PASS_SHORT"] = df["DAY_PASS_SHORT"].astype(int)

    date_str = datetime.now().strftime("%Y%m%d")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"{args.ticker}_tpsl_data_{date_str}.csv")
    df.to_csv(filename, index=False)
    print(f"\nSaved to {filename}")

    # Print summary
    total = len(df)
    long_tp = (df["VERDICT_LONG"] == "TP").sum()
    long_sl = (df["VERDICT_LONG"] == "SL").sum()
    short_tp = (df["VERDICT_SHORT"] == "TP").sum()
    short_sl = (df["VERDICT_SHORT"] == "SL").sum()
    print(f"\n--- Label Summary ---")
    print(f"Total candles (all complete, no NaN): {total}")
    print(f"\nLONG positions:")
    print(f"  TP Hit: {long_tp} ({long_tp/total*100:.1f}%)")
    print(f"  SL Hit: {long_sl} ({long_sl/total*100:.1f}%)")
    print(f"  Avg days to TP: {df[df['VERDICT_LONG']=='TP']['DAY_PASS_LONG'].mean():.1f}")
    print(f"  Avg days to SL: {df[df['VERDICT_LONG']=='SL']['DAY_PASS_LONG'].mean():.1f}")
    print(f"\nSHORT positions:")
    print(f"  TP Hit: {short_tp} ({short_tp/total*100:.1f}%)")
    print(f"  SL Hit: {short_sl} ({short_sl/total*100:.1f}%)")
    print(f"  Avg days to TP: {df[df['VERDICT_SHORT']=='TP']['DAY_PASS_SHORT'].mean():.1f}")
    print(f"  Avg days to SL: {df[df['VERDICT_SHORT']=='SL']['DAY_PASS_SHORT'].mean():.1f}")


if __name__ == "__main__":
    main()

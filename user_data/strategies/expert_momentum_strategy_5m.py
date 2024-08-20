# Required libraries
from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy, timeframe_to_minutes, BooleanParameter, IntParameter
from technical.util import resample_to_interval, resampled_merge

# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class ScalpingStrategy5m(IStrategy):
    """
    this strategy is based around the idea of generating a lot of
    potentatils buys and make tiny profits on each trade.
    we recommend to have at least 60 parallel trades at any time to cover
    non avoidable losses
    """

    INTERFACE_VERSION: int = 3
    # minimal_roi = {"0": 0.03,
    #                "10": 0.02,
    #                "30": 0.01,
    #                "120": 0.005,
    #                "1440": 0,
    #                "2880": -1
    #                }
    minimal_roi = {
        "0": 0.574,
        "1757": 0.158,
        "3804": 0.089,
        "6585": 0
    }
    stoploss = -0.2
    timeframe = "5m"

    # resample factor to establish our general trend. Basically don't buy if a trend is not given
    resample_factor = 5

    # Parameter definition for buying signals
    buy_adx = IntParameter(20, 50, default=32, space="buy", optimize=True)
    buy_fastd = IntParameter(15, 45, default=30, space="buy", optimize=True)
    buy_fastk = IntParameter(15, 45, default=26, space="buy", optimize=True)
    buy_mfi = IntParameter(10, 25, default=22, space="buy", optimize=True)
    buy_adx_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    buy_fastd_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    buy_fastk_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    buy_mfi_enabled = BooleanParameter(default=True, space="buy", optimize=True)

    # Parameter definition for selling signals
    sell_adx = IntParameter(50, 100, default=53, space="sell", optimize=True)
    sell_cci = IntParameter(100, 200, default=183, space="sell", optimize=True)
    sell_fastd = IntParameter(50, 100, default=79, space="sell", optimize=True)
    sell_fastk = IntParameter(50, 100, default=70, space="sell", optimize=True)
    sell_mfi = IntParameter(75, 100, default=92, space="sell", optimize=True)

    sell_adx_enabled = BooleanParameter(default=False, space="sell", optimize=True)
    sell_cci_enabled = BooleanParameter(default=True, space="sell", optimize=True)
    sell_fastd_enabled = BooleanParameter(default=True, space="sell", optimize=True)
    sell_fastk_enabled = BooleanParameter(default=True, space="sell", optimize=True)
    sell_mfi_enabled = BooleanParameter(default=False, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute indicators on the given dataframe, used for generating trading signals.
        """
        # print(len(dataframe))
        tf_res = timeframe_to_minutes(self.timeframe) * self.resample_factor
        # print(f'resample factor: {self.resample_factor}')
        df_res = resample_to_interval(dataframe, tf_res)
        # print(f'resampled: {df_res}')
        # print(len(df_res))
        df_res["sma"] = ta.SMA(df_res, 50, price="close")
        # print(df_res)
        dataframe = resampled_merge(dataframe, df_res, fill_na=True)
        # print(dataframe.tail(20))
        dataframe["resample_sma"] = dataframe[f"resample_{tf_res}_sma"]

        dataframe["ema_high"] = ta.EMA(dataframe, timeperiod=5, price="high")
        dataframe["ema_close"] = ta.EMA(dataframe, timeperiod=5, price="close")
        dataframe["ema_low"] = ta.EMA(dataframe, timeperiod=5, price="low")
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe)

        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_middleband"] = bollinger["mid"]

        # print(dataframe['fastd'].head(10))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the logic for entering trades. Checks various conditions based
        on indicators.
        """
        conditions = []

        # static conditions which always apply
        conditions.append(qtpylib.crossed_above(dataframe["fastk"], dataframe["fastd"]))
        conditions.append(dataframe["resample_sma"] < dataframe["close"])

        # additional conditions could be added here
        if hasattr(self, "buy_mfi_enabled") and self.buy_mfi_enabled.value:
            conditions.append(dataframe["mfi"] < self.buy_mfi.value)
        if hasattr(self, "buy_fastd_enabled") and self.buy_fastd_enabled.value:
            conditions.append(dataframe["fastd"] < self.buy_fastd.value)
        if hasattr(self, "buy_fastk_enabled") and self.buy_fastk_enabled.value:
            conditions.append(dataframe["fastk"] < self.buy_fastk.value)
        if hasattr(self, "buy_adx_enabled") and self.buy_adx_enabled.value:
            conditions.append(dataframe["adx"] > self.buy_adx.value)

        # Check that volume is not 0
        conditions.append(dataframe["volume"] > 0)
        # print(f'conditions: {conditions}')

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the logic for exiting trades. Checks various conditions based
        on indicators.
        """
        conditions = []

        # static conditions which always apply,
        conditions.append(dataframe["open"] > dataframe["ema_high"])
        # Check that volume is not 0
        conditions.append(dataframe["volume"] > 0)

        # additional conditions could be added here
        if hasattr(self, "sell_mfi_enabled") and self.sell_mfi_enabled.value:
            conditions.append(dataframe["mfi"] > self.sell_mfi.value)
        if hasattr(self, "sell_fastd_enabled") and self.sell_fastd_enabled.value:
            conditions.append(dataframe["fastd"] > self.sell_fastd.value)
        if hasattr(self, "sell_fastk_enabled") and self.sell_fastk_enabled.value:
            conditions.append(dataframe["fastk"] > self.sell_fastk.value)
        if hasattr(self, "sell_adx_enabled") and self.sell_adx_enabled.value:
            conditions.append(dataframe["adx"] < self.sell_adx.value)
        if hasattr(self, "sell_cci_enabled") and self.sell_cci_enabled.value:
            conditions.append(dataframe["cci"] > self.sell_cci.value)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe

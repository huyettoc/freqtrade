# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
from datetime import datetime

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    CategoricalParameter,
    BooleanParameter,
)
from freqtrade.strategy import IStrategy


class ExpertPSARStrategy(IStrategy):
    """
    this is an example class, implementing a PSAR based trailing stop loss
    you are supposed to take the `custom_stoploss()` and `populate_indicators()`
    parts and adapt it to your own strategy

    the populate_entry_trend() function is pretty nonsencial
    """

    INTERFACE_VERSION: int = 3
    timeframe = "1h"
    stoploss = -0.2
    custom_info = {}
    minimal_roi = {"14400": -1}
    # use_custom_stoploss = True
    plot_config = {
        "main_plot": {
            # Configuration for main plot indicators.
            # Specifies `ema10` to be red, and `ema50` to be a shade of gray
            "sar": {"color": "green"},
        },
        "subplots": {
            # Additional subplot RSI
            "ADX": {"adx": {}}
        },
    }
    can_short = True
    take_profit = DecimalParameter(0.05, 0.2, default=0.05, decimals=2, space="sell", optimize=True)
    acceleration = DecimalParameter(0.02, 0.2, default=0.02, decimals=2, space="buy", optimize=True)
    exit_long_ma_period = CategoricalParameter(
        [5, 10, 20, 50, 100, 200], default=5, space="sell", optimize=True
    )
    exit_long_adx_value = CategoricalParameter(
        [20, 25, 30, 35, 40, 45], default=25, space="sell", optimize=True
    )
    exit_long_rsi_value = CategoricalParameter(
        [20, 25, 30, 35, 40, 45], default=25, space="sell", optimize=True
    )
    exit_short_ma_period = CategoricalParameter(
        [5, 10, 20, 50, 100, 200], default=5, space="sell", optimize=True
    )
    exit_short_adx_value = CategoricalParameter(
        [20, 25, 30, 35, 40, 45], default=25, space="sell", optimize=True
    )
    exit_short_rsi_value = CategoricalParameter(
        [55, 60, 65, 70, 75, 80], default=75, space="sell", optimize=True
    )
    # enable_exit_long_ma = BooleanParameter(space="sell", optimize=True)
    # enable_exit_short_ma = BooleanParameter(space="sell", optimize=True)
    # enable_exit_long_adx = BooleanParameter(space="sell", optimize=True)
    # enable_exit_short_adx = BooleanParameter(space="sell", optimize=True)
    # enable_exit_long_macd = BooleanParameter(space="sell", optimize=True)
    # enable_exit_short_macd = BooleanParameter(space="sell", optimize=True)
    # enable_exit_long_rsi = BooleanParameter(space="sell", optimize=True)
    # enable_exit_short_rsi = BooleanParameter(space="sell", optimize=True)
    # enable_exit_long_bband = BooleanParameter(space="buy", optimize=True)
    # enable_exit_short_bband = BooleanParameter(space="buy", optimize=True)
    exit_long_trigger = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="sell", optimize=True
    )

    exit_short_trigger = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="sell", optimize=True
    )

    exit_long_trigger_2 = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="sell", optimize=True
    )

    exit_short_trigger_2 = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="sell", optimize=True
    )
    enable_exit_long_2 = BooleanParameter(space="sell", optimize=True)
    enable_exit_short_2 = BooleanParameter(space="sell", optimize=True)

    long_ma_period = CategoricalParameter(
        [5, 10, 20, 50, 100, 200], default=5, space="buy", optimize=True
    )
    long_adx_value = CategoricalParameter(
        [20, 25, 30, 35, 40, 45, 50], default=25, space="buy", optimize=True
    )
    long_rsi_value = CategoricalParameter(
        [20, 25, 30, 35, 40, 45], default=25, space="buy", optimize=True
    )
    short_ma_period = CategoricalParameter(
        [5, 10, 20, 50, 100, 200], default=5, space="buy", optimize=True
    )
    short_adx_value = CategoricalParameter(
        [20, 25, 30, 35, 40, 45, 50], default=25, space="buy", optimize=True
    )
    short_rsi_value = CategoricalParameter(
        [55, 60, 65, 70, 75, 80], default=75, space="buy", optimize=True
    )
    # enable_long_ma = BooleanParameter(space="buy", optimize=True)
    # enable_short_ma = BooleanParameter(space="buy", optimize=True)
    # enable_long_adx = BooleanParameter(space="buy", optimize=True)
    # enable_short_adx = BooleanParameter(space="buy", optimize=True)
    # enable_long_macd = BooleanParameter(space="buy", optimize=True)
    # enable_short_macd = BooleanParameter(space="buy", optimize=True)
    # enable_long_rsi = BooleanParameter(space="buy", optimize=True)
    # enable_short_rsi = BooleanParameter(space="buy", optimize=True)
    # enable_long_bband = BooleanParameter(space="buy", optimize=True)
    # enable_short_bband = BooleanParameter(space="buy", optimize=True)
    long_trigger = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="buy", optimize=True
    )

    short_trigger = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="buy", optimize=True
    )

    long_trigger_2 = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="buy", optimize=True
    )

    short_trigger_2 = CategoricalParameter(
        ["ma", "adx", "macd", "rsi", "bband"], default="ma", space="buy", optimize=True
    )
    enable_long_2 = BooleanParameter(space="buy", optimize=True)
    enable_short_2 = BooleanParameter(space="buy", optimize=True)
    # use_custom_stoploss = True
    #
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #
    #     if self.custom_info and pair in self.custom_info and trade:
    #         # using current_time directly (like below) will only work in backtesting/hyperopt.
    #         # in live / dry-run, it'll be really the current time
    #         relative_sl = None
    #         if self.dp:
    #             # so we need to get analyzed_dataframe from dp
    #             dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #             # only use .iat[-1] in callback methods, never in "populate_*" methods.
    #             # see: https://www.freqtrade.io/en/latest/strategy-customization/#common-mistakes-when-developing-strategies
    #             last_candle = dataframe.iloc[-1].squeeze()
    #             relative_sl = last_candle['sar']
    #
    #         if (relative_sl is not None):
    #             # print("current_rate: {}, custom_stoploss().relative_sl: {}".format(current_rate, relative_sl))
    #
    #             # calculate new_stoploss relative to current_rate
    #             new_stoploss = (current_rate - relative_sl) / current_rate
    #             # turn into relative negative offset required by `custom_stoploss` return implementation
    #             result = new_stoploss
    #     # print("pair -> {}, custom_stoploss() -> {}".format(pair, result))
    #     return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sar"] = ta.SAR(dataframe, acceleration=self.acceleration.value)
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.long_ma_period.value)
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=self.short_ma_period.value)

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["rsi"] = ta.RSI(dataframe)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        # if self.dp.runmode.value in ('backtest', 'hyperopt'):
        #     self.custom_info[metadata['pair']] = dataframe[['date', 'sar']].copy().set_index('date')

        # all "normal" indicators:
        # e.g.
        # dataframe['rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: buys when SAR is smaller then candle before
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        conditions = dataframe["close"] > dataframe["sar"]
        if self.long_trigger.value == "adx":
            conditions &= dataframe["adx"] > self.long_adx_value.value
        elif self.long_trigger.value == "ma":
            conditions &= dataframe["close"] > dataframe["ema_long"]
        elif self.long_trigger.value == "macd":
            conditions &= dataframe["macd"] > dataframe["macdsignal"]
        elif self.long_trigger.value == "rsi":
            conditions &= dataframe["rsi"] < self.long_rsi_value.value
        elif self.long_trigger.value == "bband":
            conditions &= dataframe["close"] < dataframe["bb_lowerband"]

        if self.enable_long_2.value:
            if self.long_trigger_2.value == "adx":
                conditions &= dataframe["adx"] > self.long_adx_value.value
            elif self.long_trigger_2.value == "ma":
                conditions &= dataframe["close"] > dataframe["ema_long"]
            elif self.long_trigger_2.value == "macd":
                conditions &= dataframe["macd"] > dataframe["macdsignal"]
            elif self.long_trigger_2.value == "rsi":
                conditions &= dataframe["rsi"] < self.long_rsi_value.value
            elif self.long_trigger_2.value == "bband":
                conditions &= dataframe["close"] < dataframe["bb_lowerband"]

        dataframe.loc[(conditions), ["enter_long", "enter_tag"]] = (1, "long")

        conditions = dataframe["close"] < dataframe["sar"]
        if self.short_trigger.value == "adx":
            conditions &= dataframe["adx"] > self.short_adx_value.value
        if self.short_trigger.value == "ma":
            conditions &= dataframe["close"] < dataframe["ema_short"]
        if self.short_trigger.value == "macd":
            conditions &= dataframe["macd"] < dataframe["macdsignal"]
        if self.short_trigger.value == "rsi":
            conditions &= dataframe["rsi"] > self.short_rsi_value.value
        if self.short_trigger.value == "bband":
            conditions &= dataframe["close"] > dataframe["bb_upperband"]

        if self.enable_short_2.value:
            if self.short_trigger_2.value == "adx":
                conditions &= dataframe["adx"] > self.short_adx_value.value
            if self.short_trigger_2.value == "ma":
                conditions &= dataframe["close"] < dataframe["ema_short"]
            if self.short_trigger_2.value == "macd":
                conditions &= dataframe["macd"] < dataframe["macdsignal"]
            if self.short_trigger_2.value == "rsi":
                conditions &= dataframe["rsi"] > self.short_rsi_value.value
            if self.short_trigger_2.value == "bband":
                conditions &= dataframe["close"] > dataframe["bb_upperband"]

        dataframe.loc[(conditions), ["enter_short", "enter_tag"]] = (1, "short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: does nothing
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        conditions = dataframe["close"] < dataframe["sar"]
        if self.exit_long_trigger.value == "adx":
            conditions &= dataframe["adx"] > self.exit_long_adx_value.value
        if self.exit_long_trigger.value == "ma":
            conditions &= dataframe["close"] < dataframe["ema_long"]
        if self.exit_long_trigger.value == "macd":
            conditions &= dataframe["macd"] < dataframe["macdsignal"]
        if self.exit_long_trigger.value == "rsi":
            conditions &= dataframe["rsi"] > self.exit_long_rsi_value.value
        if self.exit_long_trigger.value == "bband":
            conditions &= dataframe["close"] > dataframe["bb_middleband"]

        if self.enable_exit_long_2:
            if self.exit_long_trigger_2.value == "adx":
                conditions &= dataframe["adx"] > self.exit_long_adx_value.value
            if self.exit_long_trigger_2.value == "ma":
                conditions &= dataframe["close"] < dataframe["ema_long"]
            if self.exit_long_trigger_2.value == "macd":
                conditions &= dataframe["macd"] < dataframe["macdsignal"]
            if self.exit_long_trigger_2.value == "rsi":
                conditions &= dataframe["rsi"] > self.exit_long_rsi_value.value
            if self.exit_long_trigger_2.value == "bband":
                conditions &= dataframe["close"] > dataframe["bb_middleband"]

        # Deactivated sell signal to allow the strategy to work correctly
        dataframe.loc[(conditions), "exit_long"] = 1

        conditions = dataframe["close"] > dataframe["sar"]
        if self.exit_short_trigger.value == "adx":
            conditions &= dataframe["adx"] > self.exit_short_adx_value.value
        if self.exit_short_trigger.value == "ma":
            conditions &= dataframe["close"] > dataframe["ema_short"]
        if self.exit_short_trigger.value == "macd":
            conditions &= dataframe["macd"] > dataframe["macdsignal"]
        if self.exit_short_trigger.value == "rsi":
            conditions &= dataframe["rsi"] < self.exit_short_rsi_value.value
        if self.exit_short_trigger.value == "bband":
            conditions &= dataframe["close"] < dataframe["bb_middleband"]

        if self.enable_exit_short_2.value:
            if self.exit_short_trigger_2.value == "adx":
                conditions &= dataframe["adx"] > self.exit_short_adx_value.value
            if self.exit_short_trigger_2.value == "ma":
                conditions &= dataframe["close"] > dataframe["ema_short"]
            if self.exit_short_trigger_2.value == "macd":
                conditions &= dataframe["macd"] > dataframe["macdsignal"]
            if self.exit_short_trigger_2.value == "rsi":
                conditions &= dataframe["rsi"] < self.exit_short_rsi_value.value
            if self.exit_short_trigger_2.value == "bband":
                conditions &= dataframe["close"] < dataframe["bb_middleband"]

        dataframe.loc[(conditions), "exit_short"] = 1
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        if current_profit >= self.take_profit.value:
            return "customized take profit"

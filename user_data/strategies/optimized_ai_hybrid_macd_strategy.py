from datetime import datetime, timedelta
from typing import Dict, Optional

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade

# --- Do not remove these libs ---
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy


# --------------------------------


class OptimizedAIHybridMACDStrategy(IStrategy):
    """
    author@: Gert Wohlgemuth

    idea:

        uptrend definition:
            MACD above MACD signal
            and CCI < -50

        downtrend definition:
            MACD below MACD signal
            and CCI > 100

    freqtrade hyperopt --strategy MACDStrategy --hyperopt-loss <someLossFunction> --spaces buy sell

    The idea is to optimize only the CCI value.
    - Buy side: CCI between -700 and 0
    - Sell side: CCI between 0 and 700

    """

    INTERFACE_VERSION: int = 3
    # "roi": {
    #   "0": 0.248,
    #   "246": 0.098,
    #   "526": 0.024,
    #   "1728": 0
    # },
    # "stoploss": {
    #   "stoploss": -0.048
    # }
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #     "0": 0.3,
    #     "5000": -1
    #
    # }
    stoploss = -0.8
    minimal_roi = {"0": 1, "5000": -1}

    use_custom_stoploss = True
    optimized_take_profit = DecimalParameter(
        0.05, 0.5, default=0.05, decimals=2, space="sell", optimize=True
    )
    optimized_stoploss = DecimalParameter(
        -0.5, -0.05, default=-0.05, decimals=2, space="sell", optimize=True
    )
    optimized_leverage = CategoricalParameter(
        [2, 3, 5, 7, 10], default=3, space="buy", optimize=True
    )

    ai_thres_buy = DecimalParameter(0.0, 0.5, default=0.0, space="buy", optimize=False)
    ai_thres_sell = DecimalParameter(-0.5, 0.0, default=-0.0, space="sell", optimize=False)

    buy_cci = IntParameter(low=-700, high=0, default=-50, space="buy", optimize=False)
    sell_cci = IntParameter(low=0, high=700, default=50, space="sell", optimize=False)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    can_short = False

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        return self.optimized_stoploss.value

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return self.optimized_leverage.value

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        dataframe["cci"] = ta.CCI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (  # ai trigger
                (dataframe["do_predict"] == 1)
                & (dataframe["&-s_close"] > self.ai_thres_buy.value)
                & (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["cci"] < self.buy_cci.value)
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        dataframe.loc[
            (  # ai trigger
                (dataframe["do_predict"] == 1)
                & (dataframe["&-s_close"] < self.ai_thres_sell.value)
                & (dataframe["macd"] < dataframe["macdsignal"])
                & (dataframe["cci"] > self.sell_cci.value)
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                # (dataframe["do_predict"] == 1) &
                # (dataframe["&-s_close"] < self.ai_thres_sell.value) &
                (dataframe["macd"] < dataframe["macdsignal"])
                & (dataframe["cci"] > self.sell_cci.value)
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                # (dataframe["do_predict"] == 1) &
                # (dataframe["&-s_close"] < self.ai_thres_sell.value) &
                (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["cci"] < self.sell_cci.value)
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_short",
        ] = 1
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

        # dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        #
        # last_candle = dataframe.iloc[-1].squeeze()
        # trade_date = timeframe_to_prev_date(
        #     self.timeframe, (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        # )
        # trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        # if trade_candle.empty:
        #     return None

        # trade_duration = (current_time - trade.open_date_utc).seconds / 60

        # if trade_duration > 1000:
        #     return "trade expired"

        if current_profit >= self.optimized_take_profit.value:
            return "customized take profit"

    @property
    def plot_config(self):
        """
        There are a lot of solutions how to build the return dictionary.
        The only important point is the return value.
        Example:
            plot_config = {'main_plot': {}, 'subplots': {}}

        """
        plot_config = {}
        plot_config["main_plot"] = {}
        plot_config["subplots"] = {
            # Create subplot MACD
            "MACD": {
                "macd": {"color": "blue", "fill_to": "macdhist"},
                "macdsignal": {"color": "orange"},
                "macdhist": {"type": "bar", "plotly": {"opacity": 0.9}},
            },
            # Additional
            "AI Prediction": {"&-s_close": {"color": "red"}, "s_close_mean": {}},
        }

        return plot_config

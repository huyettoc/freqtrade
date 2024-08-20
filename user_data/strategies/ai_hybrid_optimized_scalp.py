import logging
from functools import reduce
from typing import Dict

import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from technical.util import resample_to_interval, resampled_merge

from freqtrade.strategy import (  # noqa
    IntParameter,
    IStrategy,
    merge_informative_pair,
    timeframe_to_minutes,
)


logger = logging.getLogger(__name__)


class AIHybridOptimizedScalp(IStrategy):
    # Buy hyperspace params:
    buy_params = {
        "ai_threshold": 0.001,
        "buy_adx": 32,
        "buy_adx_enabled": True,
        "buy_fastd": 26,
        "buy_fastd_enabled": True,
        "buy_fastk": 38,
        "buy_fastk_enabled": False,
        "buy_mfi": 24,
        "buy_mfi_enabled": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_adx": 64,
        "sell_adx_enabled": False,
        "sell_cci": 109,
        "sell_cci_enabled": True,
        "sell_fastd": 50,
        "sell_fastd_enabled": False,
        "sell_fastk": 95,
        "sell_fastk_enabled": True,
        "sell_mfi": 99,
        "sell_mfi_enabled": True,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.21,
        "11": 0.061,
        "40": 0.015,
        "137": 0
    }

    # Stoploss:
    stoploss = -0.236

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.053
    trailing_stop_positive_offset = 0.111
    trailing_only_offset_is_reached = True

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs) -> DataFrame:
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
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
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
            self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        dataframe = self.freqai.start(dataframe, metadata, self)

        tf_res = timeframe_to_minutes(self.timeframe) * 5
        df_res = resample_to_interval(dataframe, tf_res)
        df_res['sma'] = ta.SMA(df_res, 50, price='close')
        dataframe = resampled_merge(dataframe, df_res, fill_na=True)
        dataframe['resample_sma'] = dataframe[f'resample_{tf_res}_sma']

        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        if self.buy_params['buy_mfi_enabled']:
            conditions.append(dataframe['mfi'] < self.buy_params['buy_mfi'])
        if self.buy_params['buy_fastd_enabled']:
            conditions.append(dataframe['fastd'] < self.buy_params['buy_fastd'])
        if self.buy_params['buy_fastk_enabled']:
            conditions.append(dataframe['fastk'] < self.buy_params['buy_fastk'])
        if self.buy_params['buy_adx_enabled']:
            conditions.append(dataframe['adx'] > self.buy_params['buy_adx'])

        # ai trigger
        conditions.append(dataframe["do_predict"] == 1)
        conditions.append(dataframe["&-s_close"] > self.buy_params['ai_threshold'])
        # Some static conditions which always apply
        conditions.append(qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
        # conditions.append(dataframe['resample_sma'] < dataframe['close'])

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = [dataframe['open'] > dataframe['ema_high']]

        # Some static conditions which always apply

        if self.sell_params['sell_mfi_enabled']:
            conditions.append(dataframe['mfi'] > self.sell_params['sell_mfi'])
        if self.sell_params['sell_fastd_enabled']:
            conditions.append(dataframe['fastd'] > self.sell_params['sell_fastd'])
        if self.sell_params['sell_fastk_enabled']:
            conditions.append(dataframe['fastk'] > self.sell_params['sell_fastk'])
        if self.sell_params['sell_adx_enabled']:
            conditions.append(dataframe['adx'] < self.sell_params['sell_adx'])
        if self.sell_params['sell_cci_enabled']:
            conditions.append(dataframe['cci'] > self.sell_params['sell_cci'])

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe

# --- Do not remove these libs ---
from functools import reduce

import numpy  # noqa

# --------------------------------
import talib.abstract as ta
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, timeframe_to_minutes


class OptimizedSmoothScalp(IStrategy):
    """
        this strategy is based around the idea of generating a lot of potentatils buys and
        make tiny profits on each
        trade we recommend to have at least 60 parallel trades at any time to cover
        non avoidable losses
    """

    INTERFACE_VERSION: int = 3

    # Buy hyperspace params:
    buy_params = {
        "buy_adx": 40,
        "buy_adx_enabled": False,
        "buy_fastd": 17,
        "buy_fastd_enabled": False,
        "buy_fastk": 40,
        "buy_fastk_enabled": False,
        "buy_mfi": 24,
        "buy_mfi_enabled": False,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_adx": 80,
        "sell_adx_enabled": False,
        "sell_cci": 152,
        "sell_cci_enabled": False,
        "sell_fastd": 98,
        "sell_fastd_enabled": True,
        "sell_fastk": 59,
        "sell_fastk_enabled": False,
        "sell_mfi": 85,
        "sell_mfi_enabled": True,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.139,
        "33": 0.097,
        "69": 0.037,
        "140": 0
    }

    # Stoploss:
    stoploss = -0.313

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.057
    trailing_stop_positive_offset = 0.141
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

        # Some static conditions which always apply
        conditions.append(qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
        conditions.append(dataframe['resample_sma'] < dataframe['close'])

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # Some static conditions which always apply
        conditions.append(dataframe['open'] > dataframe['ema_high'])

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

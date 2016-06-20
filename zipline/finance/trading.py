#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logbook
import pandas as pd

from six import string_types
from sqlalchemy import create_engine

from zipline.assets import AssetDBWriter, AssetFinder
from zipline.data.loader import load_market_data
from zipline.utils.calendars import default_nyse_calendar
from zipline.utils.memoize import remember_last

log = logbook.Logger('Trading')


class TradingEnvironment(object):
    """
    The financial simulations in zipline depend on information
    about the benchmark index and the risk free rates of return.
    The benchmark index defines the benchmark returns used in
    the calculation of performance metrics such as alpha/beta. Many
    components, including risk, performance, transforms, and
    batch_transforms, need access to a calendar of trading days and
    market hours. The TradingEnvironment maintains two time keeping
    facilities:
      - a DatetimeIndex of trading days for calendar calculations
      - a timezone name, which should be local to the exchange
        hosting the benchmark index. All dates are normalized to UTC
        for serialization and storage, and the timezone is used to
       ensure proper rollover through daylight savings and so on.

    User code will not normally need to use TradingEnvironment
    directly. If you are extending zipline's core financial
    components and need to use the environment, you must import the module and
    build a new TradingEnvironment object, then pass that TradingEnvironment as
    the 'env' arg to your TradingAlgorithm.

    Parameters
    ----------
    load : callable, optional
        The function that returns benchmark returns and treasury curves.
        The treasury curves are expected to be a DataFrame with an index of
        dates and columns of the curve names, e.g. '10year', '1month', etc.
    bm_symbol : str, optional
        The benchmark symbol
    exchange_tz : tz-coercable, optional
        The timezone of the exchange.
    min_date : datetime, optional
        The oldest date that we know about in this environment.
    max_date : datetime, optional
        The most recent date that we know about in this environment.
    env_trading_calendar : pd.DatetimeIndex, optional
        The calendar of datetimes that define our market hours.
    asset_db_path : str or sa.engine.Engine, optional
        The path to the assets db or sqlalchemy Engine object to use to
        construct an AssetFinder.
    """

    # Token used as a substitute for pickling objects that contain a
    # reference to a TradingEnvironment
    PERSISTENT_TOKEN = "<TradingEnvironment>"

    def __init__(
        self,
        load=None,
        bm_symbol='^GSPC',
        exchange_tz="US/Eastern",
        trading_calendar=default_nyse_calendar,
        asset_db_path=':memory:'
    ):

        self.bm_symbol = bm_symbol
        if not load:
            load = load_market_data

        self.benchmark_returns, self.treasury_curves = load(
            trading_calendar.day,
            trading_calendar.schedule.index,
            self.bm_symbol,
        )

        self.exchange_tz = exchange_tz

        if isinstance(asset_db_path, string_types):
            asset_db_path = 'sqlite:///%s' % asset_db_path
            self.engine = engine = create_engine(asset_db_path)
        else:
            self.engine = engine = asset_db_path

        if engine is not None:
            AssetDBWriter(engine).init_db()
            self.asset_finder = AssetFinder(engine)
        else:
            self.asset_finder = None

    def write_data(self, **kwargs):
        """Write data into the asset_db.

        Parameters
        ----------
        **kwargs
            Forwarded to AssetDBWriter.write
        """
        AssetDBWriter(self.engine).write(**kwargs)


class SimulationParameters(object):
    def __init__(self, period_start, period_end,
                 trading_calendar,
                 capital_base=10e3,
                 emission_rate='daily',
                 data_frequency='daily',
                 arena='backtest'):

        assert type(period_start) == pd.Timestamp
        assert type(period_end) == pd.Timestamp

        assert trading_calendar is not None, \
            "Must pass in trading calendar!"
        assert period_start <= period_end, \
            "Period start falls after period end."
        assert period_start <= trading_calendar.last_trading_session, \
            "Period start falls after the last known trading day."
        assert period_end >= trading_calendar.first_trading_session, \
            "Period end falls before the first known trading day."

        self.period_start = period_start
        self.period_end = period_end
        self.capital_base = capital_base

        self.emission_rate = emission_rate
        self.data_frequency = data_frequency

        # copied to algorithm's environment for runtime access
        self.arena = arena

        self.trading_calendar = trading_calendar

        if self.period_start not in trading_calendar.schedule.index:
            # if the start date is not a valid session in this calendar,
            # push it forward to the first valid session
            self.period_start = trading_calendar.minute_to_session_label(
                self.period_start
            )

        if self.period_end not in trading_calendar.schedule.index:
            # if the end date is not a valid session in this calendar,
            # pull it backward to the last valid session before the given
            # end date.
            self.period_end = trading_calendar.minute_to_session_label(
                self.period_end, direction="previous"
            )

        self.first_open = trading_calendar.open_and_close_for_session(
            self.period_start
        )[0]
        self.last_close = trading_calendar.open_and_close_for_session(
            self.period_end
        )[1]

    @property
    def first_session(self):
        return self.period_start

    @property
    def last_session(self):
        return self.period_end

    @property
    @remember_last
    def sessions(self):
        return self.trading_calendar.sessions_in_range(
            self.period_start,
            self.period_end
        )

    def create_from_copy(self, period_start, period_end):
        return SimulationParameters(
            period_start,
            period_end,
            self.trading_calendar,
            capital_base=self.capital_base,
            emission_rate=self.emission_rate,
            data_frequency=self.data_frequency,
            arena=self.arena
        )

    def __repr__(self):
        return """
{class_name}(
    period_start={period_start},
    period_end={period_end},
    capital_base={capital_base},
    data_frequency={data_frequency},
    emission_rate={emission_rate},
    first_open={first_open},
    last_close={last_close})\
""".format(class_name=self.__class__.__name__,
           period_start=self.period_start,
           period_end=self.period_end,
           capital_base=self.capital_base,
           data_frequency=self.data_frequency,
           emission_rate=self.emission_rate,
           first_open=self.first_open,
           last_close=self.last_close)


def noop_load(*args, **kwargs):
    """
    A method that can be substituted in as the load method in a
    TradingEnvironment to prevent it from loading benchmarks.

    Accepts any arguments, but returns only a tuple of Nones regardless
    of input.
    """
    return None, None

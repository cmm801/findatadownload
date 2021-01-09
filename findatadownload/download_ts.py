from abc import ABC, abstractmethod
import coinbasepro as cbp
import csv
import datetime
import io
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import time
import urllib.request
import yfinance as yf


def download_time_series(data_source, base_ticker, start=None, end=None, 
                         frequency=None, period=None):
    """ Download data for a single base ticker (which may yield one or more time series).

        Returns pandas DataFrame object with datetimes as the index and tickers as columns.
    """
    if data_source == 'gsw':
        # Daily US Government Yields from GSW/Fed
        dobj = GSWDownloader()
    elif data_source == 'fred':
        # Download all FRED tickers that are contained in our meta database
        dobj = FREDDownloader()
    elif data_source == 'acm':
        # Daily US Term Premium estimate from ACM/Fed
        dobj = ACMDownloader()
    elif data_source == 'kenfrench':
        # Daily Fama French factor data
        dobj = KenFrenchDownloader()
    elif data_source == 'yahoo':
        # Daily Yahoo data
        dobj = YahooDownloader()
    elif data_source == 'cbp':
        # Daily CoinbasePro data
        dobj = CoinbaseProDownloader()
    else:
        raise ValueError(f'Unknown data source: "{data_source}"')

    # Download the time series data
    ts = dobj.download_time_series(base_ticker, start=start, end=end,
                                   frequency=frequency, period=period)
    
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError('The index of the output must be a pandas DatetimeIndex.')

    # Don't specify a name for the index values
    ts.index.name = None
    return ts


class AbstractDownloader(ABC):
    def __init__(self):
        pass

    def download_time_series(self, base_ticker, start=None, end=None,
                             frequency=None, period=None):
        # parse arguments and make sure they are in a consistent format
        start, end, frequency, period = self._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)
        
        # Get the downloaded raw time series
        raw_ts = self._download_raw_time_series(base_ticker, start=start, end=end, 
                                                frequency=frequency, period=period)

        # Rename any columns if necessary
        ts = self._rename_time_series(raw_ts, base_ticker)
        
        # Add any additional constructed time series
        ts_full = self._add_constructed_time_series(ts, base_ticker)
        
        # Drop any rows where all data is missing
        ts_full.dropna(inplace=True, how='all', axis=0)
        
        # Return the pandas DataFrame
        return ts_full

    @abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ This method gets the raw time series from the data source.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Standardize input arguments so the lower-level functions know what data type to expect.
        """
        return start, end, frequency, period

    def _add_constructed_time_series(self, ts, base_ticker):
        """ Add additional constructed time series to the ones that are downloaded.
        
            This function can be overloaded to provide construction methodologies for new time series.
            By default, no new time series are constructed and only the downloaded data is returned.
        """
        return ts

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        return ts


class PandasDatareaderDownloader(AbstractDownloader):
    def _format_input_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Make adjustments to input arguments so they can be processed by pandas_datareader.
        
            Returns tuple (start, end, frequency, period)
        """
        # parse arguments and make sure they are in a consistent format
        start, end, frequency, period = super()._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)
        return start, end, frequency, period

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ This method gets the raw time series from the data source.
        """
        pass


class GSWDownloader(AbstractDownloader):
    URL_GSW = 'https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv'
    
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ This method gets the raw time series for the GSW US yield curve data.
        
            The base ticker gets ignored in this class, since there is only 1 dataset.
        """
        header_row = self._find_gsw_header_row()
        ts = pd.read_csv(self.URL_GSW, skiprows=header_row, index_col=0)
        
        # Make sure the index is a pandas DatetimeIndex
        ts.index = pd.DatetimeIndex(ts.index)
        return ts

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        ts = super()._rename_time_series(ts, base_ticker)
        ts.columns = 'GSW' + ts.columns
        return ts

    def _find_gsw_header_row(self):
        """ Find the header row for the .csv file containing the GSW yields.
        """
        webpage = urllib.request.urlopen(self.URL_GSW)
        datareader = csv.reader(io.TextIOWrapper(webpage))

        header_row = 0
        while header_row < 11 and datareader:
            txt = next(datareader)
            if txt and txt[0].lower() == 'date':
                break
            else:
                header_row += 1
        return header_row


class ACMDownloader(AbstractDownloader):
    ACM_URL = 'https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls'
    ACM_SHEETNAME = 'ACM Daily'
    ACM_INDEX_COL = 'DATE'
    
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ This method gets the raw time series for the ACM term premium data.
        
            The base ticker gets ignored in this class, since there is only 1 dataset.        
        """
        ts = pd.read_excel(self.ACM_URL, sheet_name=self.ACM_SHEETNAME)
        ts = ts.set_index(self.ACM_INDEX_COL)
        ts.index = pd.DatetimeIndex(ts.index)
        return ts


class FREDDownloader(PandasDatareaderDownloader):
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ This method gets the raw time series from the data source.
        """
        fred_obj = pdr.fred.FredReader([base_ticker], start=start, end=end)
        ts = fred_obj.read()
        return ts


class KenFrenchDownloader(PandasDatareaderDownloader):
    COLUMN_MAP = {
        'Developed_5_Factors_daily' : {
            'MKT-RF' : 'FFDEVMKT', 'SMB' : 'FFDEVSMB', 'HML' : 'FFDEVHML',
            'RMW' : 'FFDEVRMW', 'CMA' : 'FFDEVCMA', 'RF' : 'FFDEVRF'
        },
        'Developed_Mom_Factor_daily' : {
            'WML' : 'FFDEVMOM', 'MOM' : 'FFDEVMOM'
        },
        'F-F_Research_Data_5_Factors_2x3_daily' : {
            'MKT-RF' : 'FFUSMKT', 'SMB' : 'FFUSSMB', 'HML' : 'FFUSHML',
            'RMW' : 'FFUSRMW', 'CMA' : 'FFUSCMA', 'RF' : 'FFUSRF'
        },
        'F-F_Momentum_Factor_Daily' : {
            'WML' : 'FFUSMOM', 'MOM' : 'FFUSMOM'
        },
    }
    
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ This method gets the raw time series from the data source.
        """
        if period is not None:
            raise NotImplementedError(f'The argument "period" has value {period} but is not supported.')

        ff_reader = pdr.famafrench.FamaFrenchReader(base_ticker, start=start, end=end,
                                                    freq=frequency)

        # Get the monthly data
        # The 0-th index is the daily or monthly data - the 1-st index is annual data
        ff_ts = ff_reader.read()[0] 

        # Reset the index to use pandas DatetimeIndex objects
        if not isinstance(ff_ts.index, pd.DatetimeIndex):
            ff_ts.index = ff_ts.index.to_timestamp()

        # Return the time series data
        return ff_ts

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        ts = super()._rename_time_series(ts, base_ticker)
        
        # Get rid of whitespace in the names
        ff_ts = ts.copy()
        ff_ts.columns = [col.upper().replace(' ', '') for col in ts.columns]
        
        # Rename the columns
        ff_ts = ff_ts.rename(self.COLUMN_MAP[base_ticker], axis=1)
        return ff_ts


class YahooDownloader(AbstractDownloader):
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ Download Yahoo time series for a single base ticker.
        
            Arguments:
        """
        yahoo_tkr_obj = yf.Ticker(base_ticker)
        ts = yahoo_tkr_obj.history(start=start, end=end, period=period, interval=frequency)
        return ts

    # Overload superclass method
    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Standardize input arguments so the lower-level functions know what data type to expect.
        """
        # Call superclass method to parse arguments and make sure they are in a standardized format.
        start, end, frequency, period = super()._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)
        if frequency is None:
            frequency = '1d'
        return start, end, frequency, period

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        ts = super()._rename_time_series(ts, base_ticker)

        # Rename the columns
        col_map = {'Open' : f'{base_ticker}(PO)',
                   'Close' : f'{base_ticker}(PC)',
                   'High' : f'{base_ticker}(PH)',
                   'Low' : f'{base_ticker}(PL)',
                   'Volume' : f'{base_ticker}(VO)',
                   'Dividends' : f'{base_ticker}(DIV)',                   
                   'Stock Splits' : f'{base_ticker}(SS)',
                  }        
        ts = ts.rename(col_map, axis=1)
        return ts

    def _add_constructed_time_series(self, ts, base_ticker):
        """ Calculate the total return index from downloaded Yahoo data.
        """
        # Call superclass method
        ts = super()._add_constructed_time_series(ts, base_ticker)
        
        # Get the prices and dividends for the target security (ticker)
        price_ticker = f'{base_ticker}(PC)'
        div_ticker = f'{base_ticker}(DIV)'
        prices = ts[price_ticker]
        income = ts[div_ticker]

        # Drop dates where there is no data
        prices, income = prices.dropna().align(income.dropna(), axis=0)

        # Calculate the price returns
        price_rtns = -1 + prices / prices.shift(1).values
        price_rtns.iloc[0] = 0.0

        # Calculate the dividend returns
        div_rtns = income / prices.shift(1).values
        div_rtns.iloc[0] = 0.0

        # Calculate the total return index
        tr_index = np.cumprod(1 + price_rtns + div_rtns) * prices.iloc[0]
        tr_index.name = f'{base_ticker}(RI)'
        
        # Combine the old and new time series and return the result
        combined_ts = pd.concat([ts, tr_index], axis=1)
        return combined_ts


class CoinbaseProDownloader(AbstractDownloader):
    ALLOWED_GRANULARITY = (60, 300, 900, 3600, 21600, 86400,)
    MAX_POINTS_PER_REQUEST = 300
    MAX_REQUEST_PER_SECOND = 3
    DEFAULT_START_DATE = datetime.datetime(2015, 12, 31) 

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None,
                                  frequency=None, period=None):
        """ Download CoinbasePro time series for a single base ticker.
        
            Arguments:
                frequency: must be one of '1min', '5min', '15min', '1h', '6h', '1d'
        """
        client = cbp.PublicClient()
        
        # Get the 'granularity', which is number of seconds in one period with the given frequency
        granularity = int(pd.Timedelta(frequency).total_seconds())
        if granularity not in self.ALLOWED_GRANULARITY:
            raise ValueError(f'Granularity {granularity} is not in ' +\
                             f'range of allowed values {self.ALLOWED_GRANULARITY}')
        
        period_start = start
        period_end = min(period_start + datetime.timedelta(seconds=granularity * self.MAX_POINTS_PER_REQUEST), end)
        ts_list = []
        while period_start < end:
            # Make sure to space out the requests
            request_spacing = datetime.timedelta(seconds=1/self.MAX_REQUEST_PER_SECOND)

            success = False
            while not success:
                try:
                    # Try to get raw data from the API
                    raw_data = client.get_product_historic_rates(base_ticker, start=period_start, 
                                                stop=period_end, granularity=granularity)
                except cbp.exceptions.RateLimitError:
                    time.sleep(1/self.MAX_REQUEST_PER_SECOND)
                except :
                    print('handling error')
                else:
                    success = True

            # Convert the data into a pandas DataFrame object
            sub_ts = pd.DataFrame(raw_data).set_index('time')
            ts_list.append(sub_ts)

            # Update the start/end times
            period_start = period_end + datetime.timedelta(seconds=granularity)
            period_end  = min(period_start + datetime.timedelta(seconds=granularity * self.MAX_POINTS_PER_REQUEST), end)

        # Combine the time series into a single object
        ts = pd.concat(ts_list, axis=0)
        
        # Convert the data to a data frame and convert the columns to float
        for col in [ 'high', 'low', 'open', 'close', 'volume']:
            ts[col] = ts[col].astype('float')

        # Sort the dates in ascending order
        ts = ts.sort_index()
        return ts

    # Overload superclass method
    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Standardize input arguments so the lower-level functions know what data type to expect.
        
            Make sure start/end are datetime objects.
        """
        # Call superclass method to parse arguments and make sure they are in a standardized format.
        start, end, frequency, period = super()._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)

        # Make sure the 'start' argument is a datetime object
        if start is None:
            ST = self.DEFAULT_START_DATE
        elif not isinstance(start, (datetime.datetime, datetime.date)):
            ST = datetime.datetime.fromisoformat(start)
        else:
            ST = start

        # Make sure the 'end' argument is a datetime object
        if end is None:
            ET = datetime.datetime.now()
        elif not isinstance(end, (datetime.datetime, datetime.date)):
            ET = datetime.datetime.fromisoformat(end)
        else:
            ET = end

        # Make sure the frequency is not None
        if frequency is None:
            frequency = '1d'
            
        # Check if period is provided
        if period is not None:
            raise NotImplementedError(f'"period" has value {period} but logic is not implemented.')
            
        return ST, ET, frequency, period

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        # Call the superclass method
        ts = super()._rename_time_series(ts, base_ticker)

        # Get the map from the CBP names to our internal names
        series_type_map = pd.Series({'open' : 'PO', 'close' : 'PC', 
                                     'high' : 'PH', 'low' : 'PL', 'volume' : 'VO'})

        # Rename the columns
        series_types = series_type_map[ts.columns]
        tickers = [f'{base_ticker}({series_type})' for series_type in series_types]
        ts.columns = tickers
        return ts

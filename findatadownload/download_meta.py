import os
import datetime
import numpy as np
import pandas as pd

EODDATA_EXCHANGES = ['NYSE', 'NASDAQ', 'AMEX']


def _parse_eoddata_symbol_list(filename):
    """ Parse the meta data from EODDATA.com, and return a DataFrame. """

    with open(filename) as f:
        raw_data = f.read()

        # Separate the rows and columns
        parsed_data = [row.split('\t') for row in raw_data.split('\n')]

        # Drop the first row, which is the header
        parsed_data = parsed_data[1:]

        # Create a data frame
        symbols = pd.DataFrame(parsed_data, columns=['symbol', 'name']).dropna()
        return symbols
    
def _get_file_creation_time(filename):
    """ Get the time that the file was created.
    """
    if os.path.isfile(filename):
        mtime = os.path.getmtime(filename)
    else:
        mtime = 0

    last_modified_date = datetime.datetime.fromtimestamp(mtime)
    return last_modified_date
    
def combine_symbol_lists(base_path):
    """ Combine symbol lists from www.eoddata.com.
    """
    # Define a format string for the creation date
    fmt_str = '%Y-%m-%d'
    
    # Combine all symbol lists
    df = pd.DataFrame()
    for exchange in EODDATA_EXCHANGES:
        filename = f'{base_path}/{exchange}.txt'
        symbols = _parse_eoddata_symbol_list(filename)
        symbols['exchange'] = exchange
        
        # Get the time the file was created
        last_modified_date = _get_file_creation_time(filename)
        created_str = datetime.datetime.strftime(last_modified_date, fmt_str)
        symbols['valid_after'] = created_str

        # Combine the new symbols into the larger data frame
        df = pd.concat([df, symbols], axis=0)

    # Drop rows with an empty 'name' field
    idx_has_name = np.array([isinstance(x, str) and len(x) > 0 for x in df.name.values])
    df = df.loc[idx_has_name]

    # Drop duplicates
    df = df.drop_duplicates(['symbol', 'name'], keep='first')
    df = df.set_index('symbol')
    return df

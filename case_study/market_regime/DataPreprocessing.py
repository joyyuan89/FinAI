# Created: 23 Aug 2023
# Author: Jiayue

class DataPreprocessing:

    def __init__(self, data, base_index, rolling_windows, calculation_types, start_date, end_date):
        
        '''
        data: dataframe, raw market data, index = date, columns = indexs
        index_set: dataframe, columns = ['set', 'index']
        '''
        
        self.data = data
        self.base_index = base_index
        self.rolling_windows = rolling_windows
        self.calculation_types = calculation_types
        self.start_date = start_date
        self.end_date = end_date


    def trunc(self):

        if self.start_date is not None:
            self.data = self.data[self.data.index >= self.start_date]



        
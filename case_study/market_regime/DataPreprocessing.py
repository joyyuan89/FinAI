# Created: 23 Aug 2023
# Author: Jiayue

import pandas as pd

class DataPreprocessing:

    def __init__(self, data, base_index, rolling_windows, calculation_types, start_date, end_date, grouped, index_set, rounded, round_level):
        
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
        self.grouped = grouped
        self.index_set = index_set
        self.rounded = rounded
        self.round_level = round_level


    def select_period(self):

        if self.start_date is not None:
            self.data = self.data[self.data.index >= self.start_date]
        if self.end_date is not None:
            self.data = self.data[self.data.index <= self.end_date]

        # save start date and end date if not specified
        self.data = self.data.sort_index()
        self.start_date = self.data.index[0]
        self.end_date = self.data.index[-1]

    def preprocessing_scaler(self):
        self.scaled_data = self.data.div(self.data.iloc[0], axis=1) # Divide by the first row to normalize the starting point to 1
        self.scaled_data = self.scaled_data.div(self.scaled_data.loc[:, self.base_index], axis=0) # Divide by Base price to neutralize the effect of inflation

    def group_data(self): # scale data before grouping!!

        # select columns from raw data where column name is in index set['index']
        selected_data = self.scaled_data[self.index_set['index']]

        # caculate the mean value of columns if columns are in the same set
        li_set = self.index_set['set'].unique()
        for set in li_set:
            selected_data[set] = selected_data[self.index_set[self.index_set['set'] == set]['index']].mean(axis=1)
        
        self.grouped_data = selected_data[li_set]
    
    @staticmethod
    def round_to_n_levels(values,n): # round values into n levels from 0 to 1

        values = (values-values.min())/(values.max()- values.min()) # normalize
        values = (values * n).round(0) / n # round to n levels
        
        return values

    # main function
    def construct_market_data(self):
        
        
        self.select_period()
        self.preprocessing_scaler()

        if self.grouped == True and self.index_set is not None:
            self.group_data()
            df_raw = self.grouped_data

        else:
            df_raw = self.data

        df_derived = pd.DataFrame()  # Create an empty DataFrame to store derived values

        for col in df_raw.columns:
            for rolling_window in self.rolling_window:
                col_data = df_raw[col]  # Extract the column data to improve readability

                if "MA" in self.calculation_type:
                    ma_col_name = f'{col}_MA{rolling_window}'
                    ma_values = col_data - col_data.rolling(window=rolling_window).mean()
                    df_derived[ma_col_name] = (ma_values > 0).astype(int)  # Convert boolean to 0 or 1

                if "STD" in self.calculation_type:
                    std_col_name = f'{col}_STD{rolling_window}'
                    std_values = col_data.rolling(window=rolling_window).std() / col_data.rolling(window=rolling_window).mean()
                    if self.rounded:
                        df_derived[std_col_name] = DataPreprocessing.round_to_n_levels(values = std_values,n =round_level)
                    
                if "DIFF" in self.calculation_type:
                    diff_col_name = f'{col}_DIFF{rolling_window}'
                    diff_values = col_data.diff(rolling_window)
                    df_derived[diff_col_name] = (diff_values > 0).astype(int)  # Convert boolean to 0 or 1

        # Drop rows with missing values
        df_derived.dropna(axis=0, how='any', inplace=True)

        self.market_data = df_derived

        return self.market_data

    




        
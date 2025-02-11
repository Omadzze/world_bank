import pandas as pd


class DataProcessing:

    def __init__(self, data):
        self.data = data

    def missing_values(self):
        """
        This function calculates counts missing value and calculates percentage of missing values

        :return: Missing percentage of values in the dataset
        """
        # Count missing values in each column and sort in descending order
        missing_data = self.data.isnull().sum().sort_values(ascending=False)
        missing_percentage = self.data.isnull().mean().sort_values(ascending=False) * 100

        # Combine into a single DataFrame
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percentage.round(2)
        })

        print(missing_df)

        return missing_percentage

    def drop_missing_values_range(self, df_missing_percentage, threshold: float = 80):
        """
        Drop missing values in certain range of missing

        :param df_missing_percentage: dataframe with missing percentage from missing_values function
        :param threshold: percentage of missing

        :return: dataframe after removing columns with missing values
        """

        cols_drop = df_missing_percentage[df_missing_percentage > threshold].index

        print("Columns that was dropped:", cols_drop.tolist())

        # Drop these columns from the DataFrame
        df_cleaned = self.data.drop(columns=cols_drop)

        return df_cleaned

    

    '''
    def drop_highly_correlated(self, correlation_threshold=0.8, df_serise=None):
        """
        Drops one column from each pair of highly correlated columns by keeping the one with lower missing values.

        :param correlation_threshold: float, threshold above which two columns are considered highly correlated (default: 0.8)
        :param df_serise: dataset with cleaned missing values
        :return: Cleaned DataFrame with the highly correlated columns dropped.
        """

        missing_percentage_series = df_serise.isnull().mean() * 100

        # Select only numerical columns since we are computing correlation
        numeric_data = df_serise.select_dtypes(include=['number'])

        # Compute Corr matrix
        corr_matrix = numeric_data.corr().abs()

        #print(corr_matrix)

        # We'll use a set to collect the names of columns to drop
        cols_to_drop = set()

        # Loop through the upper triangle of the correlation matrix to check each pair
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                if corr_matrix.iloc[i, j] >= correlation_threshold:
                    print(f"Pair ({col1}, {col2}) has correlation: {corr_matrix.iloc[i, j]}")
                    # If both columns are still available, decide which one to drop
                    # Choose the column with the higher missing percentage
                    if missing_percentage_series[col1] > missing_percentage_series[col2]:
                        cols_to_drop.add(col1)
                    else:
                        cols_to_drop.add(col2)

        print("Columns dropped due to high correlation:", list(cols_to_drop))

        # Drop the selected columns from the data
        df_cleaned = df_serise.drop(columns=list(cols_to_drop))

        return df_cleaned
        '''




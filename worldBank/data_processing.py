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




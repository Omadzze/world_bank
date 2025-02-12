import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


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


    def visualize_iqr(self, dataframe: pd.DataFrame, n_columns: int):
        """
        Visualize the IQR with the skewness cacluation

        :param dataframe: Dataframe
        :param n_columns: number of columns to visualize at one row

        :return: Show visualized IQR
        """
        # Get all numeric columns
        numeric_columns = dataframe.select_dtypes(include=['number']).columns

        # Set up a grid with exactly 2 rows and as many columns as needed.
        n_cols = n_columns
        n_rows = math.ceil(len(numeric_columns) / n_cols)

        # Create the subplots with an appropriate figure size.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 9, n_rows * 8))
        axes = axes.flatten()

        for ax, column in zip(axes, numeric_columns):
            # Plot the histogram with KDE on the specific axis
            sns.histplot(dataframe[column], kde=True, bins=30, color='skyblue', ax=ax)

            # Calculate skewness
            skew_value = dataframe[column].skew()

            # Calculate Q1 and Q3
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)

            # Add vertical lines for Q1 and Q3 on the axis
            ax.axvline(Q1, color='red', linestyle='--', label='Q1')
            ax.axvline(Q3, color='green', linestyle='--', label='Q3')

            # Set title and labels on the axis
            ax.set_title(f"Distribution of {column}\nSkewness: {skew_value:.2f}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.legend()

        # Turn off any extra subplots that don't have data.
        for i in range(len(numeric_columns), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


    def apply_boxcox_for_skewed_columns(self, dataframe: pd.DataFrame, skew_threshold=0.8, offset=0.0):
        """
        Applies Box-Cox transformation to numeric columns in the DataFrame that are highly skewed.
        Positive skewness: +0.8
        Negative skewness: -0.7
        Normal distribution in range 0


        Parameters:
          dataframe (pd.DataFrame): Input DataFrame.
          skew_threshold (float): Absolute skewness threshold above which a column will be transformed.
                                  For example, if skewness is > 1.0 (or < -1.0), the transformation is applied.
          offset (float): Optional offset to add to a column if it contains non-positive values.
                          If 0 and non-positive values exist, an offset will be calculated automatically.

        Returns:
          df_transformed (pd.DataFrame): A copy of the original DataFrame with the Box-Cox transformations applied.
          transformation_details (dict): Dictionary with keys as column names and values as a dict containing:
              - 'lambda': the Box-Cox λ value,
              - 'initial_skew': skewness before transformation,
              - 'new_skew': skewness after transformation.
        """
        df_transformed = dataframe
        transformation_details = {}

        # Loop through numeric columns
        numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            initial_skew = df_transformed[col].skew()

            # Check if the absolute skewness exceeds the threshold
            if abs(initial_skew) > skew_threshold:
                print(f"\nTransforming column: {col}")

                # Prepare the data for Box-Cox: values must be strictly positive
                # If any values are <= 0, add an offset.
                if (df_transformed[col] <= 0).any():
                    # Calculate a minimal offset if none is specified
                    min_val = df_transformed[col].min()
                    effective_offset = offset if offset > 0 else abs(min_val) + 1
                    col_data = df_transformed[col] + effective_offset
                    print(f"  Added offset of {effective_offset} because minimum value was {min_val:.4f}")
                else:
                    col_data = df_transformed[col]

                # Apply the Box-Cox transformation
                try:
                    transformed_data, lam = stats.boxcox(col_data)
                except ValueError as e:
                    print(f"  Error transforming {col}: {e}")
                    continue

                # Replace the column data with the transformed data
                df_transformed[col] = transformed_data

                # Calculate new skewness
                new_skew = pd.Series(transformed_data).skew()

                transformation_details[col] = {
                    'lambda': lam,
                    'initial_skew': initial_skew,
                    'new_skew': new_skew
                }

                print(f"Box-Cox lambda: {lam:.4f}")
                print(f"Initial skew: {initial_skew:.4f}")
                print(f"New skew: {new_skew:.4f}")
            else:
                print(f"Column {col} skipped (skewness {initial_skew:.4f} within threshold)")

        return df_transformed, transformation_details

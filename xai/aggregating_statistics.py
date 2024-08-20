import python_utils.common_utils as cu
from sklearn.datasets import fetch_california_housing
input_path = "../dataset/interim/california_housing.csv"

# Fetch the California housing dataset
sklearn_data = fetch_california_housing()


def show_on_console(text, code):
    print('\n')
    print('-' * 150)
    print("◘ ", text)
    print(code)
    print('-' * 150)


def data_desc():
    # Print the dataset description
    print(sklearn_data['DESCR'][200:1420])


class AggregatingStatistics:
    def __init__(self):
        self.input_path = input_path

    def min_max(self, df):
        return df.max() - df.min()

    def aggregating_statistics(self):
        df = cu.load_df(ext_type='csv', library='pd', path=self.input_path, d_type=False)
        show_on_console("Dataset inspection", df.head(10).to_string())

        # Description for the original dataset
        data_desc()

        # Check the total count
        show_on_console("Count Samples", df.count())

        # Check for a specific numerical data type
        show_on_console("Display int and float types", df.select_dtypes(include=['int32', 'float64']))

        # Statistical information using transpose describe
        show_on_console("Describe dataset", df.describe().T.to_string())

        # Check for non-numerical data types
        show_on_console("Display object types", df.select_dtypes(include=['object']))

        # Round off all columns for a clear inspection
        df_round = df.round(0)

        for column in df.columns:
            # Values that are higher than average of all features
            show_on_console(f"Values higher than average for {column}:", df[[column]].where(df[column] >= df[column].mean())
                            .dropna().value_counts())

            # Unique values for rounded dataset
            show_on_console(f"Rounded unique values for {column}: ", df_round[column].unique())

            # Count the unique values
            show_on_console(f"Rounded unique values for {column}: ", df_round[column].nunique())

        # Show median income for house age higher than specific years per block
        house_age = [10, 20, 30, 40, 50]
        for age in house_age:
            print(f"house age >= {age}", df[['medinc']].where(df['houseage'] >= age).dropna().mean())
            # Result: House age >= 50 & 10 has more income per block, whereas 40< age <50 has the lowest income

        # Show median income for higher & lower than average rooms per house
        show_on_console(f"Income for higher than average of {round(df['averooms'].mean(), 0)} rooms/house:", df[['medinc']].
                        where(df['averooms'] >= df['averooms'].mean()).dropna().mean())

        show_on_console(f"Income for lower than average of {round(df['averooms'].mean(), 0)} rooms/house:", df[['medinc']].
                        where(df['averooms'] <= df['averooms'].mean()).dropna().mean())
        # Result: Income for higher than average room/house is 5.019 whereas for lower is 2.99

        # Income in the most & least average population per block
        show_on_console(f"Income in most average population of {round(df['population'].mean(), 0)} per block:",
                        df[['medinc']].where(df['population'] >= df['population'].mean()).dropna().mean())

        show_on_console(f"Income in least average population of {round(df['population'].mean(), 0)} per block:",
                        df[['medinc']].where(df['population'] <= df['population'].mean()).dropna().mean())
        # Result: Previously, we concluded that three-quarter of income is 4.74,
        # Income is 3.9 when less than average population per block,
        # indicating the wealthier class resides in a more solitary zone

        # Average number of household members based on 1/3, 2/4, & 3/4 quantile of median income per block group
        quantile_values = [0.25, 0.5, 0.75]
        for q in quantile_values:
            show_on_console(f"Average occupants above {q} quantile of median income:",
                            df[['aveoccup']].where(df['medinc'] >= df['medinc'].quantile([q][0])).dropna().mean())
            # Result: 0.75 quantile with 3.30 when compared to 0.25 with 3.06 and 0.5 with 3.14,
            # However, the discrepancy is surprisingly minor despite the logical interpretation

        # Average median income at each rounded coordinate of the location
        show_on_console(f"Median income at center:",
                        df_round['medinc'].groupby([df_round['latitude'], df_round['longitude']]).mean())
        # At rounded latitude of 33 & rounded longitude of -118, the highest average median income is 5
        # Upon further investigation, the wealthiest zone is located
        # approximately at the Imperial Sand Dunes Recreation Area in California.

        for age in house_age:
            # Average room count based on house age per block
            print(f"house age >= {age}", df[['averooms']].where(df['houseage'] >= age).dropna().mean())
            # Result: House age higher than 40 & lower than 50 has the least average rooms with 4.95
            # On the contrary, age less than 20 and higher than 10 has the highest room count of 5.37

            # Population count based on house age per block
            print(f"house age >= {age}", df[['population']].where(df['houseage'] >= age).dropna().mean())
            # Result: 40< age <50 has the lowest population with 940.5 whereas 20< age <30 has the highest with 1267.6

            # Average occupants based on house age per block
            print(f"house age >= {age}", df[['aveoccup']].where(df['houseage'] >= age).dropna().mean())
            # Result: 40< age <50 has the lowest occupants with 2.82 whereas 30< age <40 has the highest with 3.31

        # MEDHOUSEVAL
        for q in quantile_values:
            # Δ average income based on Median house value (quantile)
            show_on_console(f"Income when house price is higher than {q} quantile:",
                            df[['medinc']].where(df['medhouseval'] >= df['medhouseval'].quantile([q][0])).mean())
            # Result: q1 = 4.3, q2 = 4.8, q3 = 5.7, confirming a logical linear relationship

            # Average age factor based on Median house value (quantile)
            show_on_console(f"Average Age when house price is higher than {q} quantile:",
                            df[['houseage']].where(df['medhouseval'] >= df['medhouseval'].quantile([q][0])).mean())
            # Result: Age at q1 = 28.7, age at q2 = 29.5, and age at q3 = 30.6
            # Indicating that new houses are cheaper than old house

            # Population vs Price (quantile)
            show_on_console(f"Population when house price is higher than {q} quantile:",
                            df[['population']].where(df['medhouseval'] >= df['medhouseval'].quantile([q][0])).mean())
            # Result: Age at q1 = 1476.0, age at q2 = 1426.0, and age at q3 = 1341.0
            # More population are residing at cheap house value per block


if __name__ == "__main__":
    main = AggregatingStatistics()
    main.aggregating_statistics()
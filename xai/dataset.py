import pandas as pd
from sklearn.datasets import fetch_california_housing

# Fetch the California housing dataset
sklearn_data = fetch_california_housing()


def data_desc():
    # Print the dataset description
    print(sklearn_data['DESCR'][200:1420])


class Dataset:
    def __init__(self, output_path="../dataset/raw/california_housing.csv"):
        # Set output path
        self.output_path = output_path

    def assemble_data(self):
        # Assemble the features and labels into a DataFrame
        features = pd.DataFrame(sklearn_data['data'], columns=sklearn_data['feature_names'])
        label = pd.Series(sklearn_data['target'], name='MedHouseVal')

        df_housing_data = pd.concat([features, label], axis=1)

        # Print the first 10 rows of the DataFrame
        print(df_housing_data.head(10).to_string())

        # Store the DataFrame as a CSV file
        self.store_data(df_housing_data)

    def store_data(self, df):
        try:
            df.to_csv(self.output_path, sep=',', index=False)
        except IOError as e:
            print(f"File I/O error: {e}")
        except Exception as exc:
            print(f"An error occurred: {exc}")
        else:
            print("Conversion successful.")


if __name__ == "__main__":
    # Create an instance of the Dataset class
    main = Dataset()

    # Output the description of the dataset
    data_desc()

    # Assemble the dataset and store it as a CSV file
    main.assemble_data()

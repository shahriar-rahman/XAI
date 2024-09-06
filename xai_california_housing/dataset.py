import pandas as pd
from sklearn.datasets import fetch_california_housing

# Fetching the California Housing dataset from sklearn datasets
dataset = fetch_california_housing()


class Dataset:
    def __init__(self, output_path='../dataset/raw/california_housing.csv'):
        self.output_path = output_path

    def store_dataset(self, df):
        # Save the df to a CSV file
        try:
            df.to_csv(self.output_path, sep=',', index=False)
        except IOError as e:
            print(f"File I/O error: {e}")
        except Exception as exc:
            print(f"An error occurred: {exc}")
        else:
            print("Conversion successful.")

    def assemble_data(self):
        # Prepare and store the dataset
        features = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
        label = pd.DataFrame(dataset['target'], columns=dataset['target_names'])

        df = pd.concat([features, label], axis=1)
        self.store_dataset(df)


if __name__ == "__main__":
    main = Dataset()
    main.assemble_data()

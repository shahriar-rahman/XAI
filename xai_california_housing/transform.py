import pandas as pd
from plots import Plots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import warnings
warnings.filterwarnings("ignore")

# GUIDE FOR KMEANS mechanisms
# Inertia measures how well a dataset was clustered by K-Means.
# It is calculated by measuring the distance between each data point and its centroid, squaring this distance,
# and summing these squares across one cluster.
# A good model is one with low inertia AND a low number of clusters (K).
# However, this is a tradeoff because as K increases, inertia decreases.
# Distortion = 1/n * Σ(distance(point, centroid)^2)
# Inertia = Σ(distance(point, centroid)^2)


def show_on_console(text_1, text_2):
    # Console print
    print('-'*100)
    print('◘', text_1)
    print(text_2)


def calculate_outliers(df):
    # Calculate the outliers on the selected DataFrame
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    iqr = q3 - q1

    # Upper = q3 + (1.5 * iqr), lower = q1 - (1.5 * iqr)
    outliers = df[((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)))]

    show_on_console('outliers count', str(len(outliers)))
    show_on_console('max outlier', str(outliers.max()))
    show_on_console('min outlier', str(outliers.min()))


class Transformation:
    def __init__(self, input_path='../dataset/raw/california_housing.csv',
                 output_path='../dataset/processed/california_housing.csv', fig_path='../reports/figures/'):
        self.plot = Plots()
        self.fig_path = fig_path

        self.input_path = input_path
        self.output_path = output_path
        self.df = pd.read_csv(self.input_path)

    def inspect_data(self):
        # Observe the initial loaded data
        self.df.info()
        show_on_console('Data description', self.df.describe().to_string())
        self.plot.multi_hist(self.df, ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                                       'MedHouseVal'], path=self.fig_path, bins=30)

        for column in self.df.columns:
            show_on_console(f'{column} outliers:', calculate_outliers(self.df))

        self.plot.missingno(self.df, kind='matrix', path=self.fig_path)

    def coordinate_clustering(self):
        # Apply clustering method to the geographic coordinates
        locations = self.df[['Longitude', 'Latitude']]

        # Sum of squared distances / Inertia
        sse = []
        distortions = []
        k_range = range(1, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(locations)

            # Average distortion (inertia / number of samples)
            distortion = kmeans.inertia_ / len(locations)

            # Calculate Δ for distortion, handle case for k = 1
            if k == 1:
                delta = None
            else:
                delta = distortions[-1] - distortion

            sse.append(kmeans.inertia_)
            distortions.append(distortion)

            show_on_console(f'K = {k}: Inertia = {kmeans.inertia_:.4f}, Average Distortion = {distortion:.4f}',
                            f'Δ Distortion = {delta:.4f}' if delta is not None else '')

        # Apply the 'elbow method' for diagnosis
        plt.figure(figsize=(10, 8))
        plt.plot(k_range, sse, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Distances (SSE)')
        plt.title('Elbow Method for Optimal K')
        plt.savefig(f"{self.fig_path}kmeans_elbow method.png")
        plt.show()

        # Optimal K is determined to be 4
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['Location_Cluster'] = kmeans.fit_predict(locations)

        # Plot the results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Longitude', y='Latitude', hue='Location_Cluster', data=self.df, palette='Set1')
        plt.title('Clusters of Locations')
        plt.savefig(f"{self.fig_path}kmeans_cluster_coordinates.png")
        plt.show()

        self.df = self.df.drop(columns=['Longitude', 'Latitude'])
        print(self.df.head(10).to_string())

    def feature_scaling(self):
        # Apply scaling
        numerical_features = self.df.drop(columns=['Location_Cluster']).columns.tolist()

        # Column transformer for scaling numerical features and encoding 'Location_Cluster'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), ['Location_Cluster'])
            ])

        x_preprocessed = preprocessor.fit_transform(self.df)

        scaled_feature_names = numerical_features
        onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['Location_Cluster'])
        all_feature_names = scaled_feature_names + onehot_feature_names.tolist()

        # Convert the result back to a DataFrame (optional)
        processed_df = pd.DataFrame(x_preprocessed, columns=all_feature_names)
        print(processed_df.head(10).to_string())

        # Save the dataframe to storage
        try:
            processed_df.to_csv(self.output_path, sep=',', index=False)
        except IOError as e:
            print(f"File I/O error: {e}")
        except Exception as exc:
            print(f"An error occurred: {exc}")
        else:
            print("Conversion successful.")


if __name__ == "__main__":
    main = Transformation()
    main.inspect_data()
    main.coordinate_clustering()
    main.feature_scaling()


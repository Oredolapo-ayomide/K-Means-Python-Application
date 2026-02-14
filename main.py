import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def load_data():
    try:
        file_path = "data/PrePostQuestions.csv"
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!\n")
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None


def preprocess_text(data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(data['question'])
    print("Text successfully converted into numerical features.\n")
    return X


def perform_kmeans(X, data):
    """
    Apply K-Means clustering and attach cluster labels
    """
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    print("K-Means clustering completed.")
    print("Inertia:", kmeans.inertia_)

    # Add cluster labels
    data['cluster'] = kmeans.labels_

    return data


def main():
    data = load_data()

    if data is not None:
        X = preprocess_text(data)
        print("Shape of transformed data:", X.shape)

        clustered_data = perform_kmeans(X, data)

        print("\nNumber of questions in each cluster:")
        print(clustered_data['cluster'].value_counts())

        print("\nSample questions from each cluster:\n")

        for cluster_num in clustered_data['cluster'].unique():
            print(f"\nCluster {cluster_num} sample questions:")
            print(clustered_data[clustered_data['cluster'] == cluster_num]['question'].head(5))

if __name__ == "__main__":
    main()


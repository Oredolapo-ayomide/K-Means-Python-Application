import unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class TestKMeansApplication(unittest.TestCase):

    def setUp(self):
        """
        Create small sample dataset for testing
        """
        self.sample_data = pd.DataFrame({
            "question": [
                "Is this product safe?",
                "Does this fit my laptop?",
                "What material is this made of?"
            ]
        })

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.sample_data['question'])

    def test_preprocessing_shape(self):
        """
        Test that preprocessing returns correct shape
        """
        self.assertEqual(self.X.shape[0], 3)

    def test_kmeans_clusters(self):
        """
        Test that KMeans creates correct number of clusters
        """
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(self.X)

        self.assertEqual(kmeans.n_clusters, 2)


if __name__ == '__main__':
    unittest.main()

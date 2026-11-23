"""
Music Recommendation System - Automated Test Suite
Validates feature matrix dimensions, similarity bounds, and recommendation logic.
"""

import unittest
import numpy as np
from src.recommender import MusicRecommender


class TestMusicRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recommender = MusicRecommender()

    def test_01_catalog_loaded(self):
        """Verifies dataset loaded songs cleanly."""
        songs = self.recommender.list_available_songs()
        self.assertGreater(len(songs), 0, "Song catalog should not be empty.")

    def test_02_similarity_matrix_properties(self):
        """Verifies mathematical properties of cosine similarity matrix."""
        sim = self.recommender.similarity_matrix
        n = len(self.recommender.df)
        self.assertEqual(sim.shape, (n, n), "Similarity matrix must be N x N.")
        # Diagonal must be 1.0 (a song is 100% similar to itself)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_03_song_recommendations(self):
        """Verifies song recommendations return valid match scores and ignore self."""
        first_song = self.recommender.list_available_songs()[0]
        recs = self.recommender.get_song_recommendations(first_song, top_n=3)
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertNotEqual(r["song"], first_song, "Recommendation should not include the query song itself.")
            self.assertTrue(0.0 <= r["similarity"] <= 100.0, "Similarity match percentage out of bounds.")

    def test_04_mood_filtering(self):
        """Verifies mood playlist returns songs matching target mood."""
        mood_recs = self.recommender.get_mood_recommendations("sad", top_n=3)
        self.assertGreater(len(mood_recs), 0)
        for m in mood_recs:
            self.assertEqual(m["mood"].lower(), "sad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
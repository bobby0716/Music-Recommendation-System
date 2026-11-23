"""
Music Recommendation System - Content-Based Recommendation Engine
Combines normalized acoustic audio features with genre/artist TF-IDF vectorization
to compute multi-dimensional cosine similarity matrices.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA_PATH = os.path.join("data", "songs.csv")


class MusicRecommender:
    ACOUSTIC_FEATURES = [
        "Dance", "Energy", "Acoustic", "Happy", "BPM", "Popularity"
    ]

    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.df = self._load_and_clean_data()
        self.feature_matrix = self._build_feature_matrix()
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def _load_and_clean_data(self) -> pd.DataFrame:
        """Loads CSV and standardizes columns and nulls."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Standardize strings
        df["Song"] = df["Song"].astype(str).str.strip()
        df["Artist"] = df["Artist"].astype(str).str.strip()
        df["Genres"] = df["Genres"].fillna("").astype(str).str.strip()
        df["Mood"] = df["Mood"].fillna("neutral").astype(str).str.lower().str.strip()

        # Ensure acoustic numeric columns exist and are numeric
        for col in self.ACOUSTIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            else:
                df[col] = 0

        # Create search index (lowercase song title)
        df["search_title"] = df["Song"].str.lower()
        return df

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Builds unified feature representation:
        1. Scaled acoustic features (MinMax to 0-1 range).
        2. TF-IDF representation of Genres & Artists.
        """
        # 1. Scale numeric features
        scaler = MinMaxScaler()
        scaled_acoustics = scaler.fit_transform(self.df[self.ACOUSTIC_FEATURES])

        # 2. Text vectorization on combined artist + genres
        combined_text = self.df["Artist"] + " " + self.df["Genres"]
        tfidf = TfidfVectorizer(stop_words="english", max_features=100)
        text_features = tfidf.fit_transform(combined_text).toarray()

        # 3. Concatenate acoustics (weight 60%) + genre/artist text (weight 40%)
        combined_matrix = np.hstack([scaled_acoustics * 0.60, text_features * 0.40])
        return combined_matrix

    def get_song_recommendations(self, song_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Given a song title, calculates cosine distance and returns top N most similar songs.
        """
        query = song_name.lower().strip()
        matches = self.df[self.df["search_title"].str.contains(query, regex=False)]

        if matches.empty:
            raise ValueError(f"Song '{song_name}' not found in dataset.")

        idx = matches.index[0]
        original_song = self.df.iloc[idx]

        # Get similarity vector for this song
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort descending by similarity score (ignoring self at position 0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

        recommendations = []
        for song_idx, score in sim_scores:
            row = self.df.iloc[song_idx]
            recommendations.append({
                "song": row["Song"],
                "artist": row["Artist"],
                "mood": row["Mood"].capitalize(),
                "similarity": round(float(score) * 100, 1),
                "energy": row["Energy"],
                "danceability": row["Dance"],
                "bpm": row["BPM"],
                "spotify_id": row.get("Spotify Track Id", ""),
            })

        return recommendations

    def get_mood_recommendations(self, mood: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Returns top popular songs matching an emotional vibe.
        """
        target = mood.lower().strip()
        filtered = self.df[self.df["Mood"] == target].copy()

        if filtered.empty:
            return []

        # Sort by popularity
        filtered = filtered.sort_values(by="Popularity", ascending=False).head(top_n)

        results = []
        for _, row in filtered.iterrows():
            results.append({
                "song": row["Song"],
                "artist": row["Artist"],
                "mood": row["Mood"].capitalize(),
                "popularity": row["Popularity"],
                "bpm": row["BPM"],
                "spotify_id": row.get("Spotify Track Id", ""),
            })
        return results

    def list_available_songs(self) -> List[str]:
        """Returns sorted list of all unique song titles."""
        return sorted(self.df["Song"].unique().tolist())

    def list_available_moods(self) -> List[str]:
        """Returns unique moods in dataset."""
        return sorted(self.df["Mood"].unique().tolist())
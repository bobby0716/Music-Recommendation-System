### SoundVibe AI - Content-Based Music Recommendation System

An intelligent content-based music recommendation engine that maps songs into multi-dimensional acoustic vector space, computes Cosine Similarity across Spotify audio features and genre embeddings, and delivers personalized recommendations through an interactive Streamlit Web Application and Console CLI.

## Features
1.Track-to-Track Recommendations: Input any track to receive the top N closest acoustic matches with similarity percentages.

2.Mood-Based Curation: Curate playlists for Happy, Sad, Calm, and Energetic vibes.

3.Dual Delivery Interface:
   Streamlit Web UI: Spotify-dark aesthetic, recommendation cards, and direct Spotify links.
   Console CLI: Fast terminal gameplay.

4.Automated Unit Testing: Comprehensive unittest suite validating similarity matrix symmetry, diagonal bounds, and recommendation logic.

### Project Structure
text
Music-Recommendation-System/
├── data/
│   └── songs.csv              # Track catalog with Spotify acoustic features
├── src/
│   ├── __init__.py
│   └── recommender.py         # Cosine similarity engine & feature scaler
├── tests/
│   ├── __init__.py
│   └── test_recommender.py    # Unit test suite
├── app.py                     # Streamlit web application
├── cli.py                     # Console interface
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Enforces clean version control
└── README.md                  # Project documentation


## Quickstart Guide

1. Install Dependencies

pip install -r requirements.txt

2. Launch Streamlit Web App

streamlit run app.py

3. Launch Console CLI

python cli.py

4. Run Automated Tests

python -m unittest tests.test_recommender -v

## Running the Tests & App
Run the test suite:

powershell
python -m unittest tests.test_recommender -v

Launch Streamlit:

powershell
streamlit run app.py
"""
Music Recommendation System - Streamlit Web Dashboard
Interactive Spotify-styled interface for content-based music discovery and mood playlists.
"""

import streamlit as st
import pandas as pd
from src.recommender import MusicRecommender

# Page Configuration
st.set_page_config(
    page_title="SoundVibe - Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# Custom Styling (Dark Spotify Aesthetic)
st.markdown(
    """
    <style>
    .main { background-color: #121212; color: #FFFFFF; }
    .song-card {
        background-color: #181818;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #1DB954;
        margin-bottom: 12px;
    }
    .match-pill {
        background-color: #1DB954;
        color: #000000;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    return MusicRecommender()


recommender = load_engine()

# Title Header
st.title("🎵 SoundVibe AI")
st.caption("Content-Based Music Recommendation Engine Powered by Scikit-Learn Cosine Similarity")
st.markdown("---")

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🎧 Similar Song Recommender",
    "🎭 Mood Playlist Generator",
    "📊 Playlist Audio Analytics",
])

# ------------------------------------------------------------------------------
# TAB 1: SONG SIMILARITY
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Discover Songs Similar to Your Favorite Track")
    songs_list = recommender.list_available_songs()

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_song = st.selectbox("Select a Song:", songs_list, index=0)
    with col2:
        top_k = st.slider("Recommendations Count:", min_value=3, max_value=10, value=5)

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Calculating multi-dimensional acoustic similarity..."):
            try:
                recs = recommender.get_song_recommendations(selected_song, top_n=top_k)
                st.success(f"Top {len(recs)} tracks acoustically similar to **{selected_song}**:")

                for idx, song in enumerate(recs, 1):
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="song-card">
                                <h3>#{idx} {song['song']} <span class="match-pill">{song['similarity']}% Match</span></h3>
                                <p><strong>Artist:</strong> {song['artist']} | <strong>Mood:</strong> {song['mood']} | <strong>Tempo:</strong> {song['bpm']} BPM</p>
                                <p><strong>Danceability:</strong> {song['danceability']} | <strong>Energy:</strong> {song['energy']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if song.get("spotify_id"):
                            st.link_button(
                                "▶️ Open in Spotify",
                                f"https://open.spotify.com/track/{song['spotify_id']}"
                            )
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------------------------------------------------------------------
# TAB 2: MOOD PLAYLIST
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Curate a Playlist by Mood & Vibe")
    moods = recommender.list_available_moods()

    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        selected_mood = st.selectbox("Select Your Current Mood:", [m.capitalize() for m in moods])
    with m_col2:
        mood_k = st.slider("Playlist Size:", min_value=3, max_value=12, value=6, key="mood_k")

    if st.button("Generate Mood Playlist", type="primary"):
        mood_recs = recommender.get_mood_recommendations(selected_mood, top_n=mood_k)
        if not mood_recs:
            st.warning(f"No tracks found for mood: {selected_mood}")
        else:
            st.write(f"Showing top **{selected_mood}** tracks ranked by popularity:")
            mood_df = pd.DataFrame(mood_recs)[["song", "artist", "popularity", "bpm"]]
            st.dataframe(mood_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: PLAYLIST ANALYTICS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Acoustic Feature Exploration")
    df = recommender.df

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("##### Mood Distribution Across Catalog")
        mood_counts = df["Mood"].value_counts()
        st.bar_chart(mood_counts)

    with a2:
        st.markdown("##### Tempo (BPM) Distribution")
        st.bar_chart(df["BPM"].value_counts().sort_index())
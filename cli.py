"""
Music Recommendation System - Console Interface
"""

from src.recommender import MusicRecommender


def run_cli():
    print("=" * 60)
    print("    SOUNDVIBE - CONTENT-BASED MUSIC RECOMMENDER")
    print("=" * 60)
    
    recommender = MusicRecommender()
    songs = recommender.list_available_songs()
    print(f"[INFO] Catalog loaded: {len(songs)} tracks ready.\n")

    while True:
        print("\nChoose an Option:")
        print(" [1] Recommend Similar Songs by Track Name")
        print(" [2] Discover Songs by Mood (Happy, Sad, Calm)")
        print(" [3] List All Available Songs")
        print(" [4] Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            query = input("\nEnter song title (e.g. Saiyaara, Sahiba): ").strip()
            try:
                recs = recommender.get_song_recommendations(query, top_n=5)
                print(f"\n🎧 Top Recommendations for '{query}':")
                for idx, r in enumerate(recs, 1):
                    print(f"  {idx}. {r['song']} - {r['artist']} (Match: {r['similarity']}%, Mood: {r['mood']})")
            except Exception as e:
                print(f"[ERROR] {e}")

        elif choice == "2":
            mood = input("\nEnter mood (happy, sad, calm): ").strip()
            recs = recommender.get_mood_recommendations(mood, top_n=5)
            print(f"\n🎭 Top {mood.capitalize()} Songs:")
            for idx, r in enumerate(recs, 1):
                print(f"  {idx}. {r['song']} - {r['artist']} (Popularity: {r['popularity']})")

        elif choice == "3":
            print("\nAvailable Tracks in Database:")
            for s in songs:
                print(f"  - {s}")

        elif choice == "4":
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_cli()
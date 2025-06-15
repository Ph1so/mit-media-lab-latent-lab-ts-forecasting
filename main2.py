import os
import csv
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# --- Load API Token ---
load_dotenv()
TMDB_API_TOKEN = os.getenv("TMDB-API-READ-ACCESS-TOKEN")
if not TMDB_API_TOKEN:
    raise RuntimeError("TMDB-API-READ-ACCESS-TOKEN not set in .env")

# --- Genre Dictionaries ---
MOVIE_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}
TV_GENRES = {
    10759: "Action & Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary",
    18: "Drama", 10751: "Family", 10762: "Kids", 9648: "Mystery", 10763: "News", 10764: "Reality",
    10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics", 37: "Western"
}

# --- Load CSV Data ---
def load_netflix_data(filepath, limit=None):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Title', 'Date'], inplace=True)
    if limit:
        df = df.head(limit)
    return df['Title'].tolist(), df['Date'].tolist()

# --- TMDB Search ---
def use_TMDB(title):
    url = f"https://api.themoviedb.org/3/search/multi?query={requests.utils.quote(title)}&include_adult=false&language=en-US&page=1"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

# --- Resolve Overview and Genres ---
def resolve_tmdb_data(original_title):
    current_title = original_title
    overview = "No overview available"
    genres = ["NA"]

    while current_title:
        print(f"🔍 Trying: {current_title}")
        data = use_TMDB(current_title)
        results = data.get("results", [])
        if results:
            result = results[0]
            overview = result.get("overview", overview)
            poster_path = result.get("poster_path", "NA")
            genre_ids = result.get("genre_ids", [])
            genre_map = TV_GENRES if result.get("media_type") == "tv" else MOVIE_GENRES
            genres = [genre_map.get(gid, "Unknown") for gid in genre_ids] or ["NA"]
            break
        if ":" not in current_title:
            break
        current_title = ":".join(current_title.split(":")[:-1]).strip()

    print(f"✅ Final title used: {current_title}")
    return original_title, current_title, overview, genres, poster_path

# --- CSV Writer ---
def create_csv_with_TMDB_threaded(input_path, output_path, test_mode=False):
    titles, dates = load_netflix_data(input_path, limit=100 if test_mode else None)

    overviews = [None] * len(titles)
    genres_list = [None] * len(titles)
    poster_paths = [None] * len(titles)

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(resolve_tmdb_data, title): i for i, title in enumerate(titles)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, _, overview, genres, poster_path = future.result()
                overviews[i] = overview
                genres_list[i] = genres
                poster_paths[i]= poster_path
            except Exception as e:
                print(f"⚠️ Error processing {titles[i]}: {e}")
                overviews[i] = "No overview available"
                genres_list[i] = ["NA"]

    with open(output_path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Overview", "Genres", "Poster_Path", "Date"])
        for title, overview, genre, poster_path, date in zip(titles, overviews, genres_list, poster_paths, dates):
            writer.writerow([title, overview, genre, poster_path, date])

    print(f"✅ CSV file '{output_path}' created.")

# --- Entrypoint ---
def main(input_file="NetflixViewingHistory.csv", output_file="cleaned_netflix_data.csv", test_mode=False):
    create_csv_with_TMDB_threaded(input_file, output_file, test_mode)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate overview-enriched Netflix data CSV.")
    parser.add_argument("--input", type=str, default="./data/NetflixViewingHistory.csv", help="Path to input Netflix CSV")
    parser.add_argument("--output", type=str, default="cleaned_netflix_data.csv", help="Output CSV file path")
    parser.add_argument("--test", action="store_true", help="Limit to first 100 entries for testing")
    args = parser.parse_args()
    main(input_file=args.input, output_file=args.output, test_mode=args.test)
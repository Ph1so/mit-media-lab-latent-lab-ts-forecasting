import os
import csv
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Load API Key ---
load_dotenv()
TMDB_API_TOKEN = os.getenv("TMDB-API-READ-ACCESS-TOKEN")
if not TMDB_API_TOKEN:
    raise RuntimeError("TMDB-API-READ-ACCESS-TOKEN not set in .env")

# --- Genre Maps ---
MOVIE_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}
TV_GENRES = {
    10759: "Action & Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 10762: "Kids", 9648: "Mystery",
    10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
    10767: "Talk", 10768: "War & Politics", 37: "Western"
}
P31_IDS = {
    'Q11424', 'Q5398426', 'Q1259759', 'Q15416', 'Q7889', 'Q526877', 'Q12737077',
    'Q100269041', 'Q102364578', 'Q202866', 'Q24862', 'Q24856', 'Q196600',
    'Q431289', 'Q1667921', 'Q3464665', 'Q20650540', 'Q21191270', 'Q117467246',
    'Q1261214', 'Q98807719', 'Q106594041', 'Q63952888', 'Q17517379'
}

# --- API Utilities ---
def get_wiki_id(title):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": title,
        "srnamespace": 0
    }
    r = requests.get(url, params=params).json()
    for result in r.get("query", {}).get("search", []):
        entity_id = result.get("title")
        if validate_p31(entity_id):
            return entity_id
    return None

def validate_p31(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    r = requests.get(url).json()
    claims = r["entities"].get(entity_id, {}).get("claims", {})
    p31_list = claims.get("P31", [])
    for claim in p31_list:
        id = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
        if id in P31_IDS:
            return True
    return False

def get_tmdb_data(wikidata_id):
    url = f"https://api.themoviedb.org/3/find/{wikidata_id}?external_source=wikidata_id"
    headers = {"Authorization": f"Bearer {TMDB_API_TOKEN}", "accept": "application/json"}
    r = requests.get(url, headers=headers)
    return r.json()

# --- Main Fetcher ---
def fetch_overview_and_genre(title):
    original_title = title
    current_title = title
    overview = "No overview available"
    genres = ["NA"]
    while ":" in current_title:
        # print(f"🔍 Trying: {current_title}")
        wiki_id = get_wiki_id(current_title)
        if not wiki_id:
            current_title = ":".join(current_title.split(":")[:-1]).strip()
            continue

        data = get_tmdb_data(wiki_id)
        for key in ["movie_results", "tv_results", "tv_episode_results", "tv_season_results"]:
            if data.get(key):
                result = data[key][0]
                if result.get("overview"):
                    overview = result["overview"]
                    genre_ids = result.get("genre_ids", [])
                    genre_map = TV_GENRES if key.startswith("tv") else MOVIE_GENRES
                    genres = [genre_map.get(i, "NA") for i in genre_ids] or ["NA"]
                    return original_title, overview, genres
        current_title = ":".join(current_title.split(":")[:-1]).strip()
    return original_title, overview, genres

# --- Load Netflix Viewing History ---
def load_netflix_csv(filepath="NetflixViewingHistory.csv", limit=None):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Title", "Date"])
    if limit is not None:
        df = df.head(limit)
    return df["Title"].tolist(), df["Date"].tolist()

# --- CSV Output Writer ---
def write_csv(titles, overviews, genres_list, dates, outpath="cleaned_netflix_data.csv"):
    with open(outpath, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Overview", "Genres", "Date"])
        for title, overview, genre, date in zip(titles, overviews, genres_list, dates):
            writer.writerow([title, overview, genre, date])
    print(f"✅ CSV saved to: {outpath}")

# --- Main Function ---
def main(input_file="NetflixViewingHistory.csv", output_file="cleaned_netflix_data.csv", test_mode=False):
    limit = 100 if test_mode else None
    titles, dates = load_netflix_csv(filepath=input_file, limit=limit)
    overviews = [None] * len(titles)
    genres_list = [None] * len(titles)

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(fetch_overview_and_genre, title): i for i, title in enumerate(titles)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                title, overview, genres = future.result()
                print(title, overview, genres)
                overviews[i] = overview
                genres_list[i] = genres
            except Exception as e:
                print(f"⚠️ Error processing '{titles[i]}': {e}")
                overviews[i] = "No overview available"
                genres_list[i] = ["NA"]

    write_csv(titles, overviews, genres_list, dates, outpath=output_file)

# --- Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate overview-enriched Netflix data CSV.")
    parser.add_argument("--input", type=str, default="NetflixViewingHistory2.csv", help="Path to input Netflix CSV")
    parser.add_argument("--output", type=str, default="cleaned_netflix_data.csv", help="Output CSV file path")
    parser.add_argument("--test", action="store_true", help="Limit to first 100 entries for testing")

    args = parser.parse_args()
    main(input_file=args.input, output_file=args.output, test_mode=args.test)

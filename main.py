import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from nomic import embed
import csv

load_dotenv()
TMDB_API_TOKEN = os.getenv("TMDB-API-READ-ACCESS-TOKEN") #note: for some reason in .env, if you do TMDB_API_READ_ACCESS_TOKEN it replaces - with _ in the value (dunno why)
if not TMDB_API_TOKEN:
    raise RuntimeError("TMDB-API-READ-ACCESS-TOKEN not set in .env")

MOVIE_GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}
TV_GENRES = {
    10759: "Action & Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    10762: "Kids",
    9648: "Mystery",
    10763: "News",
    10764: "Reality",
    10765: "Sci-Fi & Fantasy",
    10766: "Soap",
    10767: "Talk",
    10768: "War & Politics",
    37: "Western"
}

# valid p31 ids that give TMDB overviews
P31_IDS = {'Q63952888', 'Q5398426', 'Q106594041', 'Q506240', 'Q98807719', 'Q12737077', 'Q1259759', 'Q526877', 'Q11424', 'Q431289', 'Q7889', 'Q20650540', 'Q117467246', 'Q15416', 'Q102364578', 'Q24856', 'Q100269041', 'Q1261214', 'Q1667921', 'Q3464665', 'Q17517379', 'Q21191270', 'Q196600', 'Q24862', 'Q202866'}

# --- API Wrappers ---
def authenticate_tmdb():
    url = "https://api.themoviedb.org/3/authentication"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    print("TMDB Auth Response:", response.text)

# --- Wikidata Utilities ---
def wikidata_search(query, validate_p31=False):
    """
    validate_p31 = false:
        This returns the first entry after searching using the query given
    validate_p31 = true:
        This returns the first entry that is an INSTANCE OF one of the P31 ids from the P31 ids list. NOTE P31 = 'instance of' in wikidata
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0
    }

    response = requests.get(url, params=params)
    results = response.json().get("query", {}).get("search", [])

    for result in results:
        entity_id = result.get("title")
        if not validate_p31:
            return {
                "title": entity_id,
                "url": f"https://www.wikidata.org/wiki/{entity_id.replace(' ', '_')}",
                "snippet": result.get("snippet", "")
            }

        metadata = get_wiki_entity_metadata(entity_id)
        p31_list = metadata.get("claims", {}).get("P31", [])
        for claim in p31_list:
            p31_id = (
                claim.get("mainsnak", {})
                .get("datavalue", {})
                .get("value", {})
                .get("id")
            )
            if p31_id in P31_IDS:
                return {
                    "title": entity_id,
                    "url": f"https://www.wikidata.org/wiki/{entity_id.replace(' ', '_')}",
                    "snippet": result.get("snippet", "")
                }
    return None

def get_wiki_id(title, validate_p31=False):
    """
        validate_p31 = false:
            This returns the wikidata id of the first search result using the title given
        validate_p31 = true:
            This returns the wikidata id of the first search result that is an INSTANCE OF one of the P31 ids from the P31 ids list. NOTE P31 = 'instance of' in wikidata
    """
    result = wikidata_search(title, validate_p31=validate_p31)
    return result["title"] if result else None

def get_wiki_entity_metadata(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    response = requests.get(url)
    return response.json()['entities'].get(entity_id)

def get_media_data(title, validate_p31=False):
    wiki_id = get_wiki_id(title, validate_p31)
    if not wiki_id:
        print(f"⚠️ No Wikidata ID found for: {title}")
        return {}
    url = f"https://api.themoviedb.org/3/find/{wiki_id}?external_source=wikidata_id"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    return requests.get(url, headers=headers).json()

# --- Data Handling ---
def load_media_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Title', 'Date', 'Summary', 'Genre'], inplace=True)
    return df['Title'].tolist(), df['Summary'].tolist(), df['Genre'].tolist(), df['Date'].tolist()

def load_netflix_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Title', 'Date'], inplace=True)
    return df['Title'].tolist(), df['Date'].tolist()

def normalize_titles(titles):
    """Clean and deduplicate titles while preserving individual seasons."""
    unique_series = set()
    cleaned_titles = []
    for title in titles:
        parts = title.split(":")
        if len(parts) >= 3 and "Season" in parts[1]:
            clean_title = ":".join(parts[:2]).strip()
        elif len(parts) == 2 and "Episode" in parts[1]:
            clean_title = parts[0].strip()
        else:
            clean_title = title.strip()
        unique_series.add(clean_title)
        cleaned_titles.append(clean_title)
    return list(unique_series), cleaned_titles

def get_valid_media_entries(titles, validate_p31=False, max_entries = 2000):
    """Fetch media data and collect valid entries with an overview."""
    type_counts = {"movie": 0, "tv": 0, "tv_episode": 0, "tv_season": 0}
    failed_titles = []
    valid_titles = []
    for i, title in enumerate(titles):
        if i >= max_entries:
            break
        print(f"#{i} Querying: {title}")
        data = get_media_data(title, validate_p31=validate_p31)
        if not data:
            failed_titles.append(title)
            continue

        for key in ["movie_results", "tv_results", "tv_episode_results", "tv_season_results"]:
            if data.get(key):
                type_counts[key.replace("_results", "")] += 1
                if data[key][0].get("overview"):
                    valid_titles.append(title)
                break

    return type_counts, failed_titles, valid_titles


def extract_p31_ids(valid_titles, validate_p31=False, max_entries = 2000):
    """Extract P31 IDs from Wikidata for titles with an overview."""
    p31_ids = set()
    for i, title in enumerate(valid_titles):
        if i >= max_entries:
            break
        metadata = get_wiki_entity_metadata(get_wiki_id(title, validate_p31=validate_p31))
        if not metadata:
            continue
        try:
            p31 = (
                metadata.get("claims", {})
                .get("P31", [])[0]
                .get("mainsnak", {})
                .get("datavalue", {})
                .get("value", {})
                .get("id")
            )
            if p31:
                p31_ids.add(p31)
        except Exception:
            continue
    return p31_ids

def describe_p31_ids(p31_ids, logfile="p31_results.txt"):
    """Log human-readable P31 labels."""
    for p31_id in p31_ids:
        label = get_wiki_entity_metadata(p31_id).get("labels", {}).get("en", {}).get("value", "Unknown")
        log(f"{p31_id}: {label}", logfile)

def log(message, logfile="p31_results.txt"):
    print(message)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

def get_P31_IDs(validate_p31=False, logfile="p31_results.txt", max_entries = 2000):
    titles, _ = load_netflix_data("NetflixViewingHistory (1).csv")
    # titles, summaries, genres, dates = load_media_data("processed_netflix.csv")
    unique_series, _ = normalize_titles(titles)
    type_counts, failed_titles, valid_titles = get_valid_media_entries(unique_series, validate_p31, max_entries)
    p31_ids = extract_p31_ids(valid_titles, validate_p31, max_entries)

    max_entries = min(max_entries, len(unique_series))
    log("\nMedia Type Breakdown: " + str(type_counts), logfile)
    log(f"Missing Wikidata IDs: {len(failed_titles)} / {max_entries} ({len(failed_titles)/max_entries:.2%})", logfile)
    log(f"Media with Overview: {len(valid_titles)} / {max_entries} ({len(valid_titles)/max_entries:.2%})", logfile)
    log(f"P31 IDs: {p31_ids}", logfile)

    describe_p31_ids(p31_ids, logfile)

    return p31_ids

def create_csv_with_TMDB():
    titles, dates = load_netflix_data("NetflixViewingHistory (1).csv")
    cleaned_titles = titles

    if len(cleaned_titles) != len(dates):
        raise ValueError("Mismatch between number of cleaned titles and dates.")

    overview_cache = {}
    title_resolution_map = {}
    overviews = []
    genres = []

    for original_title in cleaned_titles:
        if original_title in title_resolution_map:
            resolved_title = title_resolution_map[original_title]
            overviews.append(overview_cache[resolved_title])
            continue

        current_title = original_title
        overview = "No overview available"
        found_overview = False

        while len(current_title.split(":")) >= 1:
            print(f"Trying {current_title}")
            data = get_media_data(current_title, validate_p31=True)

            if data:
                for key in ["movie_results", "tv_results", "tv_episode_results", "tv_season_results"]:
                    if data.get(key):
                        result = data[key][0]
                        if result.get("overview"):
                            overview = result["overview"]
                            found_overview = True
                            genre_ids = result.get("genre_ids")
                            genre_converted = []
                            if genre_ids:
                                for id in genre_ids:
                                    if key[0:2] == "tv":
                                        genre_converted.append(TV_GENRES[id])
                                    else:
                                        genre_converted.append(MOVIE_GENRES[id])
                                print(f"Genres: {genre_converted}")
                            if genre_converted == []:
                                genre_converted.append("NA")
                            genres.append(genre_converted)
                            break

            if found_overview or len(current_title.split(":")) == 1:
                break  # Exit if overview found or nothing left to strip
            current_title = ":".join(current_title.split(":")[:-1]).strip()

        print(f"✅ Using title: {current_title}")
        overview_cache[current_title] = overview
        title_resolution_map[original_title] = current_title
        overviews.append(overview)

    with open("cleaned_netflix_data.csv", mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Overview", "Genres", "Date"])
        for title, overview, genre, date in zip(cleaned_titles, overviews, genres, dates):
            writer.writerow([title, overview, genre, date])

    print("✅ CSV file 'cleaned_netflix_data.csv' created.")


# --- Embedding Pipeline ---
def embed_texts(texts):
    return np.array(embed.text(texts=texts, model="nomic-embed-text-v1")['embeddings'])

def visualize_embeddings(titles, genres, embeddings, query_emb, ref_emb, ref_name="Reference"):
    all_embeddings = np.vstack([embeddings, ref_emb, query_emb])
    reduced = PCA(n_components=2).fit_transform(all_embeddings)

    corpus_2d, ref_2d, query_2d = reduced[:-2], reduced[-2], reduced[-1]
    cmap = cm.get_cmap('tab20', len(set(genres)))
    genre_colors = {genre: cmap(i) for i, genre in enumerate(sorted(set(genres)))}
    colors = [genre_colors[g] for g in genres]

    plt.figure(figsize=(20, 20))
    plt.scatter(corpus_2d[:, 0], corpus_2d[:, 1], c=colors)
    for i, title in enumerate(titles):
        plt.annotate(title, (corpus_2d[i, 0], corpus_2d[i, 1]), fontsize=8)
    plt.scatter(query_2d[0], query_2d[1], color='red', marker='x', s=100)
    plt.annotate("Query", query_2d, fontsize=10, color='red')
    plt.scatter(ref_2d[0], ref_2d[1], color='green', marker='x', s=100)
    plt.annotate(ref_name, ref_2d, fontsize=10, color='green')
    
    legend = [plt.Line2D([0], [0], marker='o', color='w', label=genre, markerfacecolor=color, markersize=8)
              for genre, color in genre_colors.items()]
    plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc='upper left', title="Genres")
    plt.title("2D Visualization of Media Embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Pipeline ---
def main():
    authenticate_tmdb()
    # get_P31_IDs(validate_p31=False, max_entries=2000)
    # get_P31_IDs(validate_p31=True, max_entries=2000)
    titles, summaries, genre, date= load_media_data("cleaned_netflix_data.csv")
    # interstellar_summary = "Interstellar is a science fiction film directed by Christopher Nolan that follows a group of astronauts who travel through a wormhole in search of a new habitable planet as Earth faces ecological collapse."
    # query_summary = "A science fiction film exploring time dilation, love, and survival with emotional depth and theoretical physics."

    # # Remove Interstellar from corpus
    # titles = titles[1:]
    # summaries = summaries[1:]

    # corpus_embeddings = embed_texts(summaries)
    # interstellar_emb = embed_texts([interstellar_summary])[0]
    # query_emb = embed_texts([query_summary])[0]

    # visualize_embeddings(titles, corpus_embeddings, query_emb, interstellar_emb)

if __name__ == "__main__":
    main()

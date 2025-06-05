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

P31_IDS = {'Q63952888', 'Q17517379', 'Q21191270', 'Q20650540', 'Q526877', 'Q3464665', 'Q11424', 'Q21198342', 'Q5398426', 'Q117467246', 'Q1259759'}

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
    df.dropna(subset=['Title', 'Summary', 'Genre'], inplace=True)
    return df['Title'].tolist(), df['Summary'].tolist(), df['Genre'].tolist(), df['Date'].tolist()

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
            clean_title = parts[0].strip()
        unique_series.add(clean_title)
        cleaned_titles.append(clean_title)
    return list(unique_series), cleaned_titles


def get_valid_media_entries(titles, validate_p31=False):
    """Fetch media data and collect valid entries with an overview."""
    type_counts = {"movie": 0, "tv": 0, "tv_episode": 0, "tv_season": 0}
    failed_titles = []
    valid_titles = []
    
    for title in titles:
        print(f"Querying: {title}")
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


def extract_p31_ids(valid_titles, validate_p31=False):
    """Extract P31 IDs from Wikidata for titles with an overview."""
    p31_ids = set()
    for title in valid_titles:
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


def describe_p31_ids(p31_ids):
    """Print human-readable P31 labels."""
    for p31_id in p31_ids:
        label = get_wiki_entity_metadata(p31_id).get("labels", {}).get("en", {}).get("value", "Unknown")
        print(f"{p31_id}: {label}")

def get_P31_IDs(validate_p31=False):
    titles, summaries, genres, dates = load_media_data("processed_netflix.csv")
    unique_series, _ = normalize_titles(titles)
    type_counts, failed_titles, valid_titles = get_valid_media_entries(unique_series, validate_p31)
    p31_ids = extract_p31_ids(valid_titles, validate_p31)

    total = len(unique_series)
    print("\nMedia Type Breakdown:", type_counts)
    print(f"Missing Wikidata IDs: {len(failed_titles)} / {total} ({len(failed_titles)/total:.2%})")
    print(f"Media with Overview: {len(valid_titles)} / {total} ({len(valid_titles)/total:.2%})")
    print(f"P31 IDs: {p31_ids}")
    describe_p31_ids(p31_ids)

    return p31_ids

# --- Main Pipeline ---
def main():
    authenticate_tmdb()
    get_P31_IDs()
    get_P31_IDs(validate_p31=True)
    # titles, summaries, genres = load_movie_data("processed_netflix.csv")

    # interstellar_summary = "Interstellar is a science fiction film directed by Christopher Nolan that follows a group of astronauts who travel through a wormhole in search of a new habitable planet as Earth faces ecological collapse."
    # query_summary = "A science fiction film exploring time dilation, love, and survival with emotional depth and theoretical physics."

    # # Remove Interstellar from corpus
    # titles = titles[1:]
    # summaries = summaries[1:]
    # genres = genres[1:]

    # corpus_embeddings = embed_texts(summaries)
    # interstellar_emb = embed_texts([interstellar_summary])[0]
    # query_emb = embed_texts([query_summary])[0]

    # visualize_embeddings(titles, genres, corpus_embeddings, query_emb, interstellar_emb)

if __name__ == "__main__":
    main()

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
import ast
from scipy.spatial import ConvexHull

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

def visualize_embeddings(titles, genres, embeddings, highlight_groups, ref_emb=None, ref_name="Reference"):
    """
    highlight_groups: List of (label, keyword, color, query_emb) tuples.
    Example:
        [("Dr. Stone", "DR.STONE", "red", red_emb),
         ("Arcane", "Arcane", "blue", blue_emb)]
    """

    all_group_embs = [query_emb for _, _, _, query_emb in highlight_groups]
    all_embeddings = np.vstack([embeddings] + all_group_embs + ([ref_emb] if ref_emb is not None else []))
    reduced = PCA(n_components=2).fit_transform(all_embeddings)

    n = len(titles)
    corpus_2d = reduced[:n]
    group_embs_2d = reduced[n:n + len(highlight_groups)]
    ref_2d = reduced[-1] if ref_emb is not None else None

    # Plotting base
    cmap = cm.get_cmap('tab20', len(set(genres)))
    genre_colors = {genre: cmap(i) for i, genre in enumerate(sorted(set(genres)))}
    colors = [genre_colors[g] for g in genres]

    plt.figure(figsize=(20, 20))
    plt.scatter(corpus_2d[:, 0], corpus_2d[:, 1], c=colors)
    # for i, title in enumerate(titles):
        # plt.annotate(title, (corpus_2d[i, 0], corpus_2d[i, 1]), fontsize=8)


    # Reference point
    if ref_emb is not None:
        plt.scatter(ref_2d[0], ref_2d[1], color='black', marker='x', s=100)
        plt.annotate(ref_name, ref_2d, fontsize=10, color='black')

    # Legend
    legend = [plt.Line2D([0], [0], marker='o', color='w', label=genre,
                         markerfacecolor=color, markersize=8)
              for genre, color in genre_colors.items()]
    plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc='upper left', title="Genres")
    plt.title("2D Visualization of Media Embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(-0.1, 0.25)
    plt.ylim(-0.125, 0.175)

    # --- Plot group boundaries and queries ---
    for i, (label, keyword, color, query_emb) in enumerate(highlight_groups):
        indices = [j for j, title in enumerate(titles) if keyword.upper() in title.upper()]
        if indices:
            group_points = corpus_2d[indices]
            if len(group_points) >= 3:
                hull = ConvexHull(group_points)
                for simplex in hull.simplices:
                    plt.plot(group_points[simplex, 0], group_points[simplex, 1],
                             linestyle='--', color=color, linewidth=2)
            else:
                plt.scatter(group_points[:, 0], group_points[:, 1],
                            edgecolors=color, facecolors='none', s=100, linewidths=2)
            center = group_points.mean(axis=0)
            plt.annotate(label, center, fontsize=12, color=color)

        # Plot the group-specific query
        query_2d = group_embs_2d[i]
        plt.scatter(query_2d[0], query_2d[1], color=color, marker='x', s=100)
        plt.annotate(f"{keyword} Query", query_2d, fontsize=10, color=color)

    plt.show()

# --- Main Pipeline ---
def main():
    authenticate_tmdb()

    titles, summaries, genres, date= load_media_data("cleaned_netflix_data.csv")
    print(len(set(summaries))/len(summaries))
    cleaned_genres = [ast.literal_eval(g) if isinstance(g, str) else ["Unknown"] for g in genres]
    simplified_genres = [g[0] if isinstance(g, list) and g else "Unknown" for g in cleaned_genres]

    red = "a science-fiction anime centered around Senku Ishigami, a genius high school student " \
    "who wakes up thousands of years after a mysterious phenomenon turns all of humanity into stone."

    blue = "an animated sci-fi fantasy series set in the universe of the video game League of Legends. " \
    "It explores the origins of iconic champions and the conflict between two cities: " \
    "the wealthy, tech-advanced Piltover and the oppressed, chaotic Zaun."

    green = "Hundreds of cash-strapped players accept a strange invitation to compete in children's games " \
    "for a tempting prize — but the stakes are deadly."

    orange = "a teen sitcom that aired on Nickelodeon from 2010 to 2013. It follows Tori Vega, a talented " \
    "teenager who unexpectedly finds herself attending Hollywood Arts High School, " \
    "a performing arts school filled with eccentric and creative students."

    red_em = embed_texts([red])[0]
    blue_em = embed_texts([blue])[0]
    green_em = embed_texts([green])[0]
    orange_em = embed_texts([orange])[0]

    highlight_groups = [
        ("Dr. Stone Group", "DR.STONE", "red", red_em),
        ("Arcane Group", "Arcane", "blue", blue_em),
        ("Squid Game Group", "Squid Game", "green", green_em),
        ("Victorious Group", "Victorious", "orange", orange_em)
    ]

    # combined_texts = []
    # for summary, genre_list in zip(summaries, cleaned_genres):
    #     genre_str = ", ".join(genre_list)
    #     combined_text = f"Genres: {genre_str}. Summary: {summary}"
    #     combined_texts.append(combined_text)

    # corpus_embeddings = embed_texts(combined_texts)

    corpus_embeddings = embed_texts(summaries)

    visualize_embeddings(
        titles,
        simplified_genres,
        corpus_embeddings,
        highlight_groups=highlight_groups,
        ref_emb=None
)

if __name__ == "__main__":
    main()

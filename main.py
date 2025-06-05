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

P31_IDS: {'Q63952888', 'Q17517379', 'Q21191270', 'Q20650540', 'Q526877', 'Q3464665', 'Q11424', 'Q21198342', 'Q12737077', 'Q5398426', 'Q117467246', 'Q431289', 'Q1259759'}

# --- API Wrappers ---
def authenticate_tmdb():
    url = "https://api.themoviedb.org/3/authentication"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    print("TMDB Auth Response:", response.text)

def wikidata_search(query):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get("query", {}).get("search")
    if results:
        result = results[0]
        return {
            "title": result["title"],
            "snippet": result.get("snippet", ""),
            "url": f"https://www.wikidata.org/wiki/{result['title'].replace(' ', '_')}"
        }
    return None

def get_wiki_entity_metadata(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    response = requests.get(url)
    return response.json()['entities'].get(entity_id)

def get_wiki_id(title):
    result = wikidata_search(title)
    return result["title"] if result else None

def get_media_data(title):
    wiki_id = get_wiki_id(title)
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

# --- Main Pipeline ---
def main():
    authenticate_tmdb()
    titles, summaries, genres, dates = load_media_data("processed_netflix.csv")

    unique_series = set()
    for i, title in enumerate(titles):
        parts = title.split(":")
        if len(parts) >= 3 and "Season" in parts[1]:
            clean_title = ":".join(parts[:2]).strip()
        elif len(parts) == 2 and "Episode" in parts[1]:
            clean_title = parts[0].strip()
        else:
            clean_title = parts[0].strip()
        unique_series.add(clean_title)
        titles[i] = clean_title

    type_counts = {"movie": 0, "tv": 0, "tv_episode": 0, "tv_season": 0}
    failed_titles = []
    overviews_found = 0
    unique_entry_item_ids = set()
    for title in unique_series:
        print(f"Querying: {title}")
        data = get_media_data(title)
        if not data:
            failed_titles.append(title)
            continue

        for key in ["movie_results", "tv_results", "tv_episode_results", "tv_season_results"]:
            if key in data and data[key]:
                type = key.replace("_results", "")
                type_counts[type] += 1
                if data[key][0].get("overview"):
                    overviews_found += 1
                    metadata = get_wiki_entity_metadata(get_wiki_id(title))
                    if metadata:
                        claims = metadata.get("claims")
                        if not claims:
                            print("⚠️ No claims found in metadata")
                            continue

                        p31_list = claims.get("P31")
                        if not p31_list or not isinstance(p31_list, list):
                            print("⚠️ No P31 found for this entity")
                            continue

                        mainsnak = p31_list[0].get("mainsnak")
                        if not mainsnak:
                            print("⚠️ No mainsnak found in first P31 entry")
                            continue

                        datavalue = mainsnak.get("datavalue")
                        if not datavalue:
                            print("⚠️ No datavalue in mainsnak")
                            continue

                        value = datavalue.get("value")
                        if not value:
                            print("⚠️ No value in datavalue")
                            continue

                        p31_id = value.get("id")
                        print("P31 ID:", p31_id)

                        unique_entry_item_ids.add(p31_id)
                    else:
                        print("no meta data found")
                    break

    total = len(unique_series)
    print("\nMedia Type Breakdown:", type_counts)
    print(f"Missing Wikidata IDs: {len(failed_titles)} / {total} ({len(failed_titles)/total:.2%})")
    print(f"Media with Overview: {overviews_found} / {total} ({overviews_found/total:.2%})")
    print(f"P31 IDs: {unique_entry_item_ids}")

    for id in unique_entry_item_ids:
        print(f"{id}: {get_wiki_entity_metadata(id).get("labels").get("en").get("value")}")
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

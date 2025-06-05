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
TMDB_API_TOKEN = os.getenv('TMDB_API_READ_ACCESS_TOKEN')
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

def authenticate_tmdb():
    """Authenticate with TMDB API."""
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

    if data.get("query", {}).get("search"):
        result = data["query"]["search"][0]
        return {
            "title": result["title"],
            "snippet": result.get("snippet", ""),
            "url": f"https://www.wikidata.org/wiki/{result['title'].replace(' ', '_')}"
        }
    else:
        return None
    
def get_movie_data(title):
    movie_id = wikidata_search(title)
    if movie_id:
        movie_id = movie_id["title"]
    else:
        return None
    source_id = 'wikidata_id'
    url = f"https://api.themoviedb.org/3/find/{movie_id}?external_source={source_id}"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    return requests.get(url, headers=headers).json()

def get_movie_genres(ids, media_type):
    genres = []
    for id in ids:
        if media_type == "movie":
            genres.append(MOVIE_GENRES[id])
        else:
            genres.append(TV_GENRES[id])

def load_movie_data(filepath):
    """Load and preprocess movie data."""
    df = pd.read_csv(filepath)
    df.dropna(subset=['Title', 'Summary', 'Genre'], inplace=True)
    return df['Title'].tolist(), df['Summary'].tolist(), df['Genre'].tolist()

def embed_texts(texts):
    """Embed a list of texts using Nomic's embedding model."""
    return np.array(embed.text(texts=texts, model="nomic-embed-text-v1")['embeddings'])

def visualize_embeddings(titles, genres, embeddings, query_emb, interstellar_emb):
    """Visualize movie embeddings, coloring by genre."""
    all_embeddings = np.vstack([embeddings, interstellar_emb, query_emb])
    reduced = PCA(n_components=2).fit_transform(all_embeddings)

    corpus_2d = reduced[:-2]
    interstellar_2d = reduced[-2]
    query_2d = reduced[-1]

    # Assign colors to genres
    unique_genres = sorted(set(genres))
    cmap = cm.get_cmap('tab20', len(unique_genres))
    genre_colors = {genre: cmap(i) for i, genre in enumerate(unique_genres)}
    colors = [genre_colors[g] for g in genres]

    # Plot
    plt.figure(figsize=(20, 20))
    plt.scatter(corpus_2d[:, 0], corpus_2d[:, 1], c=colors)
    for i, title in enumerate(titles):
        plt.annotate(title, (corpus_2d[i, 0], corpus_2d[i, 1]), fontsize=8)

    plt.scatter(query_2d[0], query_2d[1], color='red', label='Query', marker='x', s=100)
    plt.annotate("Query", query_2d, fontsize=10, color='red')

    plt.scatter(interstellar_2d[0], interstellar_2d[1], color='green', label='Interstellar', marker='x', s=100)
    plt.annotate("Interstellar", interstellar_2d, fontsize=10, color='green')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=genre,
                   markerfacecolor=color, markersize=8)
        for genre, color in genre_colors.items()
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Genres")

    plt.title("2D Visualization of Movie Embeddings by Genre")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(-0.15, 0)
    plt.ylim(-0.15, 0)
    plt.show()

def main():
    titles, summaries, genres = load_movie_data("processed_netflix.csv")
    media_type = set()
    authenticate_tmdb()
    for movie_title in titles:
        print(movie_title)      
        print(get_movie_data(movie_title))
        # if movie_data:
        #     print(movie_data)
        #     movie_data = movie_data["movie_results"][0]["media_type"]
        # else:
        #     continue
        # media_type.add(movie_data)
    print(media_type)
    # overview = movie_data["movie_results"][0]["overview"]
    # genres = get_movie_genres(movie_data["movie_results"][0]["genre_ids"], movie_data["movie_results"][0]["media_type"])

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

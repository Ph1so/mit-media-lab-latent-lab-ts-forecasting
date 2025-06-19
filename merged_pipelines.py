import json
import os
import datetime
import pandas as pd
import csv

from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import openai
import asyncio
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
from asyncio import Semaphore

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TMDB_API_TOKEN = os.getenv("TMDB-API-READ-ACCESS-TOKEN")

if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")
if not TMDB_API_TOKEN:
    raise RuntimeError("TMDB-API-READ-ACCESS-TOKEN not set in .env")


# ----------------------- Netflix Utilities -----------------------
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

def load_netflix_data(filepath, limit=None):
    df = pd.read_csv(filepath).dropna(subset=['Title', 'Date'])
    if limit:
        df = df.head(limit)
    return df['Title'].tolist(), df['Date'].tolist()

def use_tmdb(title):
    url = f"https://api.themoviedb.org/3/search/multi?query={requests.utils.quote(title)}&include_adult=false&language=en-US&page=1"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_TOKEN}"}
    return requests.get(url, headers=headers).json()

def resolve_tmdb_data(original_title):
    current_title = original_title
    overview, poster_path = "No overview available", "NA"

    while current_title:
        data = use_tmdb(current_title)
        results = data.get("results", [])
        if results:
            result = results[0]
            overview = result.get("overview", overview)
            poster_path = result.get("poster_path", "NA")
            break
        if ":" not in current_title:
            break
        current_title = ":".join(current_title.split(":")[:-1]).strip()

    return original_title, current_title, overview, poster_path

async def process_netflix_viewing_history(input_path="NetflixViewingHistory.csv", output_path="netflix_tmp.csv", test_mode=False):
    titles, dates = load_netflix_data(input_path, limit=100 if test_mode else None)
    overviews, poster_paths = [None]*len(titles), [None]*len(titles)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(resolve_tmdb_data, title): i for i, title in enumerate(titles)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                _, _, overview, poster_path = future.result()
                overviews[i] = overview
                poster_paths[i] = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path != "NA" else "NA"
            except Exception:
                overviews[i], poster_paths[i] = "No overview available", "NA"

    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Date", "Description", "Header_Img_Url", "Link", "Platform"])
        for title, date, overview, poster in zip(titles, dates, overviews, poster_paths):
            writer.writerow([title, date, overview, poster, "NA", "Netflix"])

# ----------------------- TikTok Utilities -----------------------
async def summarize_description(raw_desc: str) -> str:
    if not raw_desc.strip():
        return ""
    prompt = f'Summarize: "{raw_desc}"'
    try:
        resp = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Write concise summaries of video descriptions without intro phrases."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except:
        return raw_desc

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def fetch_one(entry, browser, sem):
    share_url, raw_date = entry['link'], entry['date']
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
        date_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        date_time = raw_date

    async with sem:
        try:
            page = await browser.new_page()
            await page.goto(share_url, wait_until="networkidle", timeout=30000)
            video_url = page.url

            r = requests.get(f"https://www.tiktok.com/oembed?url={video_url}", timeout=10)
            o = r.json() if r.ok else {}

            meta_desc = await page.query_selector("meta[property='og:description']")
            og_description = (await meta_desc.get_attribute("content")).strip() if meta_desc else ""
            meta_img = await page.query_selector("meta[property='og:image']")
            og_image = (await meta_img.get_attribute("content")).strip() if meta_img else ""

            raw_desc = og_description or o.get("title", "")
            summary = await summarize_description(raw_desc)
            header_img = o.get("thumbnail_url", "") or og_image

            await page.close()

            return {
                "Title": o.get("title", ""),
                "Date": date_time,
                "Description": summary,
                "Header_Img_Url": header_img,
                "Link": video_url,
                "Platform": "TikTok"
            }

        except Exception:
            return None

async def process_all_tiktok(json_path, output_csv, concurrency=5):
    entries = load_json(json_path)["Your Activity"]["Like List"]["ItemFavoriteList"]
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        sem = Semaphore(concurrency)
        tasks = [fetch_one(entry, browser, sem) for entry in entries]
        results = await asyncio.gather(*tasks)
        await browser.close()

    df = pd.DataFrame([r for r in results if r])
    df.to_csv(output_csv, index=False)

async def process_tiktok_viewing_history(json_input="user_data_tiktok.json", output_csv="tiktok_tmp.csv", concurrency=5):
    await process_all_tiktok(json_input, output_csv, concurrency)

# ----------------------- YouTube Utilities -----------------------
def parse_youtube_history(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    videos = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'watch?v=' not in href:
            continue
        link = href if href.startswith('http') else f'https://www.youtube.com{href}'
        title = a.get_text(strip=True)
        video_id = re.search(r'v=([^&]+)', link).group(1) if re.search(r'v=([^&]+)', link) else None
        if not video_id:
            continue

        container = a.find_parent(['li', 'div', 'section', 'article'])
        date_str = next((text for text in container.stripped_strings if re.search(r'\b\d{4}\b', text)), "")
        try:
            date_time = date_parser.parse(date_str, fuzzy=True).strftime('%Y-%m-%d %H:%M:%S')
        except:
            date_time = ''
        videos.append({'title': title, 'link': link, 'date_time': date_time, 'video_id': video_id})
    return videos

def summarize_youtube(title, desc):
    try:
        prompt = f"Video Title: {title}\n\nDescription:\n{desc}\n\nPlease summarize in 2-3 sentences."
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You summarize YouTube videos concisely without starting with 'This video...'"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except:
        return desc

def process_youtube_video(vid):
    title, link, date_time, video_id = vid.values()
    try:
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
        r = requests.get(url)
        snippet = r.json().get("items", [{}])[0].get("snippet", {})
        desc = snippet.get("description", title)
        thumb_url = (
            snippet.get("thumbnails", {}).get("high", {}).get("url")
            or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        )
    except:
        desc = title
        thumb_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    return {
        'Title': title,
        'Date': date_time,
        'Description': summarize_youtube(title, desc),
        'Header_Img_Url': thumb_url,
        'Link': link,
        'Platform': 'YouTube'
    }

def process_youtube_history(html_input="watch-history.html", output_csv="youtube_tmp.csv"):
    videos = parse_youtube_history(html_input)
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_youtube_video, v) for v in videos]
        entries = [f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="YouTube")]
    pd.DataFrame(entries).to_csv(output_csv, index=False)

# ----------------------- Final Merge -----------------------
def merge_csvs(csv_paths, output_path):
    pd.concat([pd.read_csv(p) for p in csv_paths]).to_csv(output_path, index=False)
    print(f"✅ Merged into '{output_path}'")

# ----------------------- API Endpoint -----------------------
class ProcessRequest(BaseModel):
    netflix_input_path: str = "NetflixViewingHistory.csv"
    tiktok_input_path: str = "user_data_tiktok.json"
    youtube_input_path: str = "watch-history.html"
    output_path: str = "processed_user_data.csv"
    test_mode: bool = False
    concurrency: int = 5

@app.post("/process_user_data")
async def process_user_data(req: ProcessRequest):
    tmp_paths = {
        "netflix": "netflix_tmp.csv",
        "tiktok": "tiktok_tmp.csv",
        "youtube": "youtube_tmp.csv"
    }

    await process_netflix_viewing_history(req.netflix_input_path, tmp_paths["netflix"], req.test_mode)
    await process_tiktok_viewing_history(req.tiktok_input_path, tmp_paths["tiktok"], req.concurrency)
    process_youtube_history(req.youtube_input_path, tmp_paths["youtube"])
    merge_csvs(list(tmp_paths.values()), req.output_path)
    return {"message": "✅ All data merged", "output": req.output_path}
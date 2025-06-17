import os
import json
import pandas as pd
import openai
import asyncio
import requests

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from asyncio import Semaphore

# ─── 1) Load OpenAI key from .env ──────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# ─── 2) Helper: Summarize with GPT-4o ─────────────────────────────────────────
async def summarize_description(raw_desc: str) -> str:
    if not raw_desc or not raw_desc.strip():
        return ""
    example_input = (
        "Doraemon cosplay from the new movie #cosplay #doraemon #anime @cosplayer123"
    )
    example_output = (
        "A cosplayer dresses as Doraemon from the latest film, using hashtags to "
        "connect with anime and cosplay fans."
    )
    system_prompt = (
        "You are a helpful assistant that writes concise summaries of video descriptions. "
        "Generate a 2-3 sentence summary focusing on the content described. "
        "Do not begin your summary with phrases like 'This video', 'The content', "
        "or 'The description'. Start directly with the descriptive information, "
        "and do not use any promotional or advertising language."
    )
    user_prompt = (
        f"Example:\n"
        f"Description: \"{example_input}\"\n"
        f"Summary: \"{example_output}\"\n\n"
        f"Now summarize this description in 2-3 sentences focusing on the content:\n"
        f"\"{raw_desc}\""
    )
    try:
        resp = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing description: {e}")
        return raw_desc

# ─── 3) Load JSON dataset ─────────────────────────────────────────────────────
def load_json(file_path: str) -> dict:
    print(f"Loading JSON dataset from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ─── 4) Async worker: resolve + fetch metadata (oEmbed + JSON fallback) ──────
async def fetch_one(entry: dict, browser, sem: Semaphore) -> dict | None:
    share_url = entry['link']
    raw_date  = entry['date']
    print(f"→ Starting work on: {share_url}")

    # Format date_time
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
        date_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        date_time = raw_date

    async with sem:
        try:
            page = await browser.new_page()
            # 1) Navigate & wait for JS redirect
            await page.goto(share_url, wait_until="networkidle", timeout=30000)
            video_url = page.url
            print(f"   ↳ Resolved to URL: {video_url}")

            # ─── A) Try oEmbed for title & thumbnail ─────────────────────────────
            oembed_api = f"https://www.tiktok.com/oembed?url={video_url}"
            oembed_title = ""
            oembed_thumbnail = ""
            try:
                r = requests.get(oembed_api, timeout=10)
                r.raise_for_status()
                o = r.json()
                oembed_title     = o.get("title", "").strip()
                oembed_thumbnail = o.get("thumbnail_url", "").strip()
            except:
                oembed_title = ""
                oembed_thumbnail = ""

            # ─── B) Scrape og:description and og:image ───────────────────────────
            og_description = ""
            og_image = ""

            meta_desc = await page.query_selector("meta[property='og:description']")
            if meta_desc:
                og_description = (await meta_desc.get_attribute("content") or "").strip()

            meta_img = await page.query_selector("meta[property='og:image']")
            if meta_img:
                og_image = (await meta_img.get_attribute("content") or "").strip()

            # ─── C) Decide which “raw description” to summarize ──────────────────
            raw_desc = og_description if og_description else oembed_title

            # ─── D) JSON fallback for description & image ────────────────────────
            if not raw_desc or (not oembed_thumbnail and not og_image):
                script = await page.query_selector("script#__NEXT_DATA__")
                if script:
                    try:
                        text = await script.get_attribute("innerHTML")
                        data = json.loads(text or "{}")

                        # 1) Video fallback: itemStruct.desc and itemStruct.video.cover
                        item = (
                            data.get("props", {})
                                .get("pageProps", {})
                                .get("itemInfo", {})
                                .get("itemStruct", {})
                        )
                        if item:
                            if not raw_desc and item.get("desc"):
                                raw_desc = item["desc"].strip()
                            if not oembed_thumbnail and not og_image:
                                cover = item.get("video", {}).get("cover")
                                if cover:
                                    og_image = cover.strip()

                        # 2) Photo fallback: photoStruct.caption and photoStruct.covers
                        photo = (
                            data.get("props", {})
                                .get("pageProps", {})
                                .get("photoInfo", {})
                                .get("photoStruct", {})
                        )
                        if photo:
                            if not raw_desc and photo.get("caption"):
                                raw_desc = photo["caption"].strip()
                            if not oembed_thumbnail and not og_image:
                                covers = photo.get("covers", [])
                                if covers:
                                    og_image = covers[0].strip()

                    except Exception:
                        pass

            await page.close()

            # ─── E) Summarize the best‐available raw_desc ───────────────────────
            summary = await summarize_description(raw_desc)

            # ─── F) Choose header_image_url in priority order ────────────────────
            header_image_url = oembed_thumbnail or og_image or ""

            # ─── G) Build the final record ───────────────────────────────────────
            return {
                "title":            oembed_title,
                "description":      summary,
                "link":             video_url,
                "date_time":        date_time,
                "header_image_url": header_image_url,
                "platform":         "TikTok"
            }

        except Exception as e:
            print(f"   ❗ Error on {share_url}: {e}")
            # If any error occurs, return None (so it will be skipped)
            return None

# ─── 5) Orchestrator: launch browser & gather all workers ─────────────────────
async def process_all(json_path: str, output_csv: str, concurrency: int = 5):
    data = load_json(json_path)
    entries = data["Your Activity"]["Like List"]["ItemFavoriteList"]

    # Launch one shared headless browser
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        sem = Semaphore(concurrency)

        # Kick off one task per entry
        tasks = [fetch_one(entry, browser, sem) for entry in entries]

        # Await all tasks
        results = await asyncio.gather(*tasks)
        await browser.close()

    # Filter out None results
    filtered = [r for r in results if r is not None]

    # Build DataFrame & save CSV
    df = pd.DataFrame(filtered, columns=[
        "title",
        "description",
        "link",
        "date_time",
        "header_image_url",
        "platform"
    ])
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

# ─── 6) Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    json_input  = sys.argv[1] if len(sys.argv) > 1 else "user_data_tiktok.json"
    csv_output  = sys.argv[2] if len(sys.argv) > 2 else "processed_tiktok.csv"
    concurrency = 5  # adjust based on your machine’s resources

    asyncio.run(process_all(json_input, csv_output, concurrency))

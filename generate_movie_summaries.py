import os
import json
import pandas as pd
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
from pathlib import Path

class NetflixSummarizer:
    def __init__(self, input_csv: str = "NetflixViewingHistory.csv", output_csv: str = "processed_netflix.csv", model: str = "gpt-4", workers: int = 32):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.model = model
        self.workers = workers

        # Load environment variables
        env_path = Path(__file__).parent / ".env"
        load_dotenv(dotenv_path=env_path, override=True)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        openai.api_key = self.api_key

    def summarize_and_genre(self, title: str) -> Tuple[str, str]:
        try:
            resp = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that summarizes Netflix content. "
                            "Return JSON with exactly two keys: 'summary' (one paragraph max) "
                            "and 'genre' (a single-word label)."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Title: \"{title}\" Respond with JSON only."
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            text = resp.choices[0].message.content.strip()
            try:
                data = json.loads(text)
            except Exception:
                import re
                m = re.search(r"\{.*\}", text, re.DOTALL)
                data = json.loads(m.group(0)) if m else {}
            return data.get("summary", "").replace("\n", " "), data.get("genre", "")
        except Exception as e:
            print(f"❌ Error on {title!r}: {e}")
            return "", ""

    def process(self):
        df = pd.read_csv(self.input_csv, parse_dates=["Date"])
        titles = df["Title"].tolist()
        results = [("", "")] * len(titles)

        print(f"Starting {len(titles)} requests with {self.workers} threads...")

        with ThreadPoolExecutor(max_workers=self.workers) as exe:
            futures = {exe.submit(self.summarize_and_genre, t): i for i, t in enumerate(titles)}
            for idx, fut in enumerate(as_completed(futures), 1):
                i = futures[fut]
                summary, genre = fut.result()
                results[i] = (summary, genre)
                print(f"✅ [{idx}/{len(titles)}]")

        df["Summary"], df["Genre"] = zip(*results)
        df.to_csv(self.output_csv, index=False)
        print(f"Done—wrote {self.output_csv}")

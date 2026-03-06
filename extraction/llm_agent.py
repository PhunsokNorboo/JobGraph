"""LLM extraction agent -- transforms raw HTML into structured JobRecords."""
import json
import logging
import re
from html import unescape
from pathlib import Path

from pydantic import ValidationError

from extraction.schema import JobRecord
from extraction.llm_client import extract_structured
from extraction.prompts import EXTRACTION_PROMPT
from extraction.postprocess import normalize_skills

logger = logging.getLogger(__name__)
MAX_RETRIES = 3


def clean_html(html: str) -> str:
    """Strip HTML tags and extract readable text."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_job(raw_html: str, company: str, url: str) -> JobRecord | None:
    """Extract structured job data from raw HTML using LLM.

    Returns JobRecord on success, None after all retries exhausted.
    """
    text = clean_html(raw_html)[:6000]  # trim to fit LLM context
    prompt = EXTRACTION_PROMPT.format(raw_text=text)

    for attempt in range(MAX_RETRIES):
        try:
            raw = extract_structured(prompt)
            raw["company"] = company
            raw["raw_url"] = url
            # Normalize skills before validation
            if "required_skills" in raw:
                raw["required_skills"] = normalize_skills(
                    raw.get("required_skills", [])
                )
            if "nice_to_have_skills" in raw:
                raw["nice_to_have_skills"] = normalize_skills(
                    raw.get("nice_to_have_skills", [])
                )
            return JobRecord(**raw)
        except (ValidationError, KeyError, ValueError, Exception) as e:
            logger.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}"
            )

    logger.error(f"All {MAX_RETRIES} retries failed for {url}")
    return None


# ─── Batch entry point ─────────────────────────────────


def run_extraction(raw_dir: Path, output_dir: Path) -> None:
    """Process all raw job JSONs and produce jobs.parquet + skill_vocabulary.csv."""
    import pandas as pd
    from tqdm import tqdm

    from extraction.postprocess import build_skill_vocabulary

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    all_skills: list[list[str]] = []

    json_files = list(raw_dir.rglob("*.json"))
    for fpath in tqdm(json_files, desc="Extracting jobs"):
        try:
            with open(fpath) as f:
                raw = json.load(f)
            job = extract_job(raw["html"], raw["company"], raw["url"])
            if job:
                job.scraped_at = raw.get("scraped_at")
                records.append(job.model_dump())
                all_skills.append(job.required_skills + job.nice_to_have_skills)
        except Exception as e:
            logger.error(f"Failed to process {fpath}: {e}")

    # Save jobs parquet
    df = pd.DataFrame(records)
    # Use Int64 for nullable int columns
    for col in ["salary_min", "salary_max"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    df.to_parquet(output_dir / "jobs.parquet", index=False)
    logger.info(f"Saved {len(df)} jobs to {output_dir / 'jobs.parquet'}")

    # Save skill vocabulary
    vocab = build_skill_vocabulary(all_skills)
    vocab_df = pd.DataFrame(
        [
            {
                "skill_id": i,
                "name": name,
                "category": "other",
                "frequency": freq,
            }
            for i, (name, freq) in enumerate(
                sorted(vocab.items(), key=lambda x: -x[1])
            )
        ]
    )
    vocab_df.to_csv(output_dir / "skill_vocabulary.csv", index=False)
    logger.info(f"Saved {len(vocab_df)} skills to vocabulary")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    base = Path(__file__).resolve().parent.parent
    run_extraction(
        raw_dir=Path(sys.argv[1]) if len(sys.argv) > 1 else base / "data" / "raw",
        output_dir=base / "data" / "processed",
    )

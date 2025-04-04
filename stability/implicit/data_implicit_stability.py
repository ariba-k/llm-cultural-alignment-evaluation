import os
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import openai
from typing import Tuple, List, Dict

from constants import DIMENSION_MAP
from environment import TokenManager


def load_dataset_to_df(dataset_name: str, num_rows: int = 5) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    selected_indices = list(range(0, min(num_rows, len(dataset["train"]))))
    small_dataset = dataset["train"].select(selected_indices)
    return pd.DataFrame(small_dataset)


def generate_response(messages: List[Dict[str, str]], model_name: str = "gpt-4o", temperature: float = 0) -> str:
    client = openai.OpenAI(api_key=TokenManager.OA_TOKEN)
    print("\n--- Prompt ---")
    print(messages)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )

    print("\n--- Response ---")
    print(response.choices[0].message.content.strip())

    return response.choices[0].message.content.strip()


def rewrite_cover_letter(cover_letter: str, tone_a: str, tone_b: str) -> str:
    prompt = (
        f"Please rewrite the following cover letter in two ways:\n"
        f"1) With a {tone_a} tone.\n"
        f"2) With a {tone_b} tone.\n\n"
        f"Important: Keep the same length and details as the original, but adjust the style and tone.\n\n"
        f"Original cover letter:\n{cover_letter}"
    )
    messages = [
        {"role": "system", "content": "You are a professional cover letter writer."},
        {"role": "user", "content": prompt}
    ]
    return generate_response(messages)


def parse_cover_letter(response_text: str) -> Tuple[str, str]:
    parts = re.split(r'(?m)^\*\*\s*\d+\)\s*.*Tone:\*\*', response_text.strip())

    if len(parts) < 3:
        version_a = response_text.strip()
        version_b = ""
    else:
        version_a = parts[1].strip()
        version_b = parts[2].strip()

    version_a = _remove_tone_headings(version_a)
    version_b = _remove_tone_headings(version_b)

    return version_a, version_b


def _remove_tone_headings(text: str) -> str:
    cleaned = re.sub(r'(?m)^\*\*.*Tone:\*\*', '', text)
    return cleaned.strip()


def process_dimension(df: pd.DataFrame, dimension: Tuple[str, str]) -> pd.DataFrame:
    tone_a, tone_b = dimension
    column_name_a = f"{tone_a}_Cover_Letter"
    column_name_b = f"{tone_b}_Cover_Letter"

    if column_name_a not in df.columns:
        df[column_name_a] = None
    if column_name_b not in df.columns:
        df[column_name_b] = None

    for i in tqdm(range(len(df))):
        original_cover_letter = df.loc[i, "Cover Letter"]

        print(f"\n--- Processing Cover Letter {i + 1}/{len(df)} ---")
        print(f"Original Cover Letter:\n{original_cover_letter}")

        response_text = rewrite_cover_letter(original_cover_letter, tone_a, tone_b)
        version_a, version_b = parse_cover_letter(response_text)

        df.loc[i, column_name_a] = version_a
        df.loc[i, column_name_b] = version_b

    return df


def main() -> None:
    num_rows = 5
    df = load_dataset_to_df("ShashiVish/cover-letter-dataset", num_rows)

    os.makedirs("data/cover_letters", exist_ok=True)

    for dimension_key, dimension in DIMENSION_MAP.items():
        print(f"\n--- Working on dimension: {dimension_key} ---")

        df_copy = df.copy()
        df_processed = process_dimension(df_copy, dimension)

        output_path = os.path.join("data", "cover_letters", f"{dimension_key}.xlsx")
        df_processed.to_excel(output_path, index=False)
        print(f"Results saved to '{output_path}'.")


if __name__ == "__main__":
    main()

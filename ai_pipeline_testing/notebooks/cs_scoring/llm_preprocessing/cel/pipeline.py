import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# Load API Key (from .env or env var)
# ---------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Put it in a .env file or your environment variables."
    )

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are a careful text span tagger.
Given a customer comment and a fixed list of attributes, you must extract
the exact text spans that refer to each attribute.

Attributes:
- Product
- Service
- Delivery
- Payment

Rules:
1. Only use EXACT substrings from the original comment.
   - Do not paraphrase or reword.
   - The span must be copied verbatim from the comment.
2. A single span can map to multiple attributes.
   - If this happens, include that exact span in the list for each relevant attribute.
3. If an attribute is not mentioned at all, DO NOT include it in the output.
4. Preserve original casing, punctuation, and spelling.
5. Return valid JSON only, with this schema:

{
  "comment": "<original comment>",
  "attributes": {
    "Product": ["<exact substring 1>", "<exact substring 2>", ...],
    "Service": ["<exact substring>", ...],
    "Delivery": ["<exact substring>", ...],
    "Payment": ["<exact substring>", ...]
  }
}

Only include keys for attributes that actually appear in the comment.
"""

def build_user_prompt(comment: str) -> str:
    return f"""
Comment:
\"\"\"{comment}\"\"\"

Extract exact text spans from this comment and group them by attribute:
Product, Service, Delivery, Payment.

Return ONLY the JSON object with:
- "comment": the original comment
- "attributes": a JSON object mapping each used attribute
  to a list of exact substrings from the comment.

Do NOT include attributes that are not mentioned.
"""


# ---------------------------------------------------------
# LLM helper for a single comment
# ---------------------------------------------------------
def extract_attribute_spans(
    comment: str,
    model: str = "gpt-4.1-mini"
) -> Dict[str, Any]:
    """
    Call the LLM to extract attribute spans for a single comment.

    Returns:
    {
      "comment": "<original>",
      "attributes": {
        "Product": ["..."],
        "Service": ["..."],
        ...
      }
    }
    """
    if comment is None or not isinstance(comment, str) or not comment.strip():
        return {"comment": comment, "attributes": {}}

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(comment)},
        ]
        # no response_format here (compatible with older SDK)
    )

    # Get the text the model returned
    text_content = response.output[0].content[0].text

    try:
        data = json.loads(text_content)
    except json.JSONDecodeError:
        data = {"comment": comment, "attributes": {}}

    # Normalize keys
    if "comment" not in data:
        data["comment"] = comment

    attrs = data.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}

    # Ensure all values are lists of strings
    cleaned_attrs: Dict[str, List[str]] = {}
    for attr, spans in attrs.items():
        if spans is None:
            continue
        if isinstance(spans, str):
            cleaned_attrs[attr] = [spans]
        elif isinstance(spans, list):
            cleaned_attrs[attr] = [str(s) for s in spans if s is not None]

    data["attributes"] = cleaned_attrs

    return data


# ---------------------------------------------------------
# Full pipeline for a dataset
# ---------------------------------------------------------
def run_preprocessing(
    infile: str,
    comment_col: str = "Comment",
    id_col: Optional[str] = None,
    out_excel: str = "results/b2b_feedback_with_attribute_spans.xlsx",
    out_flat: str = "results/b2b_feedback_attribute_spans_flat.csv",
    out_jsonl: str = "results/b2b_feedback_attribute_spans.jsonl",
    model: str = "gpt-4.1-mini",
    limit: Optional[int] = None,
):
    """
    Run LLM preprocessing over an entire Excel file.

    - infile: path to input .xlsx
    - comment_col: column name containing comments
    - id_col: optional ID column to carry through
    - out_excel: path for Excel with JSON column
    - out_flat: path for flattened CSV
    - out_jsonl: path for JSONL with one record per comment
    - model: OpenAI model name
    - limit: if set, only process first N rows (for testing)

    Returns: (df_with_json, flat_df)
    """
    df = pd.read_excel(infile)

    if limit is not None:
        df = df.head(limit)

    results_raw: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        comment = str(row.get(comment_col, "") or "").strip()
        result = extract_attribute_spans(comment, model=model)
        results_raw.append(result)

        base: Dict[str, Any] = {
            "row_index": idx,
            "comment": result.get("comment", comment),
        }

        if id_col is not None and id_col in df.columns:
            base[id_col] = row.get(id_col)

        # Flatten attributes -> one row per (comment, attribute, span)
        attributes = result.get("attributes", {})
        if isinstance(attributes, dict):
            for attr, spans in attributes.items():
                # spans should already be a list of strings, but be defensive
                if isinstance(spans, str):
                    spans_iter = [spans]
                elif isinstance(spans, list):
                    spans_iter = spans
                else:
                    continue

                for span_text in spans_iter:
                    flat_rows.append(
                        {
                            **base,
                            "attribute": attr,
                            "text_span": span_text,
                        }
                    )

    # Attach JSON back to df
    df["llm_spans_json"] = [json.dumps(r, ensure_ascii=False) for r in results_raw]

    # Ensure results directory exists
    for path in [out_excel, out_flat, out_jsonl]:
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save Excel and flat CSV
    if out_excel is not None:
        df.to_excel(out_excel, index=False)
    flat_df = pd.DataFrame(flat_rows)
    if out_flat is not None:
        flat_df.to_csv(out_flat, index=False, encoding="utf-8-sig")

    # Save JSONL (one record per comment)
    if out_jsonl is not None:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in results_raw:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return df, flat_df

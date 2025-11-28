# setup_ecoplus_structure.py

import os
from pathlib import Path

ROOT = Path("ecoplus-cs-pipeline")

# ---------- directory structure ----------
dirs = [
    ROOT,
    ROOT / "configs",
    ROOT / "configs" / "cs_scoring",
    ROOT / "data",
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "src",
    ROOT / "src" / "common",
    ROOT / "src" / "cs_scoring",
    ROOT / "src" / "cs_scoring" / "mapping",
    ROOT / "src" / "cs_scoring" / "scoring",
    ROOT / "src" / "cs_scoring" / "experiments_legacy",
    ROOT / "src" / "summarization",
    ROOT / "src" / "issue_tracking",
    ROOT / "reports",
    ROOT / "reports" / "figures",
    ROOT / "notebooks",
]

# ---------- files & minimal contents ----------
files_with_content = {
    ROOT / "README.md": "# Ecoplus CS Pipeline\n\nShort description here.\n",
    ROOT / ".gitignore": "\n".join([
        ".env",
        "__pycache__/",
        "*.pyc",
        ".ipynb_checkpoints/",
        "data/",
        "reports/figures/",
    ]) + "\n",
    ROOT / "requirements.txt": "\n".join([
        "pandas",
        "numpy",
        "scikit-learn",
        "python-dotenv",
        "requests",
        "tqdm",
        "matplotlib",
        "transformers",
        "torch",
        # add openai / other libs as needed
    ]) + "\n",
    ROOT / ".env.example": "OPENAI_API_KEY=your-key-here\nENVIRONMENT=dev\n",
    # configs
    ROOT / "configs" / "cs_scoring" / "sentiment_baseline.yaml":
        "# config for sentiment-based mapping → CS score\n",
    ROOT / "configs" / "cs_scoring" / "llm_zero_shot.yaml":
        "# config for zero-shot LLM mapping → CS score\n",
    ROOT / "configs" / "cs_scoring" / "llm_finetuned.yaml":
        "# config for finetuned LLM mapping → CS score\n",
    ROOT / "configs" / "summarizer.yaml":
        "# config for summarization (improvement comments)\n",
    ROOT / "configs" / "issue_tracking.yaml":
        "# config for issue tracking pipeline\n",
    # common utils
    ROOT / "src" / "common" / "__init__.py": "",
    ROOT / "src" / "common" / "io_utils.py":
        '"""IO helpers: load/save CSVs, JSON, env, etc."""\n\n',
    ROOT / "src" / "common" / "preprocessing.py":
        '"""Text cleaning, tokenisation, chunking utilities."""\n\n',
    ROOT / "src" / "common" / "metrics.py":
        '"""Metric functions: MAE, RMSE, F1, QWK, etc."""\n\n',
    ROOT / "src" / "common" / "gpt_utils.py":
        '"""Wrappers around GPT API calls + caching."""\n\n',
    ROOT / "src" / "common" / "model_utils.py":
        '"""Helpers to load HuggingFace models, tokenizers, etc."""\n\n',
    # cs_scoring
    ROOT / "src" / "cs_scoring" / "__init__.py": "",
    ROOT / "src" / "cs_scoring" / "full_pipeline.py":
        '"""Run mapping + issue resolution + regression to CS score."""\n\n',
    # mapping stage
    ROOT / "src" / "cs_scoring" / "mapping" / "__init__.py": "",
    ROOT / "src" / "cs_scoring" / "mapping" / "sentiment_pipeline.py":
        '"""Sentiment-based mapping: text → aspect scores 1–5."""\n\n',
    ROOT / "src" / "cs_scoring" / "mapping" / "llm_zero_shot.py":
        '"""Zero-shot LLM classifier: text → aspect scores 1–5."""\n\n',
    ROOT / "src" / "cs_scoring" / "mapping" / "llm_finetuned.py":
        '"""Finetuned LLM classifier: text → aspect scores 1–5."""\n\n',
    ROOT / "src" / "cs_scoring" / "mapping" / "evaluate_mapping.py":
        '"""Evaluate mapping quality (per-aspect MAE, accuracy within ±1, etc.)."""\n\n',
    # scoring stage
    ROOT / "src" / "cs_scoring" / "scoring" / "__init__.py": "",
    ROOT / "src" / "cs_scoring" / "scoring" / "issue_resolution.py":
        '"""Implements issue resolution recurrence x_{i,t+1} = α x_{i,t} + k(1-α) Ω_{i,t+1}."""\n\n',
    ROOT / "src" / "cs_scoring" / "scoring" / "linreg_model.py":
        '"""Linear regression model mapping features → final CS score."""\n\n',
    ROOT / "src" / "cs_scoring" / "scoring" / "evaluate_cs.py":
        '"""Evaluate end-to-end CS scoring (MAE, RMSE, QWK, etc.)."""\n\n',
    # legacy experiments
    ROOT / "src" / "cs_scoring" / "experiments_legacy" / "__init__.py": "",
    ROOT / "src" / "cs_scoring" / "experiments_legacy" / "README.md":
        "# Legacy experiments\n\nBad/abandoned structures live here for reference.\n",
    # summarization
    ROOT / "src" / "summarization" / "__init__.py": "",
    ROOT / "src" / "summarization" / "summarizer_model.py":
        '"""LLM-based improvement-comment extraction."""\n\n',
    ROOT / "src" / "summarization" / "evaluate_summ.py":
        '"""Evaluate summaries: expert ratings + aspect coverage metrics."""\n\n',
    # issue tracking
    ROOT / "src" / "issue_tracking" / "__init__.py": "",
    ROOT / "src" / "issue_tracking" / "issue_model.py":
        '"""Classify issue type + fill issue tracker fields."""\n\n',
    ROOT / "src" / "issue_tracking" / "evaluate_issues.py":
        '"""Evaluate issue tracking: macro-F1, field accuracy, record match."""\n\n',
    # notebooks (empty .ipynb placeholders)
    ROOT / "notebooks" / "01_eda.ipynb": "",
    ROOT / "notebooks" / "02_cs_scoring.ipynb": "",
    ROOT / "notebooks" / "03_summarization.ipynb": "",
    ROOT / "notebooks" / "04_issue_tracking.ipynb": "",
}

def main():
    # create directories
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # create files
    for path, content in files_with_content.items():
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    print(f"Created project skeleton under: {ROOT.resolve()}")

if __name__ == "__main__":
    main()

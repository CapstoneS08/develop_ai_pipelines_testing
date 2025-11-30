# AI Pipelines Testing
Testing for AI models in the Develop phase of the double diamond, consisting of three major NLP pipelines:

1. **CS Scoring Model**  
   - Mapping stage (Sentiment vs. LLM)  
   - Issue-resolution smoothing  
   - Linear regression to final CS Performance Score (1–5)

2. **Summarization Model**  
   - Extracts *only* actionable improvement comments from long WhatsApp/feedback text

3. **Issue Tracking Model**  
   - Classifies incoming text into one of six issue types  
   - Auto-fills structured issue fields for internal tracking

The system is modular, configuration-driven, and designed for reproducible experiments.

---

## 📁 Repository Structure

    ecoplus-cs-pipeline/
    ├─ README.md
    ├─ .gitignore
    ├─ requirements.txt
    ├─ .env.example
    │
    ├─ configs/
    │  ├─ cs_scoring/
    │  │  ├─ sentiment_baseline.yaml
    │  │  ├─ llm_zero_shot.yaml
    │  │  └─ llm_finetuned.yaml
    │  ├─ summarizer.yaml
    │  └─ issue_tracking.yaml
    │
    ├─ data/
    │  ├─ raw/
    │  └─ processed/
    │
    ├─ src/
    │  ├─ common/
    │  │  ├─ io_utils.py
    │  │  ├─ preprocessing.py
    │  │  ├─ metrics.py
    │  │  ├─ gpt_utils.py
    │  │  └─ model_utils.py
    │  │
    │  ├─ cs_scoring/
    │  │  ├─ full_pipeline.py
    │  │  ├─ mapping/
    │  │  │  ├─ sentiment_pipeline.py
    │  │  │  ├─ llm_zero_shot.py
    │  │  │  ├─ llm_finetuned.py
    │  │  │  └─ evaluate_mapping.py
    │  │  ├─ scoring/
    │  │  │  ├─ issue_resolution.py
    │  │  │  ├─ linreg_model.py
    │  │  │  └─ evaluate_cs.py
    │  │  └─ experiments_legacy/
    │  │      └─ README.md
    │  │
    │  ├─ summarization/
    │  │  ├─ summarizer_model.py
    │  │  └─ evaluate_summ.py
    │  │
    │  └─ issue_tracking/
    │     ├─ issue_model.py
    │     └─ evaluate_issues.py
    │
    ├─ reports/
    │  └─ figures/
    │
    └─ notebooks/
       ├─ 01_eda.ipynb
       ├─ 02_cs_scoring.ipynb
       ├─ 03_summarization.ipynb
       └─ 04_issue_tracking.ipynb

---

## ⚙️ Setup Instructions

1. **Install dependencies**

    pip install -r requirements.txt

2. **Create your `.env`**

    cp .env.example .env

Fill in at least:

    OPENAI_API_KEY=your-key-here

---

## 🧠 1. CS Scoring Model – Overview

The **CS Performance Score Model** predicts a 1–5 score for each customer/project based on incoming WhatsApp and feedback messages.

It consists of three stages:

1. **Stage 1 – Mapping**: Text → aspect scores (1–5) for `Product`, `Service`, `Delivery`, `Payment`, …  
2. **Stage 2 – Issue Resolution Smoothing**: Apply recurrence to combine historical and new signals.  
3. **Stage 3 – Linear Regression**: Map features to final CS Performance Score (1–5).

---

### Stage 1 — Mapping (Text → Aspect Scores 1–5)

We extract scores for predefined aspects:

- Product  
- Service  
- Delivery  
- Payment  

There are two main mapping families: **sentiment-based** and **LLM-based**.

---

### A. Sentiment-based Mapping (current pipeline)

Flow:

1. LLM preprocessing extracts points and assigns aspects.  
2. A sentiment model (Cardiff or other HF models) produces sentiment scores for each point.  
3. A follow-up LLM call maps sentiment scores back to aspect scores from 1–5.

This tests whether sentiment signal can reliably approximate aspect quality.

Notes:

- Baseline model: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- Other HF models can be plugged in via config.  
- Sentiment model fine-tuning is planned but not yet attempted.

---

### B. LLM-based Mapping (current pipeline)

Flow:

1. The LLM directly classifies text into aspect scores 1–5.  
2. Output format (example):

       {
         "Service": 1,
         "Payment": 5
       }

Variants:

- **Zero-shot** LLM mapping (prompt-only).  
- **Finetuned classifier** (dataset prepared; to be integrated).

This is the **primary mapping approach** going forward.

---

### Stage 2 — Issue Resolution Smoothing

Implements the recurrence:

    x_{t+1} = α x_t + k (1 − α) Ω_{t+1}

where:

- `x_t` = previous score  
- `Ω_{t+1}` = new observation  
- `α` = smoothing factor  
- `k` = modifier depending on satisfaction (not satisfied / neutral / satisfied)

This models how successfully resolved issues influence ongoing satisfaction.

---

### Stage 3 — Linear Regression Model

Uses:

- Aspect scores from Stage 1  
- Issue-resolution features from Stage 2  
- (Optionally) additional structured features

Output:

- Final **CS Performance Score** (1–5) per customer / project.  
- Can also generate a time-series of scores for trend visualisation.

---

## 📊 Metrics

### Mapping metrics (Stage 1)

- Per-aspect **MAE** (mean absolute error)  
- Per-aspect **accuracy within ±1 point**  
- Macro-averaged MAE across aspects  
- Optional: **Quadratic Weighted Kappa (QWK)** per aspect

### CS Performance metrics (Stage 3)

- **MAE**  
- **RMSE**  
- **R²**  
- **Accuracy within ±1 point**  
- **QWK**

These metrics are computed in `src/cs_scoring/mapping/evaluate_mapping.py` and `src/cs_scoring/scoring/evaluate_cs.py`.

---

## 🧪 Experiments Legacy (Mapping Stage)

These early experiments attempted to compute CX/CS scores **without aspect mapping** and were later abandoned. They are kept under `src/cs_scoring/experiments_legacy/` for reference.

Common issue across all:  

> Sentiment alone does not provide aspect-level information, so downstream CS scores were unstable and hard to interpret.

### Legacy Experiment 1 — Combined Chunk → Sentiment → LLM → CX Score

- Combined positive/negative content into a single chunk.  
- Produced one sentiment score for the chunk.  
- Asked LLM to map this sentiment to a CX score directly (1–5), without aspects.

**Reason abandoned:** Could not distinguish between Delivery vs Service vs Product problems; good vs bad experiences were flattened.

---

### Legacy Experiment 2 — P/N/N Lists → Sentiment → LLM → CX Score

- LLM produced:

  - List of positive points  
  - List of negative points  
  - (Sometimes) list of neutral points  

- Sentiment model scored each list / point.  
- LLM mapped aggregated sentiment back to a single CX score.

**Reason abandoned:**  
Although more granular than Experiment 1, there were still **no aspect labels**; CS scores were difficult to relate to specific operational issues.

---

### Legacy Experiment 3 — Combined Chunk → Sentiment → Math → CX Score

- Produced one sentiment score for the entire chunk.  
- Used manual mathematical mappings to convert sentiment probability into a 1–5 CX score.  

  Example idea:

      CX = 5 * P(positive) + 3 * P(neutral) + 1 * P(negative)

**Reason abandoned:**  
Heuristic mapping was brittle and behaved badly for mixed or ambiguous comments. Without aspect separation, there was no way to know *why* a score changed.

---

## 🧪 Legacy vs Current Pipelines — Quick Summary

**CS Scoring Model**

- **Mapping**
  - **Sentiment-analysis**
    - Current pipeline  
      - LLM aspect extraction → sentiment model → LLM mapping → aspect scores.  
    - Experiments-legacy  
      - Chunk sentiment → LLM → CX (no aspects)  
      - P/N/N → LLM → CX (no aspects)  
      - Sentiment → math → CX (no aspects)
  - **LLM**
    - Current pipeline  
      - Zero-shot LLM, and a finetuned classifier (to be integrated).  
    - Experiments-legacy  
      - None (LLM mapping was introduced after aspect design).

- **Linear Regression Model**
  - Uses aspect and issue-resolution features to predict final CS score.

- **Overall Pipeline**
  - Mapping → Issue Resolution Smoothing → Linear Regression → Final CS Performance Score.

---

## 📝 2. Summarization Model

Goal: extract **only improvement comments** from long messages.

Example:

- Input:  
  “Delivery was slow but product is good. Please update us faster next time.”  

- Output:  
  “Improve delivery speed and response time.”

Implementation (high level):

- Uses an LLM with instruction-style prompts to generate concise, actionable improvement comments.  
- May optionally use aspect mapping to check coverage of key issues.

Evaluation:

- **Expert ratings** on a small sample:  
  - Actionability (1–5)  
  - Faithfulness (1–5)  
  - Conciseness (1–5)  
- **Aspect coverage**:  
  - Compare aspects in original text vs summary (precision / recall).

---

## 🛠️ 3. Issue Tracking Model

Classifies incoming text into the six internal **issue types** (from the issue tracker sheet), e.g.:

- Delay  
- Product Quality  
- Stock Issues  
- Service Issues  
- Fulfillment Error  
- Payments

Then auto-fills structured fields in the issue tracker, such as:

- Created By  
- Created At  
- To Inform  
- Assigned To  
- Resolved At  
- Status to Close  
- Closed At  
- Remarks  

Evaluation:

- **Macro-F1** for issue-type classification.  
- **Per-field accuracy** for each structured field.  
- **Record-level exact match**: percentage of rows where all fields are correct.

---

## 🚀 Running the Pipelines (high level)

Exact commands will depend on how you implement the scripts, but the general pattern is:

- **CS Scoring: Mapping evaluation**

      python src/cs_scoring/mapping/evaluate_mapping.py

- **CS Scoring: Full pipeline**

      python src/cs_scoring/full_pipeline.py
      python src/cs_scoring/scoring/evaluate_cs.py

- **Summarizer**

      python src/summarization/summarizer_model.py
      python src/summarization/evaluate_summ.py

- **Issue Tracking**

      python src/issue_tracking/issue_model.py
      python src/issue_tracking/evaluate_issues.py

You can also use the notebooks in `notebooks/` for exploratory runs and sanity checks.

---

## 🔭 Future Work

- Integrate and compare finetuned LLM mapping vs zero-shot.  
- Fine-tune sentiment models for better domain adaptation.  
- Incorporate issue-tracking features directly into CS scoring.  
- Build a labelled dataset for supervised summarization and potential instruction fine-tuning.  
- Add more aspects and support per-customer trend dashboards.

---

## 📄 License / Usage

This repository is intended for the Ecoplus Capstone project and internal academic use.  
External use or distribution should be approved by the project stakeholders.

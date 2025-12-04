# Ecoplus Customer Experience AI Pipelines

This repository contains experimental AI pipelines developed for **Ecoplus**, a B2B pipes/foundry SME in Singapore.  
The goal is to explore how AI can support customer-experience workflows by:

- scoring customer satisfaction (per aspect)
- generating actionable improvement suggestions
- converting customer WhatsApp complaints into structured issue-tracking rows

These pipelines are prototypes created for feasibility testing.  
The repository contains **three main models**:

1. **CS Scoring Model**  
2. **Summarising Model**  
   - Summarising Postprocessing  
3. **Issue Tracking Model**

---

# 1. Environment Setup

## Python
Use **Python 3.10+**.

## Install dependencies
```bash
pip install -r requirements.txt
```

## OpenAI API Key
Create a `.env` file:
```ini
OPENAI_API_KEY=your_key_here
```

Or export manually:
```bash
export OPENAI_API_KEY=your_key_here
```

---

# 2. Repository Structure (High-Level)

```
AI_PIPELINE_TESTING/
├── data/
│   ├── processed/
│   ├── raw/
│   └── synthetic/
│
├── notebooks/
│   ├── cs_scoring/
│   │   ├── llm_preprocessing/
│   │   └── mapping/
│   │       ├── llm/
│   │       ├── sentiment_analysis/
│   │       └── seq2seq/
│   │
│   ├── issue_tracking/
│   │   ├── ed/
│   │   ├── cel/
│   │   └── esd/
│   │
│   ├── linear_regression/
│   │
│   └── summarising/
│       ├── ed/
│       ├── cel/
│       └── esd/
│
└── utils/
    └── setup_ecoplus_structure.py
```

Notes:
- Contributor folders (`ed/`, `cel/`, `esd/`) reflect ownership of prototype notebooks.
- `setup_ecoplus_structure.py` provides a **future production-ready modular layout** for Ecoplus (configs/, src/, utils/, etc.) but is not used in this experimental repo.

---

# 3. Model 1 — CS Scoring Model

The CS Scoring Model transforms raw customer messages into:

1. **attribute-level text spans** (Product / Service / Delivery / Payment)
2. **attribute-level satisfaction scores** (1–5)
3. **an overall CS Performance Score** (via linear regression)

It contains:
- LLM Preprocessing  
- Mapping Models (multiple variants)  
- CS Score Regression  

---

## 3.1 LLM Preprocessing  
*(attribute → exact text spans)*

### Purpose
Extract **verbatim text spans** mentioning each customer satisfaction attribute:

- Product  
- Service  
- Delivery  
- Payment  

### Example Input
```
"Kim is very nice, but products always come late."
```

### Example Output
```json
{
  "comment": "Kim is very nice, but products always come late",
  "spans": [
    { "attribute": "Service",  "text_span": "Kim is very nice" },
    { "attribute": "Delivery", "text_span": "products always come late" }
  ]
}
```

### Notes
- Uses **OpenAI GPT (gpt-4.1-mini)**  
- Extracts exact substrings (no paraphrasing)  
- Omits attributes not mentioned  
- One span may map to multiple attributes  

Location:  
`notebooks/cs_scoring/llm_preprocessing/`

---

## 3.2 Mapping Models  
*(text span → score 1–5)*

There are **two independent families** of mapping pipelines:

1. Sentiment-based pipelines (legacy)  
2. Direct mapping (LLM or Seq2Seq)  

---

### 3.2.1 Sentiment-Based Mapping (Legacy Pipelines)

These pipelines were **discarded** because **they did not use aspect extraction**, making them incompatible with attribute-level CS scoring and regression.

#### A. Chunk-Based Pipeline *(scrapped)*
- Split message into positive/negative “chunks”
- Sentiment applied per chunk
- LLM aggregated into a CX score  
*Did not use aspect extraction.*

#### B. Individual-Point Pipeline *(scrapped)*
- LLM extracted positive, negative, neutral points
- Sentiment applied per point
- LLM aggregated into a CX score  
*Did not use aspect extraction.*

Location:  
`notebooks/cs_scoring/mapping/sentiment_analysis/`

---

### 3.2.2 Direct Mapping Pipelines  
*(LLM or Seq2Seq → score 1–5)*

Pipeline flow:
```
Raw Text
 → LLM Preprocessing
 → Direct Mapping (LLM or Seq2Seq)
 → Attribute Score (1–5)
```

---

### A. LLM Direct Mapping

#### Example Input
```json
{
  "attribute": "Delivery",
  "text_span": "products always come late"
}
```

#### Example Output
```json
{
  "score": 2
}
```

Location:  
`notebooks/cs_scoring/mapping/llm/`

---

### B. Seq2Seq Mapping (Fine-Tuned Models)

Models trained and tested:
- **google/flan-t5-base**  
- **facebook/bart-base**

#### Training Example
```json
{
  "attribute": "Service",
  "text_span": "Sales team is very responsive on WhatsApp",
  "score": 5
}
```

Outputs:
- Predictions CSV  
- Predictions JSONL  
- Checkpoints (ignored by Git)  

Location:  
`notebooks/cs_scoring/mapping/seq2seq/`

---

## 3.3 CS Score Regression

A simple linear regression combines:
- attribute-level scores  
- optional issue-related variables  

It outputs a **final CS Performance Score**.

Location:  
`notebooks/linear_regression/`

---

# 4. Model 2 — Summarising Model

This model generates:
- one actionable **improvement suggestion** per customer message  
- a postprocessing step that groups these suggestions into **aspect-level CX insights**  

---

## 4.1 Summarising Model  
*(Improvement Comment Generator)*

### Purpose
Turn raw customer comments into a **single, business-tone improvement suggestion**.

### Example Input
```
"Driver never updated us about delay yesterday."
```

### Example Output
```json
{
  "comment_id": 12,
  "improvement_comment": "Provide customers with timely delivery status updates."
}
```

Location:  
`notebooks/summarising/*/pipeline.py`

---

## 4.2 Summarising Postprocessing  
*(Aggregation into CX Themes)*

### Example Input
```
[
  "Respond faster on WhatsApp.",
  "Give clearer expectations when delays happen.",
  "Contact customers earlier when drivers run late."
]
```

### Example Output
```
SERVICE
- Respond more promptly to WhatsApp queries.
- Provide clearer expectations when delays occur.

DELIVERY
- Contact customers earlier during delays.
```

Location:  
`notebooks/summarising/*/postprocessing.py`

---

# 5. Model 3 — Issue Tracking Model  
*(WhatsApp → structured issue row)*

Converts a raw WhatsApp message into a **fully structured issue-tracking entry** used in Ecoplus’ workflow.

## Example Input
```json
{
  "chat_id": "65018",
  "message_id": "WAM_040",
  "timestamp": "2024-11-16 10:34:17",
  "sender_name": "Ang Mo Kio Hardware",
  "message_text": "Bro suppose to deliver yesterday..."
}
```

## Example Output
```json
{
  "issue": "Delivery Delay",
  "customer": "Ang Mo Kio Hardware",
  "created_by": "Customer",
  "created_at": "2024-11-16 10:34:17",
  "to_inform": "AM, Cust Svc",
  "assigned_to": "Cust Svc",
  "resolved_at": "",
  "closed_at": "",
  "resolution_score": "Not resolved yet",
  "comments": "Customer chasing overdue delivery; check driver status and provide confirmed ETA."
}
```

Notes:
- Uses OpenAI GPT with **strict JSON schema**  
- Includes rules for missing information  
- Prevents hallucinating names/timestamps  

Location:  
`notebooks/issue_tracking/`

---

# 6. Additional Notes

- Contributor folders (`ed/`, `cel/`, `esd/`) reflect work ownership but have **no functional meaning** in the pipeline.  
- `setup_ecoplus_structure.py` proposes a **future production-ready modular layout** for Ecoplus’ final system.  

---

# End of README
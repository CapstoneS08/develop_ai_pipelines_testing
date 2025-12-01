import os
import json
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
import re


# ---------------------------------------------------------
# Load JSONL Records
# ---------------------------------------------------------
def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------
# Helper: build (attribute, text, score) examples from a record
# - Handles BOTH flat and chat-style schemas
# ---------------------------------------------------------
def record_to_examples(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = []

    # Case 1: flat format with explicit fields
    if "attribute" in rec and "text_span" in rec and "score" in rec:
        examples.append(
            {
                "attribute": rec["attribute"],
                "text_span": rec["text_span"],
                "score": rec["score"],
            }
        )
        return examples

    # Case 2: chat-style { "messages": [...] } format
    if "messages" in rec:
        msgs = rec.get("messages", [])

        # user message text
        user_msg = None
        for m in msgs:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
        if not user_msg:
            return examples

        if "Message:" in user_msg:
            span_text = user_msg.split("Message:", 1)[1].strip()
        else:
            span_text = user_msg.strip()

        # assistant JSON with aspects
        assistant_msg = None
        for m in msgs:
            if m.get("role") == "assistant":
                assistant_msg = m.get("content", "")
                break
        if not assistant_msg:
            return examples

        try:
            label_obj = json.loads(assistant_msg)
        except json.JSONDecodeError:
            return examples

        aspects = label_obj.get("aspects", {})
        for aspect_name, score in aspects.items():
            attr = aspect_name.capitalize()
            examples.append(
                {
                    "attribute": attr,
                    "text_span": span_text,
                    "score": int(score),
                }
            )

    return examples


# ---------------------------------------------------------
# Build HF Dataset (works for chat-style OR flat JSONL)
# ---------------------------------------------------------
def make_seq2seq_dataset(
    jsonl_path: str,
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 8,
) -> Dataset:
    records = load_jsonl_records(jsonl_path)

    all_examples: List[Dict[str, Any]] = []
    for rec in records:
        all_examples.extend(record_to_examples(rec))

    if not all_examples:
        raise ValueError(f"No usable examples found in {jsonl_path}")

    def _to_model_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        attr = ex["attribute"]
        span = ex["text_span"]
        score = ex["score"]

        input_text = (
            f"rate the sentiment score (1-5) for the {attr} aspect "
            f"of this text:\n{span}"
        )
        target_text = str(score)

        return {"input_text": input_text, "target_text": target_text}

    model_examples = [_to_model_example(e) for e in all_examples]
    ds = Dataset.from_list(model_examples)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_ds = ds.map(preprocess, batched=True)
    return tokenized_ds


# ---------------------------------------------------------
# Train Seq2Seq Model
# ---------------------------------------------------------
def train_seq2seq_model(
    model_name: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 8,
    learning_rate: float = 5e-5,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔧 Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("📥 Building train dataset from:", train_jsonl)
    train_dataset = make_seq2seq_dataset(train_jsonl, tokenizer)
    print("📥 Building validation dataset from:", val_jsonl)
    val_dataset = make_seq2seq_dataset(val_jsonl, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Minimal arguments so it's compatible with older transformers
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer, tokenizer, model


# ---------------------------------------------------------
# Predict scores
# ---------------------------------------------------------
def predict_scores_for_spans(
    model_path: str,
    spans_df: pd.DataFrame,
    attribute_col: str = "attribute",
    text_col: str = "text_span",
    batch_size: int = 16,
    max_input_length: int = 256,
    max_new_tokens: int = 4,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    inputs = [
        f"rate the sentiment score (1-5) for the {row[attribute_col]} aspect "
        f"of this text:\n{row[text_col]}"
        for _, row in spans_df.iterrows()
    ]

    pred_scores = []

    for i in range(0, len(inputs), batch_size):
        batch_texts = inputs[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**enc, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for txt in decoded:
            cleaned = txt.strip()
            m = re.search(r"[1-5]", cleaned)
            score = int(m.group()) if m else None
            pred_scores.append(score)

    result_df = spans_df.copy()
    result_df["pred_score"] = pred_scores
    return result_df
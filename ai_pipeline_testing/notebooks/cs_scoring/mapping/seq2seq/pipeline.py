import os
import json
from typing import List, Dict, Any, Optional

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
# Build HF Dataset
# ---------------------------------------------------------
def make_seq2seq_dataset(
    jsonl_path: str,
    tokenizer,
    max_input_length: int = 128,
    max_target_length: int = 8,
) -> Dataset:
    records = load_jsonl_records(jsonl_path)

    def _to_examples(rec: Dict[str, Any]) -> Dict[str, Any]:
        attr = rec.get("attribute", "")
        span = rec.get("text_span", "")
        score = rec.get("score", None)
        input_text = f"Attribute: {attr}\nText: {span}"
        target_text = str(score) if score is not None else ""
        return {"input_text": input_text, "target_text": target_text}

    examples = [_to_examples(r) for r in records]
    ds = Dataset.from_list(examples)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
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

    train_dataset = make_seq2seq_dataset(train_jsonl, tokenizer)
    val_dataset = make_seq2seq_dataset(val_jsonl, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=20,
        report_to=[],
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
    max_input_length: int = 128,
    max_new_tokens: int = 4,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    inputs = [
        f"Attribute: {row[attribute_col]}\nText: {row[text_col]}"
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
            score = None
            for ch in cleaned:
                if ch in "12345":
                    score = int(ch)
                    break
            pred_scores.append(score)

    result_df = spans_df.copy()
    result_df["pred_score"] = pred_scores
    return result_df
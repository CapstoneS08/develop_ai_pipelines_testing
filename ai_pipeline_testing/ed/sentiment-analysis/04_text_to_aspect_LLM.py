import json
import os
from collections import defaultdict
from openai import OpenAI
import re

# Load your validation file
VALIDATION_PATH = r"D:\GitHub Coding\Capstone\validation_gpt_50.jsonl"

# Your fine-tuned model name
MODEL = INSERT
client = INSERT
# ========= HELPERS =========

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def safe_parse_json(s: str):
    """
    Try to robustly parse JSON from the model output.
    Handles:
    - pure JSON
    - JSON wrapped in extra text
    - ```json fenced blocks
    """
    s = s.strip()

    # 1) Direct attempt
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Remove markdown fences if present
    if s.startswith("```"):
        # remove starting fence with optional language
        s = re.sub(r"^```[a-zA-Z]*", "", s)
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
        try:
            return json.loads(s)
        except Exception:
            pass

    # 3) Extract substring between first { and last }
    if "{" in s and "}" in s:
        try:
            s2 = s[s.index("{"): s.rindex("}") + 1]
            return json.loads(s2)
        except Exception:
            pass

    return None


# ========= METRICS TRACKING =========

correct = defaultdict(int)
total = defaultdict(int)
abs_error_sum = defaultdict(float)

num_examples = 0
num_parsed = 0
num_failed_parse = 0

# ========= MAIN VALIDATION LOOP =========

print("=== RUNNING VALIDATION ===")

data = load_jsonl(VALIDATION_PATH)

example_idx = 0  # for labeling examples in logs

for item in data:
    num_examples += 1
    example_idx += 1

    messages = item["messages"]
    system_prompt = messages[0]["content"]
    user_msg = messages[1]["content"]

    # True label: assistant content is a JSON string with {"aspects": {...}}
    true_label = json.loads(messages[2]["content"])
    true_aspects = true_label.get("aspects", {})

    # --- PRINT INPUT + GOLD LABEL FOR SLIDES ---
    print("\n" + "=" * 60)
    print(f"EXAMPLE #{example_idx}")
    print("- USER MESSAGE -----------------------------")
    print(user_msg)
    print("- TRUE LABEL (JSON) ------------------------")
    print(json.dumps(true_label, indent=2))

    # Call model
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        continue

    raw = completion.choices[0].message.content

    # --- PRINT RAW MODEL OUTPUT FOR SLIDES ---
    print("- MODEL RAW OUTPUT ------------------------")
    print(raw)

    parsed = safe_parse_json(raw)

    if parsed is None:
        num_failed_parse += 1
        print("[WARN] Could not parse JSON in response.")
        print("---")
        continue

    num_parsed += 1
    pred_aspects = parsed.get("aspects", {})

    # --- PRINT PARSED PREDICTION FOR SLIDES ---
    print("- PARSED PREDICTION (JSON) ----------------")
    print(json.dumps(parsed, indent=2))

    # --- PER-EXAMPLE MAPPING TABLE (GOOD FOR SLIDES) ---
    print("- ASPECT MAPPING TABLE --------------------")
    print(f"{'ASPECT':20} {'TRUE':>5} {'PRED':>5} {'ABS ERR':>7}")
    print("-" * 40)

    # Compare predicted vs true aspects
    # 1) For each true aspect, check prediction
    for aspect, true_score in true_aspects.items():
        total[aspect] += 1

        if aspect not in pred_aspects:
            # Missed aspect → count as wrong, and penalize with max error (5)
            abs_error_sum[aspect] += 5
            print(f"{aspect:20} {true_score:5} {'-':>5} {5:7.2f}")
            continue

        pred_score = pred_aspects[aspect]

        # If they didn't send an int for some reason, try to coerce
        try:
            pred_score_int = int(pred_score)
        except Exception:
            pred_score_int = 0

        if pred_score_int == true_score:
            correct[aspect] += 1

        err = abs(pred_score_int - true_score)
        abs_error_sum[aspect] += err

        print(f"{aspect:20} {true_score:5} {pred_score_int:5} {err:7.2f}")

    # 2) Penalize extra predicted aspects that are not in truth
    for aspect in pred_aspects:
        if aspect not in true_aspects:
            total[aspect] += 1
            try:
                pred_score_int = int(pred_aspects[aspect])
            except Exception:
                pred_score_int = 0
            # Treat hallucinated aspect as full error vs neutral 3 (or just 5)
            abs_error_sum[aspect] += 5
            print(f"{aspect:20} {'-':>5} {pred_score_int:5} {5:7.2f}")

print("\n=== EVALUATION RESULTS ===")
print(f"Total examples in file: {num_examples}")
print(f"Successfully parsed model outputs: {num_parsed}")
print(f"Failed to parse model outputs: {num_failed_parse}")

if not total:
    print("No aspects evaluated. Check that your validation file has 'aspects' labels.")
else:
    for aspect in sorted(total.keys()):
        if total[aspect] == 0:
            continue
        acc = correct[aspect] / total[aspect]
        mae = abs_error_sum[aspect] / total[aspect]
        print(f"\nASPECT: {aspect.upper()}")
        print(f"  Total pairs: {total[aspect]}")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Mean Abs Error: {mae:.3f}")
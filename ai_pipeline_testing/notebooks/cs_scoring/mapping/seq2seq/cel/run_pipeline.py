import os

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
    print("Project root:", project_root)

    train_jsonl = os.path.join(
        project_root,
        "data", "processed", "training",
        "whatsapp_aspect_training_300.jsonl"
    )

    print("Training file located at:", train_jsonl)

    # Nothing else needed — training happens in analysis.ipynb
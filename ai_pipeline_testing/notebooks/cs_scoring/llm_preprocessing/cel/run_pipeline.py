from pipeline import run_preprocessing
import os

if __name__ == "__main__":
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    infile = "B2B_Customer_Feedback_Dataset.xlsx"  # put the file in the project root
    comment_col = "Comment"  # change if your comment column has a different name

    run_preprocessing(
        infile=infile,
        comment_col=comment_col,
        id_col=None,  # or e.g. "Comment_ID"
        out_excel="results/b2b_feedback_with_attribute_spans.xlsx",
        out_flat="results/b2b_feedback_attribute_spans_flat.csv",
        out_jsonl="results/b2b_feedback_attribute_spans.jsonl",
        model="gpt-4.1-mini",
        limit=None,  # set to e.g. 50 if you want to test on first 50 rows
    )
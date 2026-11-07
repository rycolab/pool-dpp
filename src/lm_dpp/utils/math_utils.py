import re
import pandas as pd
from datasets import load_dataset

def load_gsm8k_dataset(split="test"):
    """
    Loads the GSM8K dataset from huggingface.
    Returns a pandas DataFrame with 'question' and 'answer' columns.
    Raises if the dataset cannot be loaded (it used to print the error and return an empty
    DataFrame, which made every caller run on zero questions).
    """
    dataset = load_dataset("gsm8k", "main", split=split)
    df = pd.DataFrame(dataset)
    if df.empty:
        raise ValueError(f"GSM8K split {split!r} loaded with no rows")
    return df

def extract_model_num(completion: str) -> str:
    text = str(completion)

    # Prefer "Answer: ..." / "Final: ..." / "#### ..." if present
    cue_patterns = [
        r"(?:^|\n)\s*(?:####\s*)?(-?\d[\d,]*(?:\.\d+)?)\s*$",   # number at end of text/line
        r"(?:Answer|Final|Therefore|So)\s*[:\-]?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]
    for pat in cue_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)

    # Fallback: take the LAST number anywhere in the text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1] if nums else ""

def normalize_num(s: str) -> str:
    return s.replace(",", "").strip()

def is_correct(completion: str, gold_answer: str) -> bool:
    """
    Checks if the extracted answer matches the gold answer.
    Gold answer in GSM8K usually contains the reasoning and then #### Answer.
    We need to extract the number after ####.
    """
    gold_num = gold_answer.split("####")[-1].strip()
    model_num = extract_model_num(completion)
    return normalize_num(model_num) == normalize_num(gold_num)

def build_math_prompt(question: str) -> str:
    """
    Standard prompt format for math problems.
    """
    return f"Question: {question}\nLet's think step by step and then return the final answer on the last line exactly as: #### <number>.\nAnswer:"

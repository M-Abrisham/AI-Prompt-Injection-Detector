import os
import pandas as pd

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' not installed. Run: pip install datasets")


# JAILBREAK Data (from GitHub)
JAILBREAK_URL = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv"
JAILBREAK_PATH = "data/raw/jailbreaks.csv"

# Good Data (Awesome ChatGPT Prompts dataset)
SAFE_URL = "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"
SAFE_PATH = "data/raw/safe_prompts.csv"

# Hugging Face datasets for additional safe prompts
HF_SAFE_PATH = "data/raw/hf_safe_prompts.csv"


def download_data(url, path):
    try:
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Downloaded {len(df)} rows to {path}")
        return len(df)
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        return 0


def download_hf_safe_prompts(path, max_samples=5000):
    """Download safe prompts from Hugging Face datasets."""
    if not HF_AVAILABLE:
        print("Skipping HF datasets (library not installed)")
        return 0

    safe_prompts = []

    try:
        # Alpaca dataset - instruction following (safe)
        print("Loading Alpaca dataset...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        for item in alpaca.select(range(min(2000, len(alpaca)))):
            if item.get("instruction"):
                safe_prompts.append(item["instruction"])

        # Dolly dataset - safe instructions
        print("Loading Dolly dataset...")
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for item in dolly.select(range(min(2000, len(dolly)))):
            if item.get("instruction"):
                safe_prompts.append(item["instruction"])

        # OpenAssistant - safe conversations
        print("Loading OpenAssistant dataset...")
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        count = 0
        for item in oasst:
            if item.get("role") == "prompter" and item.get("text"):
                safe_prompts.append(item["text"])
                count += 1
                if count >= 1000:
                    break

    except Exception as e:
        print(f"Error loading HF dataset: {e}")

    # Deduplicate and limit
    safe_prompts = list(set(safe_prompts))[:max_samples]

    if safe_prompts:
        df = pd.DataFrame({"prompt": safe_prompts, "label": 0})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Downloaded {len(df)} safe prompts from Hugging Face to {path}")
        return len(df)

    return 0


if __name__ == "__main__":
    print("Downloading datasets...\n")

    # Download original datasets
    bad_count = download_data(JAILBREAK_URL, JAILBREAK_PATH)
    good_count = download_data(SAFE_URL, SAFE_PATH)

    # Download additional safe prompts from Hugging Face
    hf_count = download_hf_safe_prompts(HF_SAFE_PATH)

    total_safe = good_count + hf_count
    print(f"\n--- Summary ---")
    print(f"Malicious prompts: {bad_count}")
    print(f"Safe prompts: {total_safe} (GitHub: {good_count}, HuggingFace: {hf_count})")

    if bad_count > 0 and total_safe > 0:
        ratio = bad_count / total_safe if total_safe > 0 else float('inf')
        print(f"Ratio (malicious:safe): {ratio:.1f}:1")
        if ratio > 3:
            print("Tip: Consider undersampling malicious data during training")
    else:
        print("\nSome downloads failed")

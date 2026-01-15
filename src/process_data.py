import pandas as pd
import os


JAILBREAK_PATH = "data/raw/jailbreaks.csv"
SAFE_PATH = "data/raw/safe_prompts.csv"
OUTPUT_PATH = "data/processed/combined_data.csv"

def merge_datasets():

    
    try:
        # Load Jailbreak data
        jailbreak_df = pd.read_csv(JAILBREAK_PATH)
        jailbreak_df['label'] = 1  # Malicious
        
        jailbreak_df = jailbreak_df.rename(columns={'prompt': 'text'})
        # clean-up
        jailbreak_df = jailbreak_df[['text', 'label']]

        # Load Safe data
        df_good = pd.read_csv(SAFE_PATH)
        df_good['label'] = 0  # Safe
    
        df_good = df_good.rename(columns={'prompt': 'text'})
        df_good = df_good[['text', 'label']]

        # Combine
        combined = pd.concat([jailbreak_df, df_good], axis=0).sample(frac=1, random_state=42)
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        
        print(f"✅ Success! Merged data saved to {OUTPUT_PATH}")
        print(f"Total: {len(combined)}")
        print(f"   - JAILBREAK: {len(jailbreak_df)}")
        print(f"   - SAFE: {len(df_good)}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    merge_datasets()
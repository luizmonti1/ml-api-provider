import pandas as pd
from pathlib import Path

# Define dataset root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "external_data" / "UCI HAR Dataset"

def load_data(split):
    # Load features
    X = pd.read_csv(DATA_DIR / split / f"X_{split}.txt", delim_whitespace=True, header=None)
    y = pd.read_csv(DATA_DIR / split / f"y_{split}.txt", header=None, names=["activity"])
    subjects = pd.read_csv(DATA_DIR / split / f"subject_{split}.txt", header=None, names=["subject"])
    
    # Merge
    df = pd.concat([subjects, y, X], axis=1)
    df["set"] = split
    return df

def load_feature_names():
    features = pd.read_csv(DATA_DIR / "features.txt", sep='\s+', header=None)
    raw_names = features[1].tolist()

    # Add subject, activity, set
    final_names = ["subject", "activity"] + raw_names + ["set"]

    # Fix duplicate names
    seen = {}
    for i, name in enumerate(final_names):
        if name not in seen:
            seen[name] = 1
        else:
            seen[name] += 1
            final_names[i] = f"{name}_{seen[name]}"

    return final_names


def load_activity_labels():
    labels = pd.read_csv(DATA_DIR / "activity_labels.txt", delim_whitespace=True, header=None, names=["id", "label"])
    return labels.set_index("id")["label"].to_dict()

def save_merged_data(df):
    raw_output = BASE_DIR / "data" / "raw"
    raw_output.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_output / "har_merged.parquet", index=False)
    print("✅ Merged HAR dataset saved to data/raw/har_merged.parquet")

if __name__ == "__main__":
    print("📥 Loading HAR Dataset...")

    # Load and combine
    df_train = load_data("train")
    df_test = load_data("test")
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    # Apply column names
    df_all.columns = load_feature_names()

    # Map activity names
    label_map = load_activity_labels()
    df_all["activity_name"] = df_all["activity"].map(label_map)

    save_merged_data(df_all)

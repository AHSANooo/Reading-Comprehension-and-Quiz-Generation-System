"""
data_splitter.py
----------------
Reads dev.csv from Google Drive, shuffles it, and splits it into
80% train / 10% validation / 10% test.

Checkpointing: If the processed/ folder already contains the three
split files, the script exits early without recomputing.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = "/content/drive/MyDrive/AI_Project_2026"
if not os.path.isdir(BASE_DIR):
    LOCAL_BASE = "/home/ahsan/Documents/Uni work/Sem 6/AI Lab/Project/AI_Project_2026"
    if os.path.isdir(LOCAL_BASE):
        BASE_DIR = LOCAL_BASE
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_CSV        = os.path.join(BASE_DIR, "dataset", "dev.csv")
PROCESSED_DIR  = os.path.join(BASE_DIR, "processed")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH   = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH  = os.path.join(PROCESSED_DIR, "test.csv")

RANDOM_STATE = 42


def mount_drive():
    """Mount Google Drive if running inside Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("[✓] Google Drive mounted.")
    except (ImportError, AttributeError):
        print("[INFO] Not running in an interactive Colab kernel — skipping Drive mount.")


def checkpoint_exists() -> bool:
    """Return True only when all three split files are already present."""
    return all(os.path.isfile(p) for p in [TRAIN_PATH, VAL_PATH, TEST_PATH])


def split_and_save(df: pd.DataFrame) -> None:
    """Perform an 80-10-10 stratified split and persist results."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 80 % train  |  20 % temp
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )

    # 50 % of temp → val (10 % overall)  |  50 % of temp → test (10 % overall)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH,   index=False)
    test_df.to_csv(TEST_PATH,  index=False)

    print(f"[✓] Split complete.")
    print(f"    Train : {len(train_df):>6,} rows  →  {TRAIN_PATH}")
    print(f"    Val   : {len(val_df):>6,} rows  →  {VAL_PATH}")
    print(f"    Test  : {len(test_df):>6,} rows  →  {TEST_PATH}")


def main():
    mount_drive()

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if checkpoint_exists():
        print("[✓] Checkpoint found — split files already exist. Skipping.")
        return

    # ── Load raw data ────────────────────────────────────────────────────────
    if not os.path.isfile(RAW_CSV):
        raise FileNotFoundError(
            f"Raw dataset not found at '{RAW_CSV}'.\n"
            "Please upload dev.csv to /dataset/ on your Drive."
        )

    print(f"[→] Loading raw dataset from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"    Total rows loaded: {len(df):,}")

    # ── Split ────────────────────────────────────────────────────────────────
    split_and_save(df)


if __name__ == "__main__":
    main()

import os
import shutil

# ==========================================
# 1. BENCHMARK TOGGLE (CHANGE THIS!)
# ==========================================
# Set to "xsub" for Cross-Subject or "xset" for Cross-Setup (NTU-120 standard)
BENCHMARK_MODE = "xsub120"  # Options: "xsub" or "xset" 

print(f"🗂️ Initializing NTU-RGBD 120 {BENCHMARK_MODE.upper()} Splitter (with Corruption Filter)...")

# ==========================================
# 2. DIRECTORY SETUP
# ==========================================
# Assuming you merged both NTU-60 and NTU-120 folders into one as discussed
RAW_DATA_DIR = "data/nturgbd_skeletons_s018_to_s032"

# Isolate the output folders based on the chosen benchmark
TRAIN_DIR = f"data/{BENCHMARK_MODE}/train_skeletons"
VAL_DIR = f"data/{BENCHMARK_MODE}/val_skeletons"

# NOTE: Make sure you download the official NTU-120 missing skeletons list!
MISSING_SKELETONS_TXT = "NTU_RGBD120_samples_with_missing_skeletons.txt"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ==========================================
# 3. BUILD THE BLACKLIST
# ==========================================
ignored_samples = set()
try:
    with open(MISSING_SKELETONS_TXT, 'r') as f:
        for line in f:
            clean_name = line.strip()
            if clean_name:
                ignored_samples.add(clean_name)
    print(f"🛡️ Loaded {len(ignored_samples)} corrupted filenames to ignore.")
except FileNotFoundError:
    print(f"⚠️ Warning: '{MISSING_SKELETONS_TXT}' not found! No files will be skipped.")

# ==========================================
# 4. THE SORTING ENGINE
# ==========================================
# The 53 training subjects for NTU-120 Cross-Subject
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]

total_files = 0
train_count = 0
val_count = 0
skipped_count = 0

print(f"⏳ Sorting files using {BENCHMARK_MODE.upper()} rules and copying to destination... ")

for filename in os.listdir(RAW_DATA_DIR):
    if not filename.endswith(".skeleton"):
        continue
        
    sample_name = filename.replace('.skeleton', '')
    
    # Check the blacklist!
    if sample_name in ignored_samples:
        skipped_count += 1
        continue
        
    total_files += 1
    
    # --- EXTRACT ID TAGS ---
    # Example Format: S001C001P001R001A060
    
    # 1. Extract Setup ID (e.g., S001 -> 1)
    setup_id_str = filename.split('S')[1][:3]
    setup_id = int(setup_id_str)
    
    # 2. Extract Person ID (e.g., P001 -> 1)
    person_id_str = filename.split('P')[1][:3]
    person_id = int(person_id_str)
    
    source_path = os.path.join(RAW_DATA_DIR, filename)
    dest_path = None
    
    # --- ROUTING LOGIC ---
    if BENCHMARK_MODE == "xsub120":
        # Cross-Subject: specific 53 subjects go to train, the rest to val
        if person_id in training_subjects:
            dest_path = os.path.join(TRAIN_DIR, filename)
            train_count += 1
        else:
            dest_path = os.path.join(VAL_DIR, filename)
            val_count += 1
            
    elif BENCHMARK_MODE == "xset120":
        # Cross-Setup: Even setups go to train, Odd setups go to val
        if setup_id % 2 == 0:
            dest_path = os.path.join(TRAIN_DIR, filename)
            train_count += 1
        else:
            dest_path = os.path.join(VAL_DIR, filename)
            val_count += 1
            
    # Safely COPY the file instead of moving it
    if dest_path:
        shutil.copy(source_path, dest_path)

print("-" * 50)
print(f"✅ {BENCHMARK_MODE.upper()} DATA SPLIT & FILTER COMPLETE")
print(f"Total Clean Skeletons Processed: {total_files}")
print(f"Corrupted Files Skipped:         {skipped_count}")
print(f"Copied to Training Set:          {train_count}")
print(f"Copied to Validation Set:        {val_count}")
print("-" * 50)
import os
import shutil

"""
# NTU-RGB+D 60 is not handled by this script. To prepare xsub or xview,
# create a separate NTU-60 splitter or make a dedicated configuration that:
# 1. Uses the NTU-60 skeleton source directory.
# 2. Uses the official NTU-60 cross-subject or cross-view split lists.
# 3. Applies the NTU-60 missing-skeleton list, if available.
# 4. Writes to data/xsub/ or data/xview/ with train_skeletons and
#    val_skeletons subdirectories.
# Do not replace the NTU-120 subject lists or missing-skeleton file here.
"""


# ==========================================
# BENCHMARK & DIRECTORY SETUP (CHANGE THIS!)
# ==========================================
BENCHMARK_MODE = "xsub120"  # Options: "xsub120" or "xset120"
DATASET_DIRECTORY = BENCHMARK_MODE

print(f"Initializing NTU-RGB+D 120 {BENCHMARK_MODE.upper()} splitter with corruption filtering...")

# Dataset directories
RAW_DATA_DIR = "data/nturgbd_skeletons_s018_to_s032" 
TRAIN_DIR = f"data/{DATASET_DIRECTORY}/train_skeletons"
VAL_DIR = f"data/{DATASET_DIRECTORY}/val_skeletons"
MISSING_SKELETONS_TXT = "NTU_RGBD120_samples_with_missing_skeletons.txt"
# ==========================================

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Skip missing skeletons
ignored_samples = set()
try:
    with open(MISSING_SKELETONS_TXT, 'r') as f:
        for line in f:
            clean_name = line.strip()
            if clean_name:
                ignored_samples.add(clean_name)
    print(f"Loaded {len(ignored_samples)} corrupted filenames to ignore.")
except FileNotFoundError:
    print(f"Warning: '{MISSING_SKELETONS_TXT}' not found. No files will be skipped.")

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

print(f"Sorting files using {BENCHMARK_MODE.upper()} rules and copying to destination...")

for filename in os.listdir(RAW_DATA_DIR):
    if not filename.endswith(".skeleton"):
        continue
        
    sample_name = filename.replace('.skeleton', '')
    
    if sample_name in ignored_samples:
        skipped_count += 1
        continue
        
    total_files += 1
    
    setup_id_str = filename.split('S')[1][:3]
    setup_id = int(setup_id_str)
    
    person_id_str = filename.split('P')[1][:3]
    person_id = int(person_id_str)
    
    source_path = os.path.join(RAW_DATA_DIR, filename)
    dest_path = None
    
    if BENCHMARK_MODE == "xsub120":
        if person_id in training_subjects:
            dest_path = os.path.join(TRAIN_DIR, filename)
            train_count += 1
        else:
            dest_path = os.path.join(VAL_DIR, filename)
            val_count += 1
            
    elif BENCHMARK_MODE == "xset120":
        if setup_id % 2 == 0:
            dest_path = os.path.join(TRAIN_DIR, filename)
            train_count += 1
        else:
            dest_path = os.path.join(VAL_DIR, filename)
            val_count += 1
            
    if dest_path:
        shutil.copy(source_path, dest_path)

print("-" * 50)
print(f"{BENCHMARK_MODE.upper()} data split and filtering complete")
print(f"Total Clean Skeletons Processed: {total_files}")
print(f"Corrupted Files Skipped:         {skipped_count}")
print(f"Copied to Training Set:          {train_count}")
print(f"Copied to Validation Set:        {val_count}")
print("-" * 50)
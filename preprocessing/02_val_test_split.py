import os
import shutil
import random

# ==========================================
# 1. BENCHMARK TOGGLE (CHANGE THIS!)
# ==========================================
BENCHMARK_MODE = "xsub120" # Options: ["xsub120", "xset120", "xsub", "xview"]
DATASET_DIRECTORY = BENCHMARK_MODE

print(f"Initializing Validation/Test Splitter for {BENCHMARK_MODE.upper()}...")

# ==========================================
# 2. DIRECTORY SETUP
# ==========================================
BASE_VAL_DIR = f'data/{DATASET_DIRECTORY}/val_skeletons'
BASE_TEST_DIR = f'data/{DATASET_DIRECTORY}/test_skeletons'

os.makedirs(BASE_TEST_DIR, exist_ok=True)

all_skeleton_files = [f for f in os.listdir(BASE_VAL_DIR) if f.endswith('.skeleton')]

# Random SPLIT: Move half of the validation files to the test directory
random.seed(42)
random.shuffle(all_skeleton_files)

split_idx = len(all_skeleton_files) // 2
files_to_move = all_skeleton_files[split_idx:]

print(f"Total validation files found: {len(all_skeleton_files)}")
print(f"Moving {len(files_to_move)} files to {BASE_TEST_DIR}...")

moved_count = 0
for skeleton_file in files_to_move:
    src_skeleton = os.path.join(BASE_VAL_DIR, skeleton_file)
    dst_skeleton = os.path.join(BASE_TEST_DIR, skeleton_file)
    shutil.move(src_skeleton, dst_skeleton)
    
    base_name = skeleton_file.replace('.skeleton', '')
    pt_file = base_name + '.pt'
    
    src_pt = os.path.join(BASE_VAL_DIR, pt_file)
    dst_pt = os.path.join(BASE_TEST_DIR, pt_file)
    
    if os.path.exists(src_pt):
        shutil.move(src_pt, dst_pt)
        
    moved_count += 1

print("-" * 50)
print(f"SPLIT COMPLETE")
print(f"Successfully moved {moved_count} file pairs to {BASE_TEST_DIR}.")
print("-" * 50)
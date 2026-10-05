from pathlib import Path

# Define the folder path
folder_path = Path(r"data/xview/test_skeletons/raw_text")

# Count only files (excludes directories)
file_count = sum(1 for item in folder_path.iterdir() if item.is_file())

print(f"Total files: {file_count}")
import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_single_skeleton(file_path):
    """
    Reads the raw .skeleton text file and outputs a numpy array.
    """
    try:
        with open(file_path, 'r') as f:
            datas = f.readlines()
        
        if not datas:
            return None

        nframe = int(datas[0].strip())
        skeleton_tensor = np.zeros((nframe, 2, 25, 3), dtype=np.float32)

        cursor = 0
        for frame in range(nframe):
            cursor += 1
            bodycount = int(datas[cursor].strip())

            if bodycount == 0:
                continue

            for body in range(bodycount):
                cursor += 2
                njoints_in_file = int(datas[cursor].strip())

                for joint in range(njoints_in_file):
                    cursor += 1
                    if body < 2:
                        joininfo = datas[cursor].strip().split()
                        skeleton_tensor[frame, body, joint, :] = [
                            float(joininfo[0]), float(joininfo[1]), float(joininfo[2])
                        ]
        
        return skeleton_tensor
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_file(args):
    """
    Worker function to process a single file so we can run them in parallel.
    """
    source_path, target_path = args
    
    if os.path.exists(target_path):
        return True
        
    parsed_data = parse_single_skeleton(source_path)
    
    if parsed_data is not None:
        tensor_data = torch.tensor(parsed_data, dtype=torch.float32)
        torch.save(tensor_data, target_path)
        return True
    return False

def convert_directory(data_dir):
    """
    Organizes loose files into 'raw_text', then converts them to 'binary_pt'.
    """
    print(f"\nProcessing directory: {data_dir}")

    raw_dir = os.path.join(data_dir, 'raw_text')
    binary_dir = os.path.join(data_dir, 'binary_pt')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)
    
    loose_files = [f for f in os.listdir(data_dir) if f.endswith('.skeleton')]
    if loose_files:
        print(f"Moving {len(loose_files)} loose files into 'raw_text' folder...")
        for f in loose_files:
            shutil.move(os.path.join(data_dir, f), os.path.join(raw_dir, f))
            
    file_list = [f for f in os.listdir(raw_dir) if f.endswith('.skeleton')]
    total_files = len(file_list)
    
    if total_files == 0:
        print(f"No .skeleton files found in {raw_dir}")
        return

    tasks = []
    for filename in file_list:
        source_path = os.path.join(raw_dir, filename)
        target_filename = filename.replace('.skeleton', '.pt')
        target_path = os.path.join(binary_dir, target_filename)
        tasks.append((source_path, target_path))

    print(f"⚙️  Converting {total_files} files to binary .pt format...")
    success_count = 0
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=total_files, desc="Converting"):
            if future.result():
                success_count += 1
                
    print(f"Successfully converted {success_count}/{total_files} files in {data_dir}")

if __name__ == '__main__':
    print("Starting NTU-RGBD 120 Data Organization & Binary PT Conversion...")
    
    # Modify the directories below if you have a different structure or additional datasets
    directories_to_process = [
        "data/xsub120/train_skeletons",
        "data/xsub120/val_skeletons",
        "data/xsub120/test_skeletons",
        "data/xset120/train_skeletons",
        "data/xset120/val_skeletons",
        "data/xset120/test_skeletons"
    ]
    
    for directory in directories_to_process:
        if os.path.exists(directory):
            convert_directory(directory)
        else:
            print(f"\nSkipping {directory} (Folder does not exist yet)")
            
    print("\nAll Organization and Conversions Complete!")
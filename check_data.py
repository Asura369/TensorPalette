import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm

def check_data(directory, delete=False):
    print(f"Checking data in {directory}...")
    if not delete:
        print("DRY RUN: No files will be deleted. Use --delete to actually remove files.")
    else:
        print("WARNING: Invalid files WILL be deleted.")
    
    valid_count = 0
    deleted_count = 0
    
    # Pre-calculate file list for tqdm
    all_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if not filename.startswith('.'):
                all_files.append(os.path.join(root, filename))
    
    # Walk through directory with progress bar
    for filepath in tqdm(all_files, desc="Scanning Images", unit="img"):
        filename = os.path.basename(filepath)
            
        try:
            with Image.open(filepath) as img:
                # Force loading to check for corruption
                img.load()
                
                # Check format and mode
                if img.format != 'JPEG':
                    msg = f"Format is {img.format}, not JPEG"
                    if delete:
                        tqdm.write(f"Deleting {filename}: {msg}")
                        os.remove(filepath)
                    else:
                        tqdm.write(f"[Would Delete] {filename}: {msg}")
                    deleted_count += 1
                    continue
                    
                if img.mode != 'RGB':
                    msg = f"Mode is {img.mode}, not RGB"
                    if delete:
                        tqdm.write(f"Deleting {filename}: {msg}")
                        os.remove(filepath)
                    else:
                        tqdm.write(f"[Would Delete] {filename}: {msg}")
                    deleted_count += 1
                    continue
                    
                valid_count += 1
                
        except (IOError, SyntaxError) as e:
            msg = f"{filename}: Corrupt or invalid image ({e})"
            if delete:
                tqdm.write(f"Deleting {msg}")
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except OSError as err:
                    tqdm.write(f"Error deleting {filepath}: {err}")
            else:
                tqdm.write(f"[Would Delete] {msg}")
                deleted_count += 1

    print("-" * 30)
    print(f"Scan complete.")
    if delete:
        print(f"Deleted files: {deleted_count}")
    else:
        print(f"Files flagged for deletion: {deleted_count}")
    print(f"Valid RGB JPG images found: {valid_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan directory for invalid images.")
    parser.add_argument("directory", nargs="?", default="training_content", help="Directory to scan (default: training_content)")
    parser.add_argument("--delete", action="store_true", help="Actually delete invalid files (default: False)")
    
    args = parser.parse_args()
        
    if not os.path.exists(args.directory):
        print(f"Directory '{args.directory}' not found.")
        sys.exit(1)
        
    check_data(args.directory, delete=args.delete)

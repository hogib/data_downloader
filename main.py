import re
from pathlib import Path

# Import the core function from your existing script (assuming it's named preprocess.py)
from src.process import run_balanced_preprocessing


def main():
    base_waveforms_dir = Path("data/batched_waveforms")
    noise_batch_dir = "data/batched_noise_waveforms"
    
    target_directories = [
        "window_post_60s",
        "window_post_100s",
        "window_post_120s",
        "window_post_200s",
        "window_pre_100s",
        "window_pre_200s"
    ]
    
    for dir_name in target_directories:
        match = re.search(r'(\d+)s$', dir_name)
        if not match:
            print(f"[WARN] Could not parse window seconds from {dir_name}. Skipping...")
            continue
            
        # Parse the nominal window size from the folder name
        nominal_window_secs = float(match.group(1))
        
        # Apply a 0.1 second buffer to account for trace clipping off-by-one errors
        actual_window_secs = nominal_window_secs - 0.1

        eq_dir = str(base_waveforms_dir / dir_name)
        output_dir = f"dataset_{dir_name}" 
        
        print(f"\n{'#'*60}")
        print(f"RUNNING PIPELINE FOR: {dir_name}")
        print(f"Targeting {actual_window_secs}s windows (Nominal {nominal_window_secs}s)")
        print(f"{'#'*60}\n") 

        if not Path(eq_dir).exists():
            print(f"[ERROR] Directory {eq_dir} does not exist. Skipping...")
            continue

        run_balanced_preprocessing(
            eq_dir=eq_dir,      
            noise_dir=noise_batch_dir,       
            output_dir=output_dir,        
            split_ratios=(0.70, 0.15, 0.15), 
            d=100,
            fs=100.0,                  
            window_seconds=actual_window_secs,        
            overlap=0.25,
        )

if __name__ == "__main__":
    main()

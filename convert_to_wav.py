import os
from pathlib import Path
from pydub import AudioSegment
import argparse
from tqdm import tqdm  # For progress bar

def convert_mp3_to_wav(src_path, dst_path, sample_rate=44100):
    """
    Convert a single MP3 file to WAV.
    """
    try:
        audio = AudioSegment.from_mp3(src_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)  # Mono for simplicity; change if needed
        audio.export(dst_path, format="wav")
        print(f"Converted: {src_path} -> {dst_path}")
        return True
    except Exception as e:
        print(f"Error converting {src_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, sample_rate=44100):
    """
    Recursively find and convert MP3 files in input_dir, saving to output_dir with same structure.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.mp3'):
                src_file = Path(root) / file
                rel_path = src_file.relative_to(input_path)
                dst_file = output_path / rel_path.with_suffix('.wav')
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                if convert_mp3_to_wav(src_file, dst_file, sample_rate):
                    converted_count += 1
    
    print(f"Processed {input_dir}: {converted_count} MP3 files converted.")
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="Convert MP3 files to WAV in specified directories.")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate for output WAV files (default: 44100)")
    args = parser.parse_args()
    
    # Updated to your actual directory paths (relative to /home/nikos/Desktop/apollo/Apollo)
    directories = [
        ("data/small", "data/small_wav"),
        ("data/small_degraded", "data/small_degraded_wav")
    ]
    
    total_converted = 0
    for input_dir, output_dir in directories:
        if not Path(input_dir).exists():
            print(f"Warning: Input directory {input_dir} does not exist. Skipping.")
            continue
        converted = process_directory(input_dir, output_dir, args.sample_rate)
        total_converted += converted
    
    print(f"\nConversion complete! Total files converted: {total_converted}")

if __name__ == "__main__":
    main()
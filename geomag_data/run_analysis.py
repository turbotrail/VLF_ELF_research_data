#!/usr/bin/env python3
"""
Geomagnetic Data Analysis Pipeline

This script runs a comprehensive analysis on BBC Microbit magnetometer data
to identify patterns related to geomagnetic storms and magnetic pole shifts.
"""

import os
import sys
import time
import subprocess
import json

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def run_script(script_path):
    """Run a Python script and handle any errors"""
    print_header(f"Running {script_path}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"\n✓ Successfully completed {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_path}: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error running {script_path}: {e}")
        return False

def create_output_directory():
    """Create the output directory if it doesn't exist"""
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def check_data_files():
    """Check if the required data files exist"""
    collection_file = 'collection_data.json'
    
    if not os.path.exists(collection_file):
        print(f"Error: {collection_file} not found in the current directory.")
        return False
        
    print(f"Found data file: {collection_file} ({format_file_size(collection_file)})")
    return True

def format_file_size(file_path):
    """Format file size in a human-readable format"""
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def check_requirements():
    """Check if all required packages are installed"""
    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        print(f"Warning: {requirements_file} not found. Package dependencies might not be installed.")
        return True
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], 
                     check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("All required packages are installed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing required packages: {e}")
        return False

def main():
    """Main function to run the entire analysis pipeline"""
    start_time = time.time()
    
    print_header("Geomagnetic Data Analysis")
    print("This pipeline will analyze BBC Microbit magnetometer data to identify")
    print("patterns related to geomagnetic storms and magnetic pole shifts.")
    
    # Check requirements and data files
    if not check_requirements() or not check_data_files():
        return
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Run analysis scripts in sequence
    scripts = [
        'analyze_magnetometer_data.py',
        'compare_with_geomagnetic_storms.py',
        'analyze_pole_shifts.py'
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: Script {script} not found. Skipping.")
            continue
            
        success = run_script(script)
        
        if not success:
            print(f"Warning: {script} did not complete successfully. Continuing with next script.")
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    minutes, seconds = divmod(execution_time, 60)
    
    print_header("Analysis Complete")
    print(f"Total execution time: {int(minutes)} minutes, {int(seconds)} seconds")
    print(f"Results are available in the {output_dir} directory")

if __name__ == "__main__":
    main() 
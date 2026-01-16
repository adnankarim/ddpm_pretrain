#!/usr/bin/env python3
"""
Recursively scan a directory and generate a CSV with file information.

Usage:
    python scan_files.py <data_directory> <output.csv>
"""

import os
import sys
import csv
from pathlib import Path


def scan_directory(data_dir, output_csv):
    """
    Recursively scan a directory and write file information to CSV.
    
    Args:
        data_dir: Path to directory to scan
        output_csv: Path to output CSV file
    """
    data_path = Path(data_dir).resolve()
    output_path = Path(output_csv)
    
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    if not data_path.is_dir():
        print(f"Error: Not a directory: {data_dir}")
        sys.exit(1)
    
    # Write CSV header
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['relative_path', 'filename', 'extension', 'size_bytes'])
        
        # Scan recursively
        for file_path in data_path.rglob('*'):
            if file_path.is_file():
                # Get relative path from data directory
                try:
                    rel_path = file_path.relative_to(data_path)
                except ValueError:
                    # If relative path calculation fails, use absolute
                    rel_path = file_path
                
                filename = file_path.name
                extension = file_path.suffix.lstrip('.') if file_path.suffix else ''
                size_bytes = file_path.stat().st_size
                
                writer.writerow([str(rel_path), filename, extension, size_bytes])
    
    print(f"CSV generated at: {output_path}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <data_directory> <output.csv>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_csv = sys.argv[2]
    
    scan_directory(data_dir, output_csv)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract permissions declared in AndroidManifest.xml from APK using androguard
"""

import os
import sys
import argparse
from os.path import basename
from androguard.core.bytecodes.apk import APK


def extract_permissions(apk_path: str) -> list:
    """
    Extract permissions declared in AndroidManifest.xml from APK
    
    Args:
        apk_path: APK file path
        
    Returns:
        list: List of permissions
    """
    try:
        if not os.path.exists(apk_path):
            print(f"Error: APK file does not exist: {apk_path}")
            return []
            
        # Load APK
        apk = APK(apk_path)
        
        # Get all declared permissions
        permissions = apk.get_permissions()
        
        return sorted(permissions)
        
    except Exception as e:
        print(f"Error extracting permissions: {e}")
        return []


def process_apk_directory(input_dir: str):
    """
    Batch process APK files in directory
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
    """
    # Create output directory
    _apk_dir = basename(input_dir)
    if len(_apk_dir) == 0:
        _apk_dir = input_dir.split("/")[-2]
    
    if input_dir.endswith('/'):
        output_dir = f'../../Raw/{input_dir.split("/")[-3]}/permission/{_apk_dir}/permissions'
    else:
        output_dir = f'../../Raw/{input_dir.split("/")[-2]}/permission/{_apk_dir}/permissions'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all APK files
    apk_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.apk'):
            apk_files.append(file)
    
    if not apk_files:
        print(f"No APK files found in directory {input_dir}")
        return
    
    print(f"Found {len(apk_files)} APK files")
    
    # Process each APK file
    for apk_file in apk_files:
        apk_path = os.path.join(input_dir, apk_file)
        output_file = os.path.splitext(apk_file)[0] + '_Permission.txt'
        output_path = os.path.join(output_dir, output_file)
        
        print(f"\nProcessing: {apk_file}")
        
        # Extract permissions
        permissions = extract_permissions(apk_path)
        
        if not permissions:
            print(f"  - No permissions found or extraction failed")
            continue
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for permission in permissions:
                    f.write(f"{permission}\n")
            print(f"  - Extracted {len(permissions)} permissions, saved to: {output_file}")
        except Exception as e:
            print(f"  - Error saving file: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch extract permissions declared in AndroidManifest.xml from APK files')
    parser.add_argument('-d', '--input_dir', help='Input directory containing APK files')
    
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    # Batch process APK files
    process_apk_directory(args.input_dir)



if __name__ == "__main__":
    main()

    
#!/usr/bin/env python3
"""
Convert MadMax dataset to KITTI format
Usage: python madmax_to_kitti.py --csv_path /path/to/poses.csv --image_dir /path/to/images --output_dir /path/to/kitti
"""

import pandas as pd
import numpy as np
import os
import shutil
import argparse
from pathlib import Path

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (radians) to rotation matrix"""
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def detect_csv_format(csv_path):
    """Detect the format of MadMax CSV file"""
    df = pd.read_csv(csv_path, nrows=5)
    columns = [col.lower() for col in df.columns]
    
    print("CSV columns found:", df.columns.tolist())
    
    # Common column patterns
    has_6dof = any('roll' in col or 'pitch' in col for col in columns)
    has_position = any(coord in col for coord in ['x', 'y', 'z'] for col in columns)
    has_orientation = any(orient in col for orient in ['yaw', 'heading'] for col in columns)
    
    if has_6dof:
        return "6dof"
    elif has_position and has_orientation:
        return "5dof"
    else:
        return "unknown"

def convert_madmax_poses(csv_path, output_dir):
    """Convert MadMax CSV poses to KITTI format"""
    
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Detect format
    format_type = detect_csv_format(csv_path)
    print(f"Detected format: {format_type}")
    
    if format_type == "unknown":
        print("ERROR: Could not detect CSV format. Please check column names.")
        print("Expected columns for 6DoF: timestamp, x, y, z, roll, pitch, yaw")
        print("Expected columns for 5DoF: timestamp, x, y, z, yaw/heading")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    poses = []
    
    # You might need to adjust these column names based on your actual CSV
    # Common variations:
    position_cols = None
    orientation_cols = None
    
    # Try to find position columns
    for x_col in ['x', 'X', 'pos_x', 'position_x']:
        if x_col in df.columns:
            position_cols = [x_col, x_col.replace('x', 'y'), x_col.replace('x', 'z')]
            break
    
    # Try to find orientation columns
    if format_type == "6dof":
        for roll_col in ['roll', 'Roll', 'rot_x', 'rotation_x']:
            if roll_col in df.columns:
                orientation_cols = [roll_col, 
                                  roll_col.replace('roll', 'pitch').replace('x', 'y'),
                                  roll_col.replace('roll', 'yaw').replace('x', 'z')]
                break
    else:  # 5dof
        for yaw_col in ['yaw', 'Yaw', 'heading', 'Heading', 'rot_z']:
            if yaw_col in df.columns:
                orientation_cols = [yaw_col]
                break
    
    if position_cols is None:
        print("ERROR: Could not find position columns (x, y, z)")
        return None
    
    print(f"Using position columns: {position_cols}")
    print(f"Using orientation columns: {orientation_cols}")
    
    for idx, row in df.iterrows():
        try:
            # Extract position
            x, y, z = row[position_cols[0]], row[position_cols[1]], row[position_cols[2]]
            
            if format_type == "6dof":
                # Extract full 6DoF
                roll, pitch, yaw = row[orientation_cols[0]], row[orientation_cols[1]], row[orientation_cols[2]]
                
                # Convert degrees to radians if necessary
                if abs(roll) > 10 or abs(pitch) > 10 or abs(yaw) > 10:  # Likely degrees
                    roll, pitch, yaw = np.radians([roll, pitch, yaw])
                
                R = euler_to_rotation_matrix(roll, pitch, yaw)
                
            else:  # 5dof
                # Only yaw rotation
                yaw = row[orientation_cols[0]]
                
                # Convert degrees to radians if necessary
                if abs(yaw) > 10:  # Likely degrees
                    yaw = np.radians(yaw)
                
                R = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]
                ])
            
            # Create 3x4 transformation matrix [R|t]
            T = np.hstack([R, np.array([[x], [y], [z]])])
            
            # Flatten to KITTI format (12 values)
            pose_line = T.flatten()
            poses.append(pose_line)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Write poses file
    poses_file = os.path.join(output_dir, "poses.txt")
    np.savetxt(poses_file, poses, fmt='%.6f')
    
    print(f"✅ Converted {len(poses)} poses to: {poses_file}")
    return poses_file

def organize_madmax_images(image_dir, output_dir):
    """Organize MadMax images to KITTI structure"""
    
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory not found: {image_dir}")
        return
    
    # Create KITTI image directory structure
    sequence_dir = os.path.join(output_dir, "sequences", "00")
    kitti_image_dir = os.path.join(sequence_dir, "image_1")  # Right camera (you mentioned right stereo)
    os.makedirs(kitti_image_dir, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    # Sort by name (assuming they're numbered)
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    
    # Copy and rename to KITTI format
    for idx, img_path in enumerate(image_files):
        kitti_name = f"{idx:06d}.png"
        dest_path = os.path.join(kitti_image_dir, kitti_name)
        
        # Copy file (or create symlink to save space)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        
        try:
            # Try symlink first (works on Windows 10+ with developer mode)
            if os.name == 'nt':  # Windows
                # Use copy on Windows for better compatibility
                shutil.copy2(str(img_path), dest_path)
            else:
                # Use symlink on Unix systems to save space
                os.symlink(str(img_path.absolute()), dest_path)
        except (OSError, NotImplementedError):
            # Fall back to copying
            shutil.copy2(str(img_path), dest_path)
    
    print(f"✅ Organized {len(image_files)} images in: {kitti_image_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert MadMax dataset to KITTI format')
    parser.add_argument('--csv_path', required=True, help='Path to MadMax CSV poses file')
    parser.add_argument('--image_dir', help='Path to MadMax images directory')
    parser.add_argument('--output_dir', required=True, help='Output directory for KITTI format')
    
    args = parser.parse_args()
    
    print("🔄 Converting MadMax dataset to KITTI format...")
    
    # Convert poses
    poses_file = convert_madmax_poses(args.csv_path, args.output_dir)
    
    if poses_file is None:
        print("❌ Failed to convert poses")
        return
    
    # Convert images if directory provided
    if args.image_dir:
        organize_madmax_images(args.image_dir, args.output_dir)
    
    print("✅ MadMax to KITTI conversion complete!")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Windows-compatible S3LI ROS bag to KITTI converter using rosbags library
This works on Windows without full ROS installation

Usage: python s3li_windows_converter.py --bag_path /path/to/dataset.bag --output_dir /path/to/kitti

Install dependencies:
pip install rosbags opencv-python numpy scipy
"""

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import struct

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    return R

def analyze_rosbag2(bag_path):
    """Analyze ROS2 bag to find topics"""
    print(f"🔍 Analyzing bag file: {bag_path}")
    
    try:
        with Reader(bag_path) as reader:
            # Get topic info
            topic_types = reader.get_all_topics_and_types()
            
            print("\n📋 Available topics:")
            pose_topics = []
            image_topics = []
            
            for topic_info in topic_types:
                topic = topic_info.name
                msg_type = topic_info.type
                
                print(f"  {topic}: {msg_type}")
                
                # Categorize topics
                if any(x in msg_type for x in ['Pose', 'Odometry', 'Transform']):
                    pose_topics.append(topic)
                elif 'Image' in msg_type:
                    image_topics.append(topic)
            
            print(f"\n🎯 Found {len(pose_topics)} pose topics: {pose_topics}")
            print(f"📷 Found {len(image_topics)} image topics: {image_topics}")
            
            return pose_topics, image_topics
            
    except Exception as e:
        print(f"❌ Error analyzing bag: {e}")
        return [], []

def extract_pose_from_ros_msg(msg):
    """Extract pose from ROS message"""
    try:
        # Handle different message types
        if hasattr(msg, 'pose'):
            if hasattr(msg.pose, 'pose'):  # PoseWithCovarianceStamped
                pose = msg.pose.pose
            else:  # PoseStamped
                pose = msg.pose
        elif hasattr(msg, 'transform'):  # TransformStamped
            t = msg.transform
            # Create pose-like object
            pose = type('obj', (object,), {})()
            pose.position = t.translation
            pose.orientation = t.rotation
        else:
            return None
        
        # Extract position
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        
        # Extract quaternion
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        
        # Convert to rotation matrix
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # Create 3x4 transformation matrix
        T = np.hstack([R, np.array([[x], [y], [z]])])
        
        # Return flattened for KITTI format
        return T.flatten()
        
    except Exception as e:
        print(f"Error extracting pose: {e}")
        return None

def convert_ros_image_to_cv(msg):
    """Convert ROS Image message to OpenCV format"""
    try:
        # Extract image data
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        data = msg.data
        
        # Convert based on encoding
        if encoding == 'mono8':
            # Grayscale 8-bit
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = img_array.reshape((height, width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding == 'bgr8':
            # BGR 8-bit
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = img_array.reshape((height, width, 3))
        elif encoding == 'rgb8':
            # RGB 8-bit
            img_array = np.frombuffer(data, dtype=np.uint8)
            img = img_array.reshape((height, width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            print(f"Unsupported encoding: {encoding}")
            return None
        
        return img
        
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

def convert_s3li_windows(bag_path, output_dir, pose_topic=None, image_topic=None, max_time_diff=0.1):
    """Convert S3LI bag to KITTI format on Windows"""
    
    # Analyze bag if topics not specified
    if pose_topic is None or image_topic is None:
        pose_topics, image_topics = analyze_rosbag2(bag_path)
        
        if not pose_topic and pose_topics:
            pose_topic = pose_topics[0]
            print(f"🎯 Using pose topic: {pose_topic}")
        
        if not image_topic and image_topics:
            image_topic = image_topics[0]
            print(f"📷 Using image topic: {image_topic}")
    
    if not pose_topic or not image_topic:
        print("❌ Could not find suitable pose and image topics")
        return False
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    sequence_dir = os.path.join(output_dir, "sequences", "00")
    image_dir = os.path.join(sequence_dir, "image_0")
    os.makedirs(image_dir, exist_ok=True)
    
    pose_data = []
    image_data = []
    
    print("🔄 Extracting data from bag...")
    
    try:
        with Reader(bag_path) as reader:
            # Read messages
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                msg_type = connection.msgtype
                
                # Deserialize message
                msg = deserialize_cdr(rawdata, msg_type)
                
                # Convert timestamp (nanoseconds to seconds)
                timestamp_sec = timestamp / 1e9
                
                if topic == pose_topic:
                    pose = extract_pose_from_ros_msg(msg)
                    if pose is not None:
                        pose_data.append((timestamp_sec, pose))
                
                elif topic == image_topic:
                    cv_image = convert_ros_image_to_cv(msg)
                    if cv_image is not None:
                        image_data.append((timestamp_sec, cv_image))
        
    except Exception as e:
        print(f"❌ Error reading bag: {e}")
        return False
    
    print(f"📊 Extracted {len(pose_data)} poses and {len(image_data)} images")
    
    # Synchronize data
    print("🔄 Synchronizing data...")
    synchronized_data = []
    
    for img_timestamp, image in image_data:
        best_pose = None
        best_time_diff = float('inf')
        
        for pose_timestamp, pose in pose_data:
            time_diff = abs(img_timestamp - pose_timestamp)
            if time_diff < best_time_diff and time_diff < max_time_diff:
                best_time_diff = time_diff
                best_pose = pose
        
        if best_pose is not None:
            synchronized_data.append((best_pose, image))
    
    print(f"✅ Synchronized {len(synchronized_data)} pose-image pairs")
    
    if not synchronized_data:
        print("❌ No synchronized data found")
        return False
    
    # Save data
    poses = []
    
    for idx, (pose, image) in enumerate(synchronized_data):
        # Save pose
        poses.append(pose)
        
        # Save image
        image_filename = f"{idx:06d}.png"
        image_path = os.path.join(image_dir, image_filename)
        cv2.imwrite(image_path, image)
        
        if idx % 100 == 0:
            print(f"  Processed {idx+1}/{len(synchronized_data)} frames...")
    
    # Save poses file
    poses_file = os.path.join(output_dir, "poses.txt")
    np.savetxt(poses_file, poses, fmt='%.6f')
    
    print(f"✅ Conversion complete!")
    print(f"  📁 Output: {output_dir}")
    print(f"  🔢 Frames: {len(synchronized_data)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert S3LI ROS bag to KITTI format (Windows)')
    parser.add_argument('--bag_path', required=True, help='Path to ROS bag file')
    parser.add_argument('--output_dir', required=True, help='Output directory for KITTI format')
    parser.add_argument('--pose_topic', help='ROS topic for poses')
    parser.add_argument('--image_topic', help='ROS topic for images')
    parser.add_argument('--max_time_diff', type=float, default=0.1, help='Max sync time diff (sec)')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze bag file')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_rosbag2(args.bag_path)
        return
    
    print("🔄 Converting S3LI bag to KITTI format (Windows)...")
    
    success = convert_s3li_windows(
        args.bag_path,
        args.output_dir, 
        args.pose_topic,
        args.image_topic,
        args.max_time_diff
    )
    
    if success:
        print("🎉 Conversion successful!")
    else:
        print("❌ Conversion failed")

if __name__ == "__main__":
    main()
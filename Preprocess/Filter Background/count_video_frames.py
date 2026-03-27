#!/usr/bin/env python3
from pathlib import Path
"""
Simple script to count the number of frames in a video file.
Supports various video formats including MP4, AVI, MOV, MKV, etc.
"""

import cv2
import os

# List of video files to analyze
# Add your video paths here
# VIDEO_PATHS = [
#     str(Path.home() / "Videos/trimmed_videos/a1_2025-08-14 12-57-35_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a2_2025-08-14 13-03-44_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a3_2025-08-14 13-11-09_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a4_2025-08-14 13-17-36_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a5_2025-08-14 13-22-37_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a6_2025-08-14 13-33-42_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a7_2025-08-14 13-43-18_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a8_2025-08-14 13-52-28_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a9_2025-08-14 14-04-15_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a10_2025-08-14 14-11-20_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a11_2025-08-14 14-15-26_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a12_2025-08-14 14-19-33_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a13_2025-08-14 14-26-28_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a14_2025-08-14 14-30-52_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a15_2025-08-14 14-34-45_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a16_2025-08-14 14-41-17_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a17_2025-08-14 14-45-40_trimmed.mkv"),
#     str(Path.home() / "Videos/trimmed_videos/a18_2025-08-14 14-49-10_trimmed.mkv"),
#     # Add more video paths as needed
#     # str(Path.home() / "Videos/another_video.mkv"),
# ]

VIDEO_PATHS = [
    # str(Path.home() / "Videos/2026-03-06 14-52-44_trimmed.mkv")
    r"D:\Videos\2025-09-12 18-14-10 (1.5mm)_trimmed_ds_2_split_cams\camera_0.mkv"
]

def count_video_frames(video_path):
    """
    Count the number of frames in a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        int: Number of frames in the video, or -1 if error
    """
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' does not exist.")
        return -1
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return -1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture
    cap.release()
    
    # Print results
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Frames: {frame_count:,}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    return frame_count

def main():
    """Main function to process all videos in the batch"""
    print("Video Frame Counter - Batch Processor")
    print("=" * 50)
    
    if not VIDEO_PATHS:
        print("No videos configured. Please add video paths to VIDEO_PATHS list.")
        return
    
    successful = 0
    failed = 0
    total_frames = 0
    video_info = []  # Store info for each video
    
    for i, video_path in enumerate(VIDEO_PATHS, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(VIDEO_PATHS)}: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            failed += 1
            continue
        
        frame_count = count_video_frames(video_path)
        
        if frame_count == -1:
            failed += 1
        else:
            successful += 1
            total_frames += frame_count
            video_info.append({
                'path': video_path,
                'filename': os.path.basename(video_path),
                'frames': frame_count
            })
    
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total videos: {len(VIDEO_PATHS)}")
    
    if successful > 0:
        print(f"Total frames across all videos: {total_frames:,}")
        
        # Calculate downsampling factors for each video individually
        target_frames = 200
        
        print(f"\n{'='*70}")
        print(f"DOWNsampling Analysis:")
        print(f"{'='*70}")
        print(f"Target frames per video: {target_frames}")
        print(f"Current total frames: {total_frames:,}")
        print(f"\nRecommended downsampling factors for each video:")
        print(f"{'='*70}")
        print(f"{'Video':<40} {'Frames':<8} {'Downsample':<12} {'Result':<8}")
        print(f"{'-'*70}")
        
        total_resulting_frames = 0
        for video in video_info:
            if video['frames'] > target_frames:
                # Calculate downsampling factor to get close to target_frames
                # This is how many frames to skip (keep every nth frame)
                downsampling_factor = video['frames'] // target_frames
                resulting_frames = video['frames'] // downsampling_factor
            else:
                downsampling_factor = 1  # No downsampling needed
                resulting_frames = video['frames']
            
            total_resulting_frames += resulting_frames
            
            # Truncate filename if too long
            display_name = video['filename'][:37] + "..." if len(video['filename']) > 40 else video['filename']
            
            print(f"{display_name:<40} {video['frames']:<8} {downsampling_factor:<12} {resulting_frames:<8}")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':<40} {total_frames:<8} {'':<12} {total_resulting_frames:<8}")
        print(f"\nTotal resulting frames: {total_resulting_frames}")
        print(f"Frame reduction: {total_frames - total_resulting_frames:,} frames removed")
        print(f"Size reduction: ~{((total_frames - total_resulting_frames) / total_frames * 100):.1f}%")
        
        # Generate VIDEO_CONFIGS format for easy copy-paste
        print(f"\n{'='*70}")
        print(f"VIDEO_CONFIGS for downsample_videos.py:")
        print(f"{'='*70}")
        print("VIDEO_CONFIGS = [")
        for video in video_info:
            if video['frames'] > target_frames:
                # Calculate downsampling factor to get close to target_frames
                downsampling_factor = video['frames'] // target_frames
            else:
                downsampling_factor = 1
            print(f"    (r\"{video['path']}\", {downsampling_factor}),")
        print("]")

if __name__ == "__main__":
    main()

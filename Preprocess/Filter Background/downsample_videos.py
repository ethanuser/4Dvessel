from pathlib import Path
import subprocess
import os

# Batch downsampling configuration
# Format: (input_video_path, downsampling_factor)
VIDEO_CONFIGS = [
    (str(Path.home() / "Videos/test.mkv"), 3),
]

def get_video_info(video_path):
    """Get video information including frame count and duration"""
    import json
    import cv2
    
    # Try ffprobe first
    result = subprocess.run([
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ], capture_output=True, text=True)
    
    data = {}
    if result.returncode == 0:
        data = json.loads(result.stdout)
    
    # Find video stream
    video_stream = None
    if 'streams' in data:
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
    
    # Get frame count and duration (with fallback to OpenCV)
    frame_count = 0
    duration = 0.0
    fps = 0.0
    width = 0
    height = 0
    codec = 'unknown'
    
    if video_stream:
        frame_count = int(video_stream.get('nb_frames', 0))
        duration = float(video_stream.get('duration', 0))
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 0
        else:
            fps = float(fps_str)
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        codec = video_stream.get('codec_name', 'unknown')
    
    # Use OpenCV as fallback for frame count/duration
    if frame_count == 0 or duration == 0:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            if frame_count == 0:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
            if duration == 0 and fps > 0:
                duration = frame_count / fps
            if width == 0:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if height == 0:
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
    if frame_count == 0:
        print(f"Warning: Could not determine frame count for {video_path}")
        
    return {
        'frame_count': frame_count,
        'duration': duration,
        'fps': fps,
        'width': width,
        'height': height,
        'codec': codec
    }

def downsample_video(input_path, downsampling_factor, output_path):
    """Downsample video by keeping only every nth frame"""
    
    print(f"\n{'='*60}")
    print(f"Downsampling: {os.path.basename(input_path)}")
    print(f"Downsampling factor: {downsampling_factor}")
    print(f"Output: {os.path.basename(output_path)}")
    print(f"{'='*60}")
    
    # Get input video info
    print("Getting input video information...")
    input_info = get_video_info(input_path)
    if not input_info:
        print(f"Failed to get info for {input_path}")
        return False
    
    print(f"Input video info:")
    print(f"  - Frames: {input_info['frame_count']}")
    print(f"  - Duration: {input_info['duration']:.2f}s")
    print(f"  - FPS: {input_info['fps']:.2f}")
    print(f"  - Resolution: {input_info['width']}x{input_info['height']}")
    print(f"  - Codec: {input_info['codec']}")
    
    # Calculate expected output frames
    expected_output_frames = input_info['frame_count'] // downsampling_factor
    expected_duration = input_info['duration'] / downsampling_factor
    expected_fps = input_info['fps'] / downsampling_factor
    
    print(f"\nExpected output:")
    print(f"  - Frames: {expected_output_frames}")
    print(f"  - Duration: {expected_duration:.2f}s")
    print(f"  - FPS: {expected_fps:.2f}")
    
    # Try multiple approaches for downsampling
    try:
        # Approach 1: Precise downsampling with H.264
        print(f"\nAttempting downsampling with H.264...")
        # We use select to keep every nth frame, and setpts to fix the timestamps
        # so the output video is smooth and actually has fewer frames.
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'select=not(mod(n\\,{downsampling_factor})),setpts=N/FRAME_RATE/TB',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-an',
            '-vsync', 'vfr',       # Variable Frame Rate: allows dropping frames completely
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"H.264 stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        print("Successfully downsampled with H.264")
        
    except subprocess.CalledProcessError as e:
        print(f"H.264 failed with error: {e}")
        print("Trying FFV1 as fallback...")
        try:
            # Approach 2: FFV1 with memory optimization (lossless but large)
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-vf', f'select=not(mod(n\\,{downsampling_factor}))',  # Select every nth frame
                '-c:v', 'ffv1',        # Lossless video codec
                '-level', '3',         # Lower level = less memory usage
                '-an',                 # No audio
                output_path
            ], check=True)
            print("Successfully downsampled with FFV1")
            
        except subprocess.CalledProcessError as e2:
            print(f"FFV1 also failed with error: {e2}")
            print("All downsampling methods failed!")
            return False
    
    # Verify output
    print(f"\nVerifying output file...")
    output_info = get_video_info(output_path)
    if output_info:
        print(f"Output video info:")
        print(f"  - Frames: {output_info['frame_count']}")
        print(f"  - Duration: {output_info['duration']:.2f}s")
        print(f"  - FPS: {output_info['fps']:.2f}")
        print(f"  - Resolution: {output_info['width']}x{output_info['height']}")
        print(f"  - Codec: {output_info['codec']}")
        
        # Calculate actual downsampling ratio
        actual_ratio = input_info['frame_count'] / output_info['frame_count'] if output_info['frame_count'] > 0 else 0
        print(f"\nActual downsampling ratio: {actual_ratio:.2f}x (target: {downsampling_factor}x)")
        
        # File size comparison
        input_size = os.path.getsize(input_path) / (1024*1024)  # MB
        output_size = os.path.getsize(output_path) / (1024*1024)  # MB
        print(f"File size: {input_size:.1f}MB -> {output_size:.1f}MB ({output_size/input_size:.2f}x)")
        
        return True
    else:
        print("Failed to verify output file")
        return False

def main():
    """Main function to process all videos in the batch"""
    print("Video Downsampling Batch Processor")
    print("=" * 50)
    
    if not VIDEO_CONFIGS:
        print("No videos configured. Please add video paths and downsampling factors to VIDEO_CONFIGS.")
        return
    
    successful = 0
    failed = 0
    
    for input_path, downsampling_factor in VIDEO_CONFIGS:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            failed += 1
            continue
        
        # Generate output path
        base_name = os.path.splitext(input_path)[0]
        extension = os.path.splitext(input_path)[1]
        output_path = f"{base_name}_ds_{downsampling_factor}{extension}"
        
        # Check if output already exists
        if os.path.exists(output_path):
            print(f"Output file already exists: {output_path}")
            response = input("Overwrite? (y/n): ").lower().strip()
            if response != 'y':
                print("Skipping...")
                continue
        
        # Process the video
        if downsample_video(input_path, downsampling_factor, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(VIDEO_CONFIGS)}")

if __name__ == "__main__":
    main()

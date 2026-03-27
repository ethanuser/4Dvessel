import os
import sys
import argparse

# Configuration - set these if you want to use constants instead of command line arguments
INPUT_VIDEO = None  # Set to video path, or None to use command line argument
OUTPUT_VIDEO = None  # Set to output path, or None to use command line argument
TARGET_FPS = None  # Set to target FPS, or None to use command line argument


def change_video_fps(input_path, output_path, target_fps):
    """
    Change the frame rate of a video file using ffmpeg.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        target_fps: Target frames per second (float or int)
    """
    try:
        import imageio
        import imageio_ffmpeg as ffmpeg
    except ImportError:
        print("Error: imageio and imageio-ffmpeg are required.")
        print("Install with: pip install 'imageio[ffmpeg]'")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found: {input_path}")
        sys.exit(1)
    
    # Get the ffmpeg executable path
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    
    # Use imageio to read the video and get original FPS
    try:
        reader = imageio.get_reader(input_path)
        original_fps = reader.get_meta_data().get('fps', None)
        reader.close()
        
        if original_fps:
            print(f"Original video FPS: {original_fps:.2f}")
        else:
            print("Warning: Could not determine original FPS")
    except Exception as e:
        print(f"Warning: Could not read original video metadata: {e}")
        original_fps = None
    
    print(f"Target FPS: {target_fps}")
    print(f"Converting video...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    
    # Use ffmpeg to change FPS
    # Method 1: Using imageio (simpler but may re-encode)
    try:
        reader = imageio.get_reader(input_path)
        writer = imageio.get_writer(output_path, fps=target_fps, codec='libx264', quality=8)
        
        frame_count = 0
        for frame in reader:
            writer.append_data(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...", end='\r')
        
        reader.close()
        writer.close()
        print(f"\nSuccess! Converted {frame_count} frames.")
        print(f"Output video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments or use constants"""
    
    # Use constants if set, otherwise use command line arguments
    if INPUT_VIDEO and OUTPUT_VIDEO and TARGET_FPS:
        input_video = INPUT_VIDEO
        output_video = OUTPUT_VIDEO
        target_fps = TARGET_FPS
    else:
        parser = argparse.ArgumentParser(
            description='Change the frame rate of a video file',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python change_video_fps.py input.mp4 output.mp4 30
  python change_video_fps.py video.mp4 video_60fps.mp4 60
            """
        )
        parser.add_argument('input_video', type=str, help='Path to input video file')
        parser.add_argument('output_video', type=str, help='Path to output video file')
        parser.add_argument('target_fps', type=float, help='Target frames per second')
        
        args = parser.parse_args()
        input_video = args.input_video
        output_video = args.output_video
        target_fps = args.target_fps
    
    # Validate target FPS
    if target_fps <= 0:
        print("Error: Target FPS must be greater than 0")
        sys.exit(1)
    
    # Change the video FPS
    change_video_fps(input_video, output_video, target_fps)


if __name__ == "__main__":
    main()


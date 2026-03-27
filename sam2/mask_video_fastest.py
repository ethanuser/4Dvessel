import sys, os, cv2, tempfile, shutil, subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QSlider, QLabel, QCheckBox, QLineEdit, QMessageBox)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QCursor, QKeySequence, QImage, QAction
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, QThread, pyqtSignal, QObject
import numpy as np
from PIL import Image
import torch
import decord

# Import SAM2 video predictor
from sam2.build_sam import build_sam2_video_predictor

# --- Main Application Class ---

class VideoSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Video Segmentation App")
        self.resize(1280, 720)
        
        # Video and segmentation state variables
        self.video_path = None
        self.video_frames = []       # QPixmap list for display
        self.video_frames_np = []    # Numpy arrays for processing
        self.total_frames = 0
        self.current_frame = 0
        self.frame_masks = {}        # Maps frame index to mask (numpy array)
        self.points_per_frame = {}   # Maps frame index -> list of [x,y] points (for segmentation)
        self.labels_per_frame = {}   # Maps frame index -> list of labels (1: positive, 0: negative)
        self.undo_stack_per_frame = {}  # Per-frame undo info
        self.key_frames = []         # Sorted list of frame indices where points have been added
        
        self.playing = False
        self.propagation_in_progress = False
        self.mode = "view"         # Can be "view", "add_points", "remove_points"
        
        # Persistent UI items for performance
        self.bg_pixmap_item = None
        self.mask_pixmap_item = None
        self.point_items = []
        
        self.temp_frames_dir = None  # Temporary directory to hold JPEG frames (for SAM2)
        self.frame_paths = []        # Paths to extracted frames
        self.frame_size = None       # (w, h)
        
        self.init_sam2()
        self.init_ui()
        self.setup_timer()
        self.print_user_guide()
    
    def init_sam2(self):
        # Initialize SAM2 video predictor
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        # sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cuda")
        self.inference_state = None
        print("Initialized SAM2 Video Predictor.")
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Graphics view for displaying video frame + overlays
        self.scene = QGraphicsScene(self)
        self.view = CustomGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        main_layout.addWidget(self.view)
        
        # --- Video Control Buttons ---
        video_control_layout = QHBoxLayout()
        self.select_video_button = QPushButton("Select Video")
        self.play_button = QPushButton("Play")
        self.prev_frame_button = QPushButton("Previous Frame")
        self.next_frame_button = QPushButton("Next Frame")
        self.prev_key_frame_button = QPushButton("Previous Key Frame")
        self.next_key_frame_button = QPushButton("Next Key Frame")
        video_control_layout.addWidget(self.select_video_button)
        video_control_layout.addWidget(self.play_button)
        video_control_layout.addWidget(self.prev_frame_button)
        video_control_layout.addWidget(self.next_frame_button)
        video_control_layout.addWidget(self.prev_key_frame_button)
        video_control_layout.addWidget(self.next_key_frame_button)
        main_layout.addLayout(video_control_layout)
        
        # Slider and frame label for playback
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)  # updated after video load
        self.slider.valueChanged.connect(self.slider_changed)
        self.frame_label = QLabel("Frame: 0/0")
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.frame_label)
        main_layout.addLayout(slider_layout)

        # Add FPS input field
        self.fps_input = QLineEdit()
        self.fps_input.setPlaceholderText("FPS (1–60)")
        self.fps_input.setFixedWidth(80)
        self.fps_input.setText("60")  # Default FPS
        
        # --- Segmentation Controls ---
        seg_control_layout = QHBoxLayout()
        self.add_points_button = QPushButton("Add Points")
        self.remove_points_button = QPushButton("Remove Points")
        self.remove_all_points_button = QPushButton("Remove All Points")
        self.propagate_button = QPushButton("Propagate")
        self.restart_propagation_button = QPushButton("Restart Propagation")
        self.undo_button = QPushButton("Undo")
        self.lossless_checkbox = QCheckBox("Lossless Saving")
        self.lossless_checkbox.setChecked(True)  # Default to checked
        seg_control_layout.addWidget(self.lossless_checkbox)
        seg_control_layout.addWidget(self.fps_input)
        self.save_masked_video_button = QPushButton("Save Masked Video")
        self.quit_button = QPushButton("Quit")
        seg_control_layout.addWidget(self.add_points_button)
        seg_control_layout.addWidget(self.remove_points_button)
        seg_control_layout.addWidget(self.remove_all_points_button)
        seg_control_layout.addWidget(self.propagate_button)
        seg_control_layout.addWidget(self.restart_propagation_button)
        seg_control_layout.addWidget(self.undo_button)
        seg_control_layout.addWidget(self.save_masked_video_button)
        seg_control_layout.addWidget(self.quit_button)
        main_layout.addLayout(seg_control_layout)
        
        # Connect button signals
        self.select_video_button.clicked.connect(self.select_video)
        self.play_button.clicked.connect(self.play_pause)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.prev_key_frame_button.clicked.connect(self.prev_key_frame)
        self.next_key_frame_button.clicked.connect(self.next_key_frame)
        self.add_points_button.clicked.connect(lambda: self.set_mode("add_points"))
        self.remove_points_button.clicked.connect(lambda: self.set_mode("remove_points"))
        self.remove_all_points_button.clicked.connect(self.remove_all_points)
        self.propagate_button.clicked.connect(self.propagate)
        self.restart_propagation_button.clicked.connect(self.restart_propagation)
        self.undo_button.clicked.connect(self.undo_action)
        self.save_masked_video_button.clicked.connect(self.save_masked_video)
        self.quit_button.clicked.connect(self.close)
        
        # Undo shortcut
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_action)
        self.addAction(undo_action)
    
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
    
    def set_mode(self, mode):
        if self.propagation_in_progress:
            return
        self.mode = mode
        if mode == "add_points":
            self.view.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif mode == "remove_points":
            self.view.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
    
    # --- Video Loading and Frame Extraction ---
    
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov, *.mkv)")
        if file_path:
            self.video_path = file_path
            self.load_video()
    
    def load_video(self):
        print("Start Loading Video")
        
        # Clean up previous video data to prevent memory leaks
        self.cleanup_previous_video()
        
        # Always extract frames to a temp JPEG folder for any video file
        if self.temp_frames_dir and os.path.exists(self.temp_frames_dir):
            shutil.rmtree(self.temp_frames_dir)
        self.temp_frames_dir = tempfile.mkdtemp(prefix="video_frames_")

        # Extract frames using OpenCV
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_filename = os.path.join(self.temp_frames_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1
        cap.release()
        self.total_frames = frame_idx

        # Store paths instead of full images to save memory
        self.frame_paths = []
        img_files = [f for f in os.listdir(self.temp_frames_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        img_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
        for img_file in img_files:
            self.frame_paths.append(os.path.join(self.temp_frames_dir, img_file))
        
        # Get frame size from first frame
        if self.frame_paths:
            probe = cv2.imread(self.frame_paths[0])
            self.frame_size = (probe.shape[1], probe.shape[0])

        self.current_frame = 0
        self.slider.setMaximum(self.total_frames - 1)
        self.frame_label.setText(f"Frame: 1/{self.total_frames}")
        self.update_frame_display()

        # Pass the temp JPEG folder to the predictor
        self.inference_state = self.predictor.init_state(video_path=self.temp_frames_dir)
        self.restart_propagation()
        print(f"Loaded video: {self.video_path} ({self.total_frames} frames)")

    def cleanup_previous_video(self):
        """Clean up previous video data to prevent memory leaks"""
        # Clear lists
        self.frame_paths.clear()
        
        # Clear segmentation data
        self.frame_masks.clear()
        self.points_per_frame.clear()
        self.labels_per_frame.clear()
        self.undo_stack_per_frame.clear()
        self.key_frames.clear()
        self.bg_pixmap_item = None
        self.mask_pixmap_item = None
        self.point_items = []
        self.scene.clear()
        
        # Reset inference state
        if hasattr(self, 'inference_state') and self.inference_state is not None:
            try:
                self.predictor.reset_state(self.inference_state)
            except:
                pass
            self.inference_state = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_frame_display(self):
        if not self.frame_paths:
            return

        # 1. Load background frame
        frame = cv2.cvtColor(cv2.imread(self.frame_paths[self.current_frame]), cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        if not self.bg_pixmap_item:
            self.bg_pixmap_item = self.scene.addPixmap(pixmap)
            self.bg_pixmap_item.setZValue(0)
            self.scene.setSceneRect(0, 0, w, h)
        else:
            self.bg_pixmap_item.setPixmap(pixmap)

        # 2. Update Mask Overlay
        if self.current_frame in self.frame_masks:
            mask = self.frame_masks[self.current_frame]
            tinted_mask = np.zeros((h, w, 4), dtype=np.uint8)
            tinted_mask[mask > 0] = [200, 255, 0, 160] # yellow
            mask_qimg = QImage(tinted_mask.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            mask_pix = QPixmap.fromImage(mask_qimg)
            
            if not self.mask_pixmap_item:
                self.mask_pixmap_item = self.scene.addPixmap(mask_pix)
                self.mask_pixmap_item.setZValue(1)
            else:
                self.mask_pixmap_item.setPixmap(mask_pix)
                self.mask_pixmap_item.show()
        else:
            if self.mask_pixmap_item:
                self.mask_pixmap_item.hide()

        # 3. Update Points
        # Clear old points
        for p_item in self.point_items:
            self.scene.removeItem(p_item)
        self.point_items.clear()
        
        if self.current_frame in self.points_per_frame:
            for (x, y), label in zip(self.points_per_frame[self.current_frame],
                                       self.labels_per_frame[self.current_frame]):
                color = QColor(0, 0, 0) if label == 1 else QColor(255, 0, 0)
                point = self.scene.addEllipse(x-8, y-8, 16, 16, QPen(color, 2), QColor(color))
                point.setZValue(2)
                self.point_items.append(point)
        
        self.slider.setValue(self.current_frame)
        self.frame_label.setText(f"Frame: {self.current_frame+1}/{self.total_frames}")
    
    
    # --- Mouse Interaction for Segmentation Points ---
    
    def add_point(self, event):
        scene_pos = self.view.mapToScene(event.pos())
        if self.scene.sceneRect().contains(scene_pos) and not self.propagation_in_progress:
            if self.mode == "add_points":
                pts = self.points_per_frame.get(self.current_frame, [])
                labs = self.labels_per_frame.get(self.current_frame, [])
                label = 1 if event.button() == Qt.MouseButton.LeftButton else 0
                pts.append([scene_pos.x(), scene_pos.y()])
                labs.append(label)
                self.points_per_frame[self.current_frame] = pts
                self.labels_per_frame[self.current_frame] = labs
                # Mark current frame as a key frame if not already present
                if self.current_frame not in self.key_frames:
                    self.key_frames.append(self.current_frame)
                    self.key_frames.sort()
                self.update_mask_for_current_frame()
                # Record undo action for this frame
                undo_stack = self.undo_stack_per_frame.get(self.current_frame, [])
                undo_stack.append(("add_point", len(pts)-1))
                self.undo_stack_per_frame[self.current_frame] = undo_stack
    
    def remove_points(self, rect):
        if self.propagation_in_progress:
            return
        if self.current_frame in self.points_per_frame:
            pts = self.points_per_frame[self.current_frame]
            labs = self.labels_per_frame[self.current_frame]
            removed_indices = [i for i, (x, y) in enumerate(pts) if rect.contains(QPointF(x, y))]
            if removed_indices:
                undo_stack = self.undo_stack_per_frame.get(self.current_frame, [])
                # Save copies for undo
                undo_stack.append(("remove_points", removed_indices, pts.copy(), labs.copy()))
                self.undo_stack_per_frame[self.current_frame] = undo_stack
                pts = [p for i, p in enumerate(pts) if i not in removed_indices]
                labs = [l for i, l in enumerate(labs) if i not in removed_indices]
                self.points_per_frame[self.current_frame] = pts
                self.labels_per_frame[self.current_frame] = labs
                self.update_mask_for_current_frame()
    
    def remove_all_points(self):
        if self.propagation_in_progress:
            return
        if self.current_frame in self.points_per_frame:
            pts = self.points_per_frame[self.current_frame]
            labs = self.labels_per_frame[self.current_frame]
            undo_stack = self.undo_stack_per_frame.get(self.current_frame, [])
            undo_stack.append(("remove_all", pts.copy(), labs.copy()))
            self.undo_stack_per_frame[self.current_frame] = undo_stack
            self.points_per_frame[self.current_frame] = []
            self.labels_per_frame[self.current_frame] = []
            if self.current_frame in self.frame_masks:
                del self.frame_masks[self.current_frame]
            self.update_frame_display()
    
    def undo_action(self):
        if self.propagation_in_progress:
            return
        if self.current_frame in self.undo_stack_per_frame and self.undo_stack_per_frame[self.current_frame]:
            action = self.undo_stack_per_frame[self.current_frame].pop()
            if action[0] == "add_point":
                idx = action[1]
                if self.current_frame in self.points_per_frame:
                    del self.points_per_frame[self.current_frame][idx]
                    del self.labels_per_frame[self.current_frame][idx]
            elif action[0] == "remove_points":
                _, old_pts, old_labs = action[1], action[2], action[3]
                self.points_per_frame[self.current_frame] = old_pts
                self.labels_per_frame[self.current_frame] = old_labs
            elif action[0] == "remove_all":
                old_pts, old_labs = action[1], action[2]
                self.points_per_frame[self.current_frame] = old_pts
                self.labels_per_frame[self.current_frame] = old_labs
            self.update_mask_for_current_frame()
    
    def update_mask_for_current_frame(self):
        if self.current_frame in self.points_per_frame and len(self.points_per_frame[self.current_frame]) > 0:
            pts = np.array(self.points_per_frame[self.current_frame])
            labs = np.array(self.labels_per_frame[self.current_frame])
            try:
                # Use SAM2 API for the current frame (using object id = 1)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame,
                    obj_id=1,
                    points=pts,
                    labels=labs
                )
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                self.frame_masks[self.current_frame] = mask
            except Exception as e:
                print("Error during mask update:", e)
        self.update_frame_display()
    
    # --- Frame Navigation and Slider ---
    
    def next_frame(self):
        if not self.propagation_in_progress:
            if self.current_frame == self.total_frames - 1:
                self.current_frame = 0
            else:
                self.current_frame += 1
            self.update_frame_display()
    
    def prev_frame(self):
        if not self.propagation_in_progress:
            if self.current_frame == 0:
                self.current_frame = self.total_frames - 1
            else:
                self.current_frame -= 1
            self.update_frame_display()

    
    def next_key_frame(self):
        if not self.key_frames or self.propagation_in_progress:
            return
        for kf in self.key_frames:
            if kf > self.current_frame:
                self.current_frame = kf
                self.update_frame_display()
                return
    
    def prev_key_frame(self):
        if not self.key_frames or self.propagation_in_progress:
            return
        for kf in reversed(self.key_frames):
            if kf < self.current_frame:
                self.current_frame = kf
                self.update_frame_display()
                return
    
    def slider_changed(self, value):
        if not self.propagation_in_progress:
            self.current_frame = value
            self.update_frame_display()
    
    def play_pause(self):
        if self.propagation_in_progress:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText("Pause")
            self.timer.start(1000 // int(self.fps_input.text()))
        else:
            self.play_button.setText("Play")
            self.timer.stop()
    
    # --- Propagation System ---
    
    def propagate(self):        
        if self.propagation_in_progress:
            return
        
        # Check if any points are selected across all frames
        any_points = any(len(pts) > 0 for pts in self.points_per_frame.values())
        if not any_points:
            QMessageBox.warning(self, "No Points Selected", 
                                "Please add at least one point before starting propagation.")
            return
        
        self.propagate_button.setText("Propagating")
        self.propagation_in_progress = True
        # Disable navigation and segmentation controls during propagation
        for btn in [self.play_button, self.next_frame_button, self.prev_frame_button,
                    self.next_key_frame_button, self.prev_key_frame_button,
                    self.add_points_button, self.remove_points_button, self.remove_all_points_button, self.undo_button, self.save_masked_video_button]:
            btn.setEnabled(False)
        
        # Run propagation in a separate thread so the UI remains responsive
        self.propagation_worker = PropagationWorker(
            self.predictor,
            self.inference_state,
            self.total_frames,
            self.points_per_frame,
            self.labels_per_frame
        )

        self.propagation_worker.progress.connect(self.handle_propagation_progress)
        self.propagation_worker.finished.connect(self.handle_propagation_finished)
        self.propagation_thread = QThread()
        self.propagation_worker.moveToThread(self.propagation_thread)
        self.propagation_thread.started.connect(self.propagation_worker.run)
        self.propagation_thread.start()
    
    def handle_propagation_progress(self, frame_idx, mask):
        mask = np.squeeze(mask)
        self.frame_masks[frame_idx] = mask
        self.current_frame = frame_idx
        
        # Optimize UI updates - update display every 5 frames or so during fast propagation
        if frame_idx % 5 == 0 or frame_idx == self.total_frames - 1:
            self.slider.setValue(frame_idx)
            self.frame_label.setText(f"Frame: {frame_idx+1}/{self.total_frames}")
            self.update_frame_display()
            QApplication.processEvents() # Keep UI responsive

    
    def handle_propagation_finished(self):
        self.propagation_in_progress = False

        # Clean up thread and worker properly
        if getattr(self, 'propagation_thread', None) is not None:
            self.propagation_thread.quit()
            self.propagation_thread.wait()
            self.propagation_thread = None
        if getattr(self, 'propagation_worker', None) is not None:
            self.propagation_worker = None

        # Re-enable controls
        for btn in [self.play_button, self.next_frame_button, self.prev_frame_button,
                    self.next_key_frame_button, self.prev_key_frame_button,
                    self.add_points_button, self.remove_points_button,
                    self.remove_all_points_button, self.undo_button, self.save_masked_video_button]:
            btn.setEnabled(True)

        self.propagate_button.setText("Propagate")
        print("Propagation finished.")


    def restart_propagation(self):
        if self.propagation_in_progress:
            print("Stopping propagation and resetting state...")
            if hasattr(self, 'propagation_worker') and self.propagation_worker is not None:
                self.propagation_worker.stop()
            if hasattr(self, 'propagation_thread') and self.propagation_thread is not None:
                self.propagation_thread.quit()
                self.propagation_thread.wait()
            self.propagation_in_progress = False
            self.propagation_worker = None
            self.propagation_thread = None
        self.frame_masks = {}
        self.points_per_frame = {}
        self.labels_per_frame = {}
        self.undo_stack_per_frame = {}
        self.key_frames = []
        self.current_frame = 0
        self.update_frame_display()
        print("Cleared all masks and points. Ready for fresh propagation.")

    
    # --- Saving Masked Video ---
    
    def save_masked_video(self):
        if not self.frame_masks:
            return
        
        try:
            fps = int(self.fps_input.text())
            if not (1 <= fps <= 60): raise ValueError
        except ValueError:
            fps = 60

        lossless = self.lossless_checkbox.isChecked()
        base, _ = os.path.splitext(self.video_path)
        output_path = base + "_masked" + (".mkv" if lossless else ".mp4")
        
        w, h = self.frame_size
        
        # FFmpeg command builder
        if lossless:
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}', 
                   '-framerate', str(fps), '-i', '-', '-c:v', 'ffv1', '-qscale:v', '0', output_path]
        else:
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}', 
                   '-framerate', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', 
                   '-pix_fmt', 'yuv420p', output_path]

        print(f"Starting optimized save to {output_path}...")
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        for idx in range(self.total_frames):
            # Load frame from disk lazily
            frame = cv2.cvtColor(cv2.imread(self.frame_paths[idx]), cv2.COLOR_BGR2RGB)
            
            if idx in self.frame_masks:
                mask = self.frame_masks[idx]
                # Apply mask (black background)
                frame[mask == 0] = 0
            
            # Write raw bytes to ffmpeg pipe
            process.stdin.write(frame.tobytes())
            
            if idx % 10 == 0:
                print(f"Exporting: {idx}/{self.total_frames}", end='\r')

        process.stdin.close()
        process.wait()
        print(f"\nSaved masked video: {output_path}")
    
    def closeEvent(self, event):
        # Clean up temp directories
        if self.temp_frames_dir and os.path.exists(self.temp_frames_dir):
            shutil.rmtree(self.temp_frames_dir)
        
        # Clean up video data
        self.cleanup_previous_video()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        event.accept()

    def print_user_guide(self):
        print("""
================ SAM2 Video Segmentation App - Detailed User Guide ================

OVERVIEW:
This application uses Meta's SAM2 (Segment Anything Model 2) for interactive video object 
segmentation. You can segment objects in videos by clicking points and having the AI 
automatically track and segment the object across all frames.

1. VIDEO LOADING:
   - Click 'Select Video' to load a video file (supports mp4, avi, mov, mkv formats)
   - The app will extract all frames to temporary JPEG files for processing
   - Loading progress is shown in the console
   - Video frames are displayed in the main view area

2. FRAME NAVIGATION:
   - 'Previous Frame'/'Next Frame': Step through frames one at a time
   - 'Previous Key Frame'/'Next Key Frame': Jump between frames where you've added points
   - Timeline Slider: Drag to quickly navigate to any frame
   - Frame Counter: Shows current frame number and total frames
   - 'Play'/'Pause': Auto-play video at specified FPS (default 60, adjustable 1-60)

3. INTERACTIVE SEGMENTATION:
   
   Adding Points:
   - Click 'Add Points' button to enter point-adding mode
   - Left-click: Add positive (foreground) points - tells the AI "this is the object"
   - Right-click: Add negative (background) points - tells the AI "this is NOT the object"
   - Points appear as colored circles: black for positive, red for negative
   - Each frame maintains its own set of points independently
   
   Removing Points:
   - Click 'Remove Points' button to enter removal mode
   - Drag a rectangle around points you want to remove
   - 'Remove All Points' clears all points on the current frame only
   - 'Undo' (Ctrl+Z) undoes the last point action on the current frame
   
   Visual Feedback:
   - Yellow mask overlay shows the current segmentation result
   - Masks update in real-time as you add/remove points
   - Only frames with points will show segmentation masks

4. PROPAGATION SYSTEM:
   
   How It Works:
   - After adding points on key frames, click 'Propagate' to segment the entire video
   - SAM2 uses your points to understand the object and tracks it across all frames
   - Propagation runs in the background - UI remains responsive
   - Progress is shown as masks appear frame by frame
   
   Controls:
   - 'Propagate': Start video-wide segmentation from current points
   - 'Restart Propagation': Clear all points, masks, and undo history for fresh start
   - During propagation, most controls are disabled to prevent conflicts
   
   Tips:
   - Add points on frames where the object is clearly visible
   - Use both positive and negative points for better accuracy
   - More points generally lead to better segmentation results

5. VIDEO EXPORT:
   
   Output Options:
   - 'Save Masked Video': Export the segmented result as a new video file
   - FPS Input: Set output video frame rate (1-60, default 60)
   - Lossless Checkbox: Choose between lossless (FFV1) or compressed (H.264) encoding
   
   File Formats:
   - Lossless: Saves as .mkv with FFV1 codec (larger files, perfect quality)
   - Compressed: Saves as .mp4 with H.264 codec (smaller files, high quality)
   - Output filename: [original_name]_masked.[extension]
   
   Process:
   - Frames without masks show the original video
   - Frames with masks show only the segmented object (black background)
   - Export uses FFmpeg for professional video encoding

6. ADVANCED FEATURES:
   
   Zoom and Pan:
   - Mouse wheel: Zoom in/out (hold Ctrl for finer control)
   - Drag: Pan around the video when zoomed in
   
   Keyboard Shortcuts:
   - Ctrl+Z: Undo last point action on current frame
   - Standard video player controls work during playback
   
   Memory Management:
   - App automatically cleans up temporary files
   - GPU memory is managed for optimal performance
   - Previous video data is cleared when loading new videos

7. TROUBLESHOOTING:
   
   Common Issues:
   - "Error opening video file": Check file format and permissions
   - "CUDA out of memory": Try restarting the app or using a smaller video
   - "Propagation failed": Ensure you have points added and try restarting propagation
   - Slow performance: Close other applications to free up GPU memory
   
   Best Practices:
   - Work with videos under 5 minutes for optimal performance
   - Add points on frames where the object is most distinct
   - Use negative points to exclude similar-looking objects
   - Save your work frequently by exporting masked videos

8. TECHNICAL NOTES:
   - Uses SAM2.1 Hierarchical Large model for best accuracy
   - Requires CUDA-compatible GPU for optimal performance
   - Temporary files are created in system temp directory
   - All processing happens in real-time with visual feedback

================================================================================
""")

# --- Propagation Worker (runs in a separate thread) ---

class PropagationWorker(QObject):
    progress = pyqtSignal(int, np.ndarray)
    finished = pyqtSignal()

    def __init__(self, predictor, inference_state, total_frames, points_per_frame, labels_per_frame):
        super().__init__()
        self.predictor = predictor
        self.inference_state = inference_state
        self.total_frames = total_frames
        self.points_per_frame = points_per_frame
        self.labels_per_frame = labels_per_frame
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    def run(self):
        # 1. Reset to clear any old prompt state
        self.predictor.reset_state(self.inference_state)

        # 2. Rebuild prompt state from all past points (across all frames)
        for frame_idx in sorted(self.points_per_frame.keys()):
            if self._should_stop:
                self.finished.emit()
                return
            pts = np.array(self.points_per_frame[frame_idx], dtype=np.float32)
            labs = np.array(self.labels_per_frame[frame_idx], dtype=np.int32)

            if len(pts) == 0:
                continue

            # Important: feed ALL points on this frame (not just new ones)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=pts,
                labels=labs
            )

        # 3. Propagate from this new set of prompts
        with torch.autocast("cuda"):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                if self._should_stop:
                    self.finished.emit()
                    return
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                self.progress.emit(out_frame_idx, mask)

        self.finished.emit()





# --- Custom GraphicsView to handle mouse events ---

class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene, parent_app):
        super().__init__(scene)
        self.parent_app = parent_app
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    
    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
    
    def mousePressEvent(self, event):
        if self.parent_app.mode == "add_points":
            self.parent_app.add_point(event)
        elif self.parent_app.mode == "remove_points":
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.parent_app.mode == "remove_points" and self.dragMode() == QGraphicsView.DragMode.RubberBandDrag:
            rect = self.rubberBandRect()
            self.parent_app.remove_points(self.mapToScene(rect).boundingRect())
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

# --- Main Execution ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSegmentationApp()
    window.show()
    sys.exit(app.exec())

#!/usr/bin/env python3
"""
Interactive tool for selecting and labeling regions in a 3D point cloud.
Features:
- Loads point cloud data
- Interactive 3D visualization (PyVista)
- separate GUI process (Tkinter) for controls to avoid macOS threading conflicts
- Select regions by clicking
- Manage regions: Label, Radius, Color
- Save/Load regions to JSON
"""

import sys
import os
import time
import json
import random
import numpy as np
# pyvista and tkinter are imported locally to avoid conflicts
from pathlib import Path
from scipy.spatial.distance import cdist
from multiprocessing import Process, Queue, Event
import queue # for queue.Empty exception

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.file_utils import resolve_path, get_project_root
from utils.data_utils import load_clustered_numpy

# Configuration
# DEFAULT_NPY_FILE = r"cluster_data/cluster_mesh_timeseries_synthetic_translate_real_params.npy"
DEFAULT_NPY_FILE = str(Path.home() / "Documents/Blender/Vessel-Renders/synthetic_translate/blender-data/mesh_data_single.npz")
DEFAULT_RADIUS = 0.2
RADIUS_MIN = 0.1
RADIUS_MAX_FACTOR = 0.1
GUI_WINDOW_SIZE = "500x700"
SCENE_SCALE_DEFAULT = 20.0
COLORED_POINT_SIZE = 10
UNCOLORED_POINT_SIZE = 5

# Protocol for messages
# GUI -> PyVista:
#   {"type": "update_region", "data": region_dict, "index": i}
#   {"type": "delete_region", "index": i}
#   {"type": "select_region", "index": i} (initiated from UI select)
#   {"type": "save", "data": [regions]}
#   {"type": "shutdown"}

# PyVista -> GUI:
#   {"type": "sync_regions", "regions": [dicts], "selected_index": i}
#   {"type": "shutdown"}

class Region:
    def __init__(self, center, radius, title, color):
        self.center = list(center) if isinstance(center, np.ndarray) else list(center)
        self.radius = float(radius)
        self.title = str(title)
        self.color = color

    def to_dict(self):
        return {
            "center": self.center,
            "radius": self.radius,
            "title": self.title,
            "color": self.color
        }
    
    @staticmethod
    def from_dict(d):
        return Region(d["center"], d["radius"], d["title"], d["color"])

# ==============================================================================
# GUI PROCESS
# ==============================================================================
def run_gui(to_pv_queue, from_pv_queue, scene_scale, initial_regions_data, json_path):
    import tkinter as tk
    from tkinter import ttk, colorchooser, messagebox

    root = tk.Tk()
    root.title("Region Selector Controls")
    root.geometry(GUI_WINDOW_SIZE)
    
    # State handling
    current_regions = [Region.from_dict(d) for d in initial_regions_data]
    selected_index = None
    is_updating = False  # Flag to prevent loops when updating UI
    
    # --- UI SETUP ---
    
    # 1. Regions List
    list_frame = ttk.LabelFrame(root, text="Regions", padding=10)
    list_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    columns = ("id", "title", "coords", "radius", "color")
    tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
    
    tree.heading("id", text="#")
    tree.column("id", width=30, stretch=False)
    tree.heading("title", text="Title")
    tree.column("title", width=100)
    tree.heading("coords", text="Center")
    tree.column("coords", width=120)
    tree.heading("radius", text="Radius")
    tree.column("radius", width=60)
    tree.heading("color", text="Color")
    tree.column("color", width=60)
    
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # 2. Edit Frame
    edit_frame = ttk.LabelFrame(root, text="Edit Selected Region", padding=10)
    edit_frame.pack(fill="x", padx=10, pady=5)
    
    # Title
    ttk.Label(edit_frame, text="Title:").grid(row=0, column=0, sticky="w")
    var_title = tk.StringVar()
    entry_title = ttk.Entry(edit_frame, textvariable=var_title)
    entry_title.grid(row=0, column=1, sticky="ew", padx=5)
    
    # Radius
    ttk.Label(edit_frame, text="Radius:").grid(row=1, column=0, sticky="w")
    var_radius = tk.DoubleVar(value=DEFAULT_RADIUS)
    scale_radius = ttk.Scale(edit_frame, from_=RADIUS_MIN, to=scene_scale * RADIUS_MAX_FACTOR, variable=var_radius)
    scale_radius.grid(row=1, column=1, sticky="ew", padx=5)
    lbl_radius_val = ttk.Label(edit_frame, text=f"{DEFAULT_RADIUS:.2f}")
    lbl_radius_val.grid(row=1, column=2, padx=5)

    
    # Color
    ttk.Label(edit_frame, text="Color:").grid(row=2, column=0, sticky="w")
    btn_color = tk.Button(edit_frame, text="Change Color", bg="white") # command set later
    btn_color.grid(row=2, column=1, sticky="w", padx=5)
    
    edit_frame.columnconfigure(1, weight=1)

    # 3. Actions & Options
    action_frame = ttk.Frame(root, padding=10)
    action_frame.pack(fill="x")
    
    # Visualization Options
    var_show_spheres = tk.BooleanVar(value=True)
    check_spheres = ttk.Checkbutton(action_frame, text="Show Spheres", variable=var_show_spheres)
    check_spheres.pack(anchor="w", pady=(0,2))

    var_show_labels = tk.BooleanVar(value=True)
    check_labels = ttk.Checkbutton(action_frame, text="Show Labels", variable=var_show_labels)
    check_labels.pack(anchor="w", pady=(0,5))

    btn_delete = ttk.Button(action_frame, text="Delete Selected")
    btn_delete.pack(side="left", padx=5)
    
    btn_save = ttk.Button(action_frame, text="Save JSON")
    btn_save.pack(side="right", padx=5)
    
    btn_deselect = ttk.Button(action_frame, text="Deselect All")
    btn_deselect.pack(side="left", padx=5)

    # --- FUNCTIONS ---
    
    def refresh_tree():
        for item in tree.get_children():
            tree.delete(item)
        for i, r in enumerate(current_regions):
            coords_str = f"[{r.center[0]:.2f}, {r.center[1]:.2f}, {r.center[2]:.2f}]"
            tree.insert("", "end", values=(i+1, r.title, coords_str, f"{r.radius:.2f}", r.color))
        
        if selected_index is not None and 0 <= selected_index < len(current_regions):
            item = tree.get_children()[selected_index]
            tree.selection_set(item)
            tree.see(item)
            fill_edit_fields(selected_index)
        else:
            clear_edit_fields()

    def fill_edit_fields(idx):
        nonlocal is_updating
        is_updating = True
        r = current_regions[idx]
        var_title.set(r.title)
        var_radius.set(r.radius)
        lbl_radius_val.config(text=f"{r.radius:.2f}")
        btn_color.config(bg=r.color)
        is_updating = False

    def clear_edit_fields():
        nonlocal is_updating
        is_updating = True
        var_title.set("")
        # Don't reset radius slider, keeps user preference
        lbl_radius_val.config(text="")
        btn_color.config(bg="lightgray")
        is_updating = False
    
    def on_tree_select(event):
        nonlocal selected_index
        items = tree.selection()
        if not items:
            return
        idx = tree.index(items[0])
        if idx != selected_index:
            selected_index = idx
            fill_edit_fields(idx)
            # Notify PV
            to_pv_queue.put({"type": "select_region", "index": idx})

    def on_property_change(*args):
        if is_updating: return
        
        # Always update local slider label
        lbl_radius_val.config(text=f"{var_radius.get():.2f}")
        
        if selected_index is None: return
        r = current_regions[selected_index]
        
        # Check what changed
        new_title = var_title.get()
        new_radius = var_radius.get()
        
        changed = False
        if new_title != r.title:
            r.title = new_title
            changed = True
        
        # Radius updates are now explicit via release, but we check here too for completeness
        # if other inputs trigger this. 
        # CAIT: Radius slider uses release event, so we trust `new_radius`.
        if abs(new_radius - r.radius) > 0.001:
            r.radius = new_radius
            changed = True
            
        if changed:
            # Update Tree visuals immediately
            item = tree.get_children()[selected_index]
            coords_str = f"[{r.center[0]:.2f}, {r.center[1]:.2f}, {r.center[2]:.2f}]"
            tree.item(item, values=(selected_index+1, r.title, coords_str, f"{r.radius:.2f}", r.color))
            # Send update
            to_pv_queue.put({"type": "update_region", "data": r.to_dict(), "index": selected_index})
            
    def on_radius_release(event):
        # Explicit handler for slider release to avoid spamming
        on_property_change()

    def on_color_click():
        if selected_index is None: return
        r = current_regions[selected_index]
        code = colorchooser.askcolor(title="Choose color", initialcolor=r.color)
        if code[1]:
            r.color = code[1]
            btn_color.config(bg=r.color)
            # Update Tree
            item = tree.get_children()[selected_index]
            tree.set(item, "color", r.color)
            # Send update
            to_pv_queue.put({"type": "update_region", "data": r.to_dict(), "index": selected_index})

    def on_toggle_spheres():
        to_pv_queue.put({"type": "toggle_spheres", "visible": var_show_spheres.get()})

    def on_toggle_labels():
        to_pv_queue.put({"type": "toggle_labels", "visible": var_show_labels.get()})

    def delete_region():
        nonlocal selected_index
        if selected_index is None: return
        
        # Send delete first
        to_pv_queue.put({"type": "delete_region", "index": selected_index})
        
        # Local update
        del current_regions[selected_index]
        selected_index = None
        refresh_tree()
        
    def deselect():
        nonlocal selected_index
        selected_index = None
        tree.selection_remove(tree.selection())
        clear_edit_fields()
        to_pv_queue.put({"type": "select_region", "index": -1})

    def save_json():
        # Just write to disk immediately from here for simplicity if path is known
        # But to be safe properly sync data: we use current_regions
        data_to_save = [r.to_dict() for r in current_regions]
        try:
            with open(json_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            messagebox.showinfo("Saved", f"Saved to {os.path.basename(json_path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_title_commit(event=None):
        on_property_change()
        # Return focus to main window or just un-focus entry to prevent stuck focus
        root.focus_set()

    # Bindings
    tree.bind("<<TreeviewSelect>>", on_tree_select)
    # var_title.trace("w", on_property_change) # Removed immediate trace
    entry_title.bind("<FocusOut>", on_title_commit)
    entry_title.bind("<Return>", on_title_commit)
    scale_radius.bind("<ButtonRelease-1>", on_radius_release) # Only update on release to fix lag
    # Also update label while dragging (without sending update)
    scale_radius.configure(command=lambda v: lbl_radius_val.config(text=f"{float(v):.2f}"))
    
    btn_color.configure(command=on_color_click)
    btn_delete.configure(command=delete_region)
    btn_save.configure(command=save_json)
    btn_deselect.configure(command=deselect)
    check_spheres.configure(command=on_toggle_spheres)
    check_labels.configure(command=on_toggle_labels)
    
    # Process Loop
    def check_queue():
        try:
            while True:
                msg = from_pv_queue.get_nowait()
                msg_type = msg.get("type")
                
                if msg_type == "shutdown":
                    root.quit()
                    return
                elif msg_type == "sync_regions":
                    nonlocal current_regions, selected_index
                    # Full Sync from PyVista
                    current_regions = [Region.from_dict(d) for d in msg["regions"]]
                    selected_index = msg["selected_index"] if msg["selected_index"] != -1 else None
                    refresh_tree()
                    
        except queue.Empty:
            pass
        
        root.after(50, check_queue)

    # Initial population
    refresh_tree()
    root.after(50, check_queue)
    root.protocol("WM_DELETE_WINDOW", lambda: to_pv_queue.put({"type": "shutdown"}))
    root.mainloop()


# ==============================================================================
# DATA LOADING (Shared)
# ==============================================================================

def load_session_data(npy_path_str):
    """
    Loads point cloud and region data without initializing PyVista/VTK.
    Returns: points (np.array), regions (list of dicts), scene_scale (float), json_path (Path)
    """
    project_root = get_project_root(__file__)
    resolved_path = resolve_path(npy_path_str, project_root)
    
    # Prioritize the central regions file in the cluster_data directory
    json_path = project_root / "cluster_data" / "vessel_regions.json"
    
    print(f"Loading data from {resolved_path}")
    if json_path.exists():
        print(f"Using regions from central storage: {json_path}")
    else:
        print(f"Note: Central regions file not found at {json_path}")
    
    # 1. Load Points
    points = np.array([])
    try:
        if resolved_path.suffix == '.npz':
            # Raw GT data (Single-File Efficient Storage)
            mesh_data = np.load(str(resolved_path))
            coords = mesh_data['coords']
            frame_numbers = mesh_data['frame_numbers']
            # Find first frame mask
            first_frame_mask = frame_numbers == 0
            points = coords[first_frame_mask]
            print(f"Loaded raw GT from .npz: {len(points)} points from frame 0.")
        else:
            # Clustered / Processed data
            data = load_clustered_numpy(str(resolved_path), project_root)
            points = data["cluster_positions"][0]
            print(f"Loaded clustered data from .npy: {len(points)} clusters.")
    except Exception as e:
        # Fallback for other formats
        try:
            d = np.load(str(resolved_path), allow_pickle=True)
            if isinstance(d, np.ndarray) and d.ndim==2: 
                points = d
            elif isinstance(d, dict) and 'coords' in d: 
                points = d['coords'][0] if d['coords'].ndim==3 else d['coords']
            print(f"Loaded via fallback: {len(points)} points.")
        except Exception as e2:
            print(f"Data load error: {e}")
            print(f"Fallback error: {e2}")

    # 2. Scene Scale
    scene_scale = SCENE_SCALE_DEFAULT
    if len(points) > 0:
        bbox = np.max(points, axis=0) - np.min(points, axis=0)
        scene_scale = np.linalg.norm(bbox)

    # 3. Load Regions
    regions_dicts = []
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                regions_dicts = json.load(f)
            print(f"Loaded {len(regions_dicts)} regions from JSON.")
        except Exception as e:
            print(f"JSON load error: {e}")
            
    return points, regions_dicts, scene_scale, json_path


# ==============================================================================
# VISUALIZATION PROCESS (MAIN)
# ==============================================================================

class PyVistaManager:
    def __init__(self, points, regions_dicts, json_path, to_gui_queue, from_gui_queue):
        import pyvista as pv
        
        self.points = points
        self.regions = [Region.from_dict(d) for d in regions_dicts]
        self.json_path = Path(json_path)
        
        # Queues
        self.send_q = to_gui_queue      # Write to this
        self.recv_q = from_gui_queue    # Read from this
        
        self.selected_index = None
        self.gui_last_known_radius = DEFAULT_RADIUS
        self.region_actor_names = [] # Track actors to remove efficiently
        self.show_spheres = True     # Visibility Toggle
        self.show_labels = True      # Visibility Toggle

        # Setup Plotter
        self.plotter = pv.Plotter(title="3D View - Click to Select")
        self.plotter.set_background("white")
        
        if len(self.points) > 0:
            self.plotter.add_mesh(
                pv.PolyData(self.points), 
                color="555555", # Lightened from "333333"
                point_size=UNCOLORED_POINT_SIZE,
                render_points_as_spheres=True,
                pickable=True,
                name="main_cloud",
                # lighting=False
            )
            
        self.plotter.enable_point_picking(callback=self.on_pick, show_message=False, use_picker=True, show_point=False)
        # self.plotter.add_timer_event(max_steps=2147483647, duration=100, callback=self.process_messages)
        
        # Initial draw
        self.refresh_visuals()

    def sync_to_gui(self):
        msg = {
            "type": "sync_regions",
            "regions": [r.to_dict() for r in self.regions],
            "selected_index": self.selected_index if self.selected_index is not None else -1
        }
        self.send_q.put(msg)

    def refresh_visuals(self):
        import pyvista as pv
        
        # Clear old actors efficiently
        for name in self.region_actor_names:
            self.plotter.remove_actor(name)
        self.region_actor_names.clear()

        # Redraw
        for i, r in enumerate(self.regions):
            # Sphere
            width = 5 if i == self.selected_index else 2
            sphere_name = f"region_sphere_{i}"
            if self.show_spheres:
                self.plotter.add_mesh(
                    pv.Sphere(radius=r.radius, center=r.center), 
                    color=r.color, 
                    style="wireframe", 
                    opacity=0.8, 
                    line_width=width,
                    pickable=False,
                    name=sphere_name
                )
                self.region_actor_names.append(sphere_name)
            
            # Points
            if len(self.points) > 0:
                dists = cdist([r.center], self.points).flatten()
                mask = dists <= r.radius
                pts = self.points[mask]
                if len(pts) > 0:
                     pts_name = f"region_pts_{i}"
                     self.plotter.add_mesh(
                        pv.PolyData(pts),
                        color=r.color,
                        point_size=COLORED_POINT_SIZE,
                        render_points_as_spheres=True,
                        pickable=False,
                        name=pts_name,
                        lighting=False
                    )
                     self.region_actor_names.append(pts_name)
            
            if self.show_labels:
                lbl_name = f"region_lbl_{i}"
                lbl_actor = self.plotter.add_point_labels(
                    [r.center], [r.title], 
                    point_size=1, 
                    point_color=r.color,
                    render_points_as_spheres=True,
                    font_size=40, 
                    text_color="black",
                    shape=None,
                    justification_horizontal='center',
                    justification_vertical='center',
                    always_visible=False, # Changed to False to allow styling via vtkLabeledDataMapper
                    name=lbl_name,
                )
                # Add white "stroke" effect using a white shadow
                if lbl_actor:
                    try:
                        # vtkLabeledDataMapper (used when always_visible=False) has GetLabelTextProperty
                        prop = lbl_actor.GetMapper().GetLabelTextProperty()
                        prop.SetShadow(True)
                        prop.SetShadowColor(1, 1, 1) # White shadow
                        prop.SetShadowOffset(1, -1)
                    except Exception as e:
                        print(f"Warning: Could not set label stroke: {e}")
                self.region_actor_names.append(lbl_name)

    def on_pick(self, *args):
        # PyVista/VTK picker can pass variable arguments depending on configuration
        # Usually args[0] is the picked mesh or point info
        point = None
        try:
            picking_info = args[0]
            if isinstance(picking_info, (list, tuple, np.ndarray)):
                point = np.array(picking_info).flatten()[:3]
            elif hasattr(picking_info, 'center'):
                point = np.array(picking_info.center)
            elif hasattr(picking_info, 'GetPickPosition'): # VTK Picker object
                point = np.array(picking_info.GetPickPosition())
            
            if point is None and len(args) > 1:
                # Sometimes the point is the second argument
                p = args[1]
                if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
                    point = np.array(p).flatten()
            
            if point is None: 
                # Fallback to last picked point if available via picker
                # This depends on what exactly passed
                print(f"DEBUG Pick Args: {args}")
                return

            if len(self.points) > 0:
                 dists = cdist([point], self.points)
                 point = self.points[np.argmin(dists)]
        except Exception as e: 
            print(f"Pick Error: {e}")
            return
        
        if self.selected_index is not None:
             self.regions[self.selected_index].center = list(point)
        else:
             new_color = "#%06x" % random.randint(0, 0xFFFFFF)
             title = f"Region_{len(self.regions)+1}"
             self.regions.append(Region(point, self.gui_last_known_radius, title, new_color))
             self.selected_index = len(self.regions)-1
             
        self.refresh_visuals()
        self.sync_to_gui()

    def process_messages(self, step_id):
        try:
            # Limit number of messages processed per frame to avoid freezing
            for _ in range(5): 
                try:
                    msg = self.recv_q.get_nowait()
                except queue.Empty:
                    break
                    
                mtype = msg.get("type")
                
                if mtype == "shutdown":
                    self.plotter.close()
                    sys.exit(0)
                
                elif mtype == "toggle_spheres":
                    self.show_spheres = msg["visible"]
                    self.refresh_visuals()

                elif mtype == "toggle_labels":
                    self.show_labels = msg["visible"]
                    self.refresh_visuals()

                elif mtype == "update_region":
                    idx = msg["index"]
                    data = msg["data"]
                    if 0 <= idx < len(self.regions):
                        self.regions[idx] = Region.from_dict(data)
                        self.gui_last_known_radius = self.regions[idx].radius
                        self.refresh_visuals()
                        
                elif mtype == "delete_region":
                    idx = msg["index"]
                    if 0 <= idx < len(self.regions):
                        del self.regions[idx]
                        self.selected_index = None
                        self.refresh_visuals()
                        
                elif mtype == "select_region":
                    idx = msg["index"]
                    self.selected_index = idx if idx != -1 else None
                    if self.selected_index is not None:
                         # Update local radius state so next click uses this radius
                         self.gui_last_known_radius = self.regions[self.selected_index].radius
                    self.refresh_visuals()
                    
        except Exception as e:
            print(f"Error in process_messages: {e}")

    def start(self):
        # Initial sync
        self.sync_to_gui()
        print("Starting PyVista (Main Process)...")
        
        # Manual Event Loop to prevent macOS freeze
        self.plotter.show(interactive_update=True)
        
        while True:
            self.process_messages(None)
            self.plotter.update()
            
            # Check if window was closed
            if self.plotter.render_window.GetGenericDisplayId() is None and self.plotter.render_window.GetWindowName() is None:
                 # Should probably break, but PyVista's closed check is subtle. 
                 # Usually update() helps keep it alive or errors if closed.
                 pass
            
            # Simple check if plotter is active (this is basic, might need refinement if 'q' is pressed)
            if hasattr(self.plotter, "_closed") and self.plotter._closed:
                break

            time.sleep(0.02)
        
        print("PyVista Window Closed")


def main():
    import multiprocessing
    # Ensure spawn method is used (default on macOS but good to be explicit)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. Parse Args
    npy_path_str = DEFAULT_NPY_FILE
    if len(sys.argv) >= 2:
        npy_path_str = sys.argv[1]

    # Queue Directions:
    # gui_in_q: Main -> GUI (written by Main, read by GUI)
    # gui_out_q: GUI -> Main (written by GUI, read by Main)
    gui_in_q = Queue()
    gui_out_q = Queue()
    
    # 2. Load data
    points, regions_dicts, scale, json_path = load_session_data(npy_path_str)
    
    # 3. Start GUI Process (Background/Child)
    # run_gui args: (to_pv, from_pv, ...) -> (gui_out_q, gui_in_q)
    print("Launching GUI Process...")
    gui_p = Process(
        target=run_gui, 
        args=(gui_out_q, gui_in_q, scale, regions_dicts, str(json_path))
    )
    gui_p.start()
    
    # 4. Run PyVista in Main Process
    # PyVistaManager args: (..., to_gui, from_gui) -> (gui_in_q, gui_out_q)
    try:
        manager = PyVistaManager(points, regions_dicts, json_path, gui_in_q, gui_out_q)
        manager.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main Process Error: {e}")
    finally:
        print("Shutting down...")
        gui_in_q.put({"type": "shutdown"})
        gui_p.join(timeout=2)
        if gui_p.is_alive():
            gui_p.terminate()

if __name__ == "__main__":
    main()
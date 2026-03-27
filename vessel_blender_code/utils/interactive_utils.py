"""
Utility functions for creating standardized interactive PyVista visualizations.

This module provides reusable components for frame-based interactive visualizations
with smooth updates, keyboard navigation, and consistent UI elements.
"""

import pyvista as pv
from typing import Callable, Optional, Dict, Any, List, Tuple
from typing_extensions import Protocol


class FrameUpdateCallback(Protocol):
    """Protocol for frame update callbacks."""
    def __call__(self, frame_list_idx: int, frame_idx: int, plotter: pv.Plotter) -> None:
        """Update the visualization for a specific frame.
        
        Args:
            frame_list_idx: Index in the sorted frame indices list
            frame_idx: The actual frame index
            plotter: The PyVista plotter instance
        """
        ...


class TextGenerator(Protocol):
    """Protocol for text generation callbacks."""
    def __call__(self, frame_idx: int, frame_data: Dict[str, Any]) -> str:
        """Generate text to display for a frame.
        
        Args:
            frame_idx: The frame index
            frame_data: Dictionary containing frame data
            
        Returns:
            Text string to display
        """
        ...


def create_interactive_visualization(
    plotter: pv.Plotter,
    frame_indices: List[int],
    frames_data: Dict[int, Dict[str, Any]],
    update_callback: FrameUpdateCallback,
    text_generator: TextGenerator,
    instructions: str,
    bounds: Optional[List[float]] = None,
    title: str = "Interactive Visualization",
    slider_title: str = "Frame",
    font: str = 'courier',
    text_font_size: int = 12,
    instructions_font_size: int = 9,
    slider_pointa: Tuple[float, float] = (0.05, 0.02),
    slider_pointb: Tuple[float, float] = (0.95, 0.02),
    add_axes: bool = False,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """
    Create a standardized interactive visualization with frame navigation.
    
    Args:
        plotter: PyVista plotter instance (should already be created)
        frame_indices: Sorted list of frame indices
        frames_data: Dictionary mapping frame indices to frame data
        update_callback: Function to call when frame changes
        text_generator: Function to generate text for each frame
        instructions: Instructions text to display in upper right
        bounds: Bounding box for camera [xmin, xmax, ymin, ymax, zmin, zmax]
        title: Window title
        slider_title: Title for the slider widget
        font: Font family for text ('courier', 'arial', etc.)
        text_font_size: Font size for frame info text
        instructions_font_size: Font size for instructions text
        slider_pointa: Start point for slider (normalized coordinates)
        slider_pointb: End point for slider (normalized coordinates)
        add_axes: Whether to add axes widget
        background_color: Background color of the plotter
        text_color: Color for text elements
    """
    plotter.set_background(background_color)
    plotter.title = title
    
    current_frame_idx = 0
    current_text_actor = None
    
    def update_frame_display(frame_list_idx: int) -> None:
        """Update the display to show a specific frame."""
        nonlocal current_text_actor, current_frame_idx
        
        if frame_list_idx < 0 or frame_list_idx >= len(frame_indices):
            return
        
        current_frame_idx = frame_list_idx
        frame_idx = frame_indices[frame_list_idx]
        
        # Call the user's update callback
        update_callback(frame_list_idx, frame_idx, plotter)
        
        # Update text section - add new text first, then remove old to avoid flashing
        frame_data = frames_data[frame_idx]
        text_content = text_generator(frame_idx, frame_data)
        
        old_text_actor = current_text_actor
        
        current_text_actor = plotter.add_text(
            text_content,
            font_size=text_font_size,
            position="upper_left",
            font=font,
            color=text_color,
        )
        
        # Now remove old text actor (after new one is already added)
        if old_text_actor is not None:
            try:
                plotter.remove_actor(old_text_actor)
            except Exception:
                pass
        
        plotter.render()
    
    def slider_callback(value: float) -> None:
        """Callback for slider widget."""
        frame_list_idx = int(value)
        update_frame_display(frame_list_idx)
    
    def navigate_frame(direction: str) -> None:
        """Navigate to next or previous frame."""
        nonlocal current_frame_idx
        
        if direction == "next":
            if current_frame_idx < len(frame_indices) - 1:
                new_idx = current_frame_idx + 1
                update_frame_display(new_idx)
                if len(plotter.slider_widgets) > 0:
                    plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)
        elif direction == "prev":
            if current_frame_idx > 0:
                new_idx = current_frame_idx - 1
                update_frame_display(new_idx)
                if len(plotter.slider_widgets) > 0:
                    plotter.slider_widgets[0].GetRepresentation().SetValue(new_idx)
        elif direction == "first":
            update_frame_display(0)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(0)
        elif direction == "last":
            last_idx = len(frame_indices) - 1
            update_frame_display(last_idx)
            if len(plotter.slider_widgets) > 0:
                plotter.slider_widgets[0].GetRepresentation().SetValue(last_idx)
    
    # Add axes if requested
    if add_axes:
        plotter.add_axes()
    
    # Add slider widget
    plotter.add_slider_widget(
        callback=slider_callback,
        rng=(0, len(frame_indices) - 1),
        value=0,
        title=slider_title,
        pointa=slider_pointa,
        pointb=slider_pointb,
        style="modern",
        tube_width=0.005,
        slider_width=0.02,
        interaction_event="always",
    )
    
    # Set font for slider
    if len(plotter.slider_widgets) > 0 and font == 'courier':
        try:
            slider_widget = plotter.slider_widgets[0]
            rep = slider_widget.GetRepresentation()
            if rep:
                title_prop = rep.GetTitleProperty()
                if title_prop:
                    title_prop.SetFontFamilyToCourier()
                label_prop = rep.GetLabelProperty()
                if label_prop:
                    label_prop.SetFontFamilyToCourier()
        except Exception:
            pass  # Font setting for slider is optional
    
    # Initialize first frame
    update_frame_display(0)
    
    # Set camera to fit all frames
    if bounds is not None:
        plotter.camera_position = "xy"
        plotter.reset_camera(bounds=bounds)
    
    # Add instructions text at top right
    plotter.add_text(
        instructions,
        font_size=instructions_font_size,
        position="upper_right",
        font=font,
        color=text_color,
    )
    
    # Keyboard events
    def on_next_frame() -> None:
        navigate_frame("next")
    
    def on_prev_frame() -> None:
        navigate_frame("prev")
    
    plotter.add_key_event("Right", on_next_frame)
    plotter.add_key_event("Left", on_prev_frame)
    plotter.add_key_event("d", on_next_frame)
    plotter.add_key_event("D", on_next_frame)
    plotter.add_key_event("a", on_prev_frame)
    plotter.add_key_event("A", on_prev_frame)
    
    print("\nInteractive visualization ready!")
    print("Use the slider or arrow keys (Left/Right) to navigate frames.")
    print("Close the window to exit.\n")


def create_default_text_generator(
    include_vertices: bool = True,
    include_time: bool = True,
    additional_fields: Optional[List[Tuple[str, Callable[[Dict], str]]]] = None,
) -> TextGenerator:
    """
    Create a default text generator function.
    
    Args:
        include_vertices: Whether to include vertex count
        include_time: Whether to include time
        additional_fields: List of (label, value_func) tuples for additional fields
        
    Returns:
        A text generator function
    """
    def text_generator(frame_idx: int, frame_data: Dict[str, Any]) -> str:
        parts = [f"Frame: {frame_idx}/{frame_data.get('max_frame', frame_idx)}"]
        
        if include_time and 'time' in frame_data:
            parts.append(f"Time: {frame_data['time']:.4f}")
        
        if include_vertices and 'num_vertices' in frame_data:
            parts.append(f"Original Points: {frame_data['num_vertices']:,}")
        
        if additional_fields:
            for label, value_func in additional_fields:
                try:
                    value = value_func(frame_data)
                    parts.append(f"{label}: {value}")
                except (KeyError, TypeError):
                    pass
        
        return " | ".join(parts)
    
    return text_generator


def create_default_instructions(
    visualization_description: str = "3D visualization",
    additional_info: Optional[List[str]] = None,
) -> str:
    """
    Create default instructions text.
    
    Args:
        visualization_description: Description of what's being visualized
        additional_info: List of additional info lines to include
        
    Returns:
        Instructions text string
    """
    instructions = (
        "Controls:\n"
        "  Slider (bottom): Drag to navigate frames\n"
        "  Arrow Keys: Left/Right to navigate frames\n"
        "  A/D keys: Previous/Next frame\n"
        "  Mouse: Rotate/zoom/pan view\n"
        "  Close window to exit"
    )
    
    if visualization_description or additional_info:
        instructions += "\n\nVisualization:\n"
        if visualization_description:
            instructions += f"  {visualization_description}\n"
        if additional_info:
            for info in additional_info:
                instructions += f"  {info}\n"
    
    return instructions

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import vessel_utils as vu

# Define the color map
cm = vu.get_colormap()

TITLE = 'Stress (MPa)'
VMIN = 0
VMAX = 0.15e6
TICKS = [0, 0.05e6, 0.10e6, 0.15e6]
TICK_LABELS = ["0.00", "0.05", "0.10", "0.15"]

# TITLE = 'Strain'
# VMIN = 0
# VMAX = 0.13
# TICKS = [0.00, 0.04, 0.08, 0.12]
# TICK_LABELS = ["0.00", "0.04", "0.08", "0.12"]

# TITLE = 'Displacement (mm)'
# VMIN = 0
# VMAX = 15
# TICKS = [0.0, 5.0, 10.0, 15.0]
# TICK_LABELS = [0.0, 5.0, 10.0, 15.0]

# TITLE = 'Displacement (mm)'
# VMIN = 0
# VMAX = 5
# TICKS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# TICK_LABELS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# Create a color bar
fig, ax = plt.subplots(figsize=(0.25, 6))

norm = plt.Normalize(vmin=VMIN, vmax=VMAX)

# Create a color bar with the same color map
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=ax, orientation='vertical')

# Fix white gap at the bottom/top by forcing tight limits and removing margins
ax.set_ylim(VMIN, VMAX)
ax.margins(0)

cb.set_label(TITLE, fontsize=24, fontweight='bold')
cb.set_ticks(TICKS)
cb.set_ticklabels(TICK_LABELS)
cb.ax.tick_params(labelsize=20, width=2.0, length=10)

# Ensure the border is robust and covers the gradient edges
cb.outline.set_linewidth(2.0)
cb.outline.set_zorder(10) # Render on top of everything

# Prepare output directory and filename
output_dir = Path(__file__).parent / 'generated_colorbars'
output_dir.mkdir(parents=True, exist_ok=True)

# Sanitize title for filename
safe_title = TITLE.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
filename = f"{safe_title}.png"
save_path = output_dir / filename

# Save the color bar as PNG and SVG
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.savefig(save_path.with_suffix('.svg'), bbox_inches='tight')
plt.close()

print(f"Color bar images saved to '{output_dir}':")
print(f"  - {save_path.name}")
print(f"  - {save_path.with_suffix('.svg').name}")

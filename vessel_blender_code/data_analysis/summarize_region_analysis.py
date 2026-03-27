#!/usr/bin/env python3
"""
Summarizes the Region Analysis CSVs into compact tables.
Finds the Max value over time for each Condition/Region and tabulates it.
Outputs .csv formats for both Displacement and Stress metrics.
"""

import csv
import sys
from pathlib import Path
import collections
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Set plot styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import vessel_utils as vu

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_RAW_GT = True

# ============================================================================
# STYLING & LAYOUT
# ============================================================================
FS_PANEL_LABEL = 40
FS_TABLE_TEXT = 14         # Main text in cells
COLOR_HEADER_BG = '#4a86e8'     # Blue for condition headers
COLOR_HEADER_TEXT = 'white'
COLOR_LABEL_BG = '#eeeeee'      # Light gray for Region/Method headers
COLOR_LABEL_TEXT = 'black'
COLOR_ROW_EVEN = "#ffffff"
COLOR_ROW_ODD = "#f9f9f9"
COLOR_METHOD_TEXT = "#444444"
COLOR_GRID_LINE = 'black'
LINE_WIDTH = 1.0
PANEL_LABEL_Y = 1.02
FIG_SIZE = (14, 5)
SUBPLOT_WSPACE = 0.03
SUBPLOT_TOP = 0.95

def parse_max_values(csv_path):
    """
    Parses the time-series CSV and computes the max value for each vessel/region.
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return None

    # data[Type][Condition][Region] = max_val
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    conditions = set()
    regions = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
        
        col_map = {}
        for idx, col_name in enumerate(header):
            if idx == 0: continue
            parts = col_name.split('_')
            # Handle possible underscores in region names or conditions
            # Usually format is Type_Cond_Region (e.g. GT_0.25_R1)
            if len(parts) >= 3:
                v_type = parts[0]
                cond = parts[1]
                region = "_".join(parts[2:])
                col_map[idx] = (v_type, cond, region)
                conditions.add(cond)
                regions.add(region)
        
        for row in reader:
            for idx, val_str in enumerate(row):
                if idx in col_map and val_str:
                    try:
                        val = float(val_str)
                        v_type, cond, region = col_map[idx]
                        if val > data[v_type][cond][region]:
                            data[v_type][cond][region] = val
                    except ValueError:
                        pass
    
    # Sort conditions. Try numeric, fallback to string length
    try:
        sorted_conds = sorted(list(conditions), key=lambda x: float(x.replace('mm','')))
    except:
        sorted_conds = sorted(list(conditions))
    sorted_regions = sorted(list(regions))
    
    return data, sorted_conds, sorted_regions

def generate_csv_rows(data, conditions, regions, output_rows):
    if not data: return
    output_rows.append(["Condition"] + conditions)
    for region in regions:
        for v_type in ["GT", "Ours"]:
            row = [f"{v_type} {region}"]
            for cond in conditions:
                val = data[v_type][cond][region]
                if val > 1000:
                    row.append(f"{val:.2f}")
                else:
                    row.append(f"{val:.6f}")
            output_rows.append(row)

def draw_table_on_ax(ax, data, conditions, regions, title, panel_label=None):
    """Draws table onto the plot Axis"""
    is_stress = "Stress" in title
    unit = "mm"
    scale = 1.0
    if is_stress:
        sample_val = data["GT"][conditions[0]][regions[0]]
        if sample_val > 1000:
            scale = 1e-6
            unit = "MPa"
        else:
            unit = "Pa"

    col_widths = [0.85, 0.95] + [0.8] * len(conditions)
    ncols_total = sum(col_widths)
    x_starts = [0]
    for w in col_widths:
        x_starts.append(x_starts[-1] + w)
    
    nrows = 1 + 2 * len(regions)
    header = ["Region", "Method"] + conditions

    ax.set_xlim(-0.02, ncols_total + 0.02)
    ax.set_ylim(-0.02, nrows + 0.02)
    ax.axis('off')

    # Header
    for j, label in enumerate(header):
        is_cond = j >= 2
        fc = COLOR_HEADER_BG if is_cond else COLOR_LABEL_BG
        tc = COLOR_HEADER_TEXT if is_cond else COLOR_LABEL_TEXT
        
        rect = patches.Rectangle((x_starts[j], nrows - 1), col_widths[j], 1, facecolor=fc, edgecolor='none')
        ax.add_patch(rect)
        ax.text(x_starts[j] + col_widths[j]/2, nrows - 0.5, label, ha='center', va='center', weight='bold', color=tc, fontsize=FS_TABLE_TEXT)

    # Data
    for i, r in enumerate(regions):
        bg = COLOR_ROW_EVEN if (i % 2 == 0) else COLOR_ROW_ODD
        y_low = nrows - 1 - (i * 2 + 2)
        ax.add_patch(patches.Rectangle((0, y_low), ncols_total, 2, facecolor=bg, edgecolor='none'))
        
        ax.text(col_widths[0]/2, y_low + 1.0, f"{r}", ha='center', va='center', weight='bold', fontsize=FS_TABLE_TEXT)
        
        for m_idx, m_name in enumerate(["GT", "Ours"]):
            curr_y = y_low + 1 if m_idx == 0 else y_low
            ax.text(x_starts[1] + col_widths[1]/2, curr_y + 0.5, m_name, ha='center', va='center', weight='bold', color=COLOR_METHOD_TEXT, fontsize=FS_TABLE_TEXT)
            
            for j, c in enumerate(conditions):
                val = data[m_name][c][r] * scale
                ax.text(x_starts[2+j] + col_widths[2+j]/2, curr_y + 0.5, f"{val:.3f}", ha='center', va='center', fontsize=FS_TABLE_TEXT)

    # Grid
    lc = COLOR_GRID_LINE
    for x_pos in x_starts:
        ax.vlines(x_pos, 0, nrows, colors=lc, linewidth=LINE_WIDTH, zorder=10, clip_on=False)
    for y in range(nrows + 1):
        if y == 0 or y >= nrows - 1 or (nrows - 1 - y) % 2 == 0:
            ax.hlines(y, 0, ncols_total, colors=lc, linewidth=LINE_WIDTH, zorder=10, clip_on=False)
        else:
            ax.hlines(y, x_starts[1], ncols_total, colors=lc, linewidth=LINE_WIDTH, zorder=10, clip_on=False)

    if panel_label:
        ax.text(0.0, PANEL_LABEL_Y, panel_label, transform=ax.transAxes, fontsize=FS_PANEL_LABEL, fontweight=1000, va='bottom', ha='left')

def generate_latex_table(data, conditions, regions, title, output_path, is_stress=False):
    unit = "mm"
    scale = 1.0
    if is_stress:
        sample_val = data["GT"][conditions[0]][regions[0]]
        if sample_val > 1000:
            scale = 1e-6
            unit = "MPa"
        else:
            unit = "Pa"

    n_conds = len(conditions)
    col_spec = "ll" + "c" * n_conds
    
    latex_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{a4paper, margin=0.5in, landscape}
\usepackage{siunitx}
\usepackage{multirow}

\begin{document}
\thispagestyle{empty}

\begin{table}[h]
\centering
\caption{""" + title + r"""}
\vspace{0.3cm}
\begin{tabular}{""" + col_spec + r"""}
\toprule
\textbf{Region} & \textbf{Method} & \multicolumn{""" + str(n_conds) + r"""}{c}{\textbf{Condition (""" + unit + r""")}} \\
\cmidrule(lr){3-""" + str(n_conds + 2) + r"""}
& & """ + " & ".join([fr"\textbf{{{c}}}" for c in conditions]) + r""" \\
\midrule
"""
    
    for i, r_name in enumerate(regions):
        for m_idx, m_name in enumerate(["GT", "Ours"]):
            row = fr"\multirow{{2}}{{*}}{{\textbf{{{r_name}}}}} & {m_name} " if m_idx == 0 else fr" & {m_name} "
            for c in conditions:
                val = data[m_name][c][r_name] * scale
                row += f" & {val:.3f}"
            row += r" \\"
            if m_idx == 1 and i < len(regions) - 1:
                row += r" \midrule"
            latex_content += row + "\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    try:
        import subprocess
        out_dir = output_path.parent
        subprocess.run(["pdflatex", "-interaction=nonstopmode", output_path.name], cwd=out_dir, check=True, capture_output=True)
        pdf_path = output_path.with_suffix('.pdf')
        png_path = output_path.with_suffix('.png')
        subprocess.run(["sips", "-s", "format", "png", pdf_path.name, "--out", png_path.name], cwd=out_dir, check=True, capture_output=True)
        return True
    except:
        return False

def generate_combined_latex_table(d_data, s_data, conditions, regions, title, output_path):
    """Unified LaTeX table for Disp and Stress"""
    s_unit = "Pa"
    s_scale = 1.0
    sample_val = s_data["GT"][conditions[0]][regions[0]]
    if sample_val > 1000:
        s_scale = 1e-6
        s_unit = "MPa"

    n = len(conditions)
    col_spec = "ll" + "c" * (2 * n)
    
    latex_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{a4paper, margin=0.5in, landscape}
\usepackage{siunitx}
\usepackage{multirow}

\begin{document}
\thispagestyle{empty}

\begin{table}[h]
\centering
\caption{""" + title + r"""}
\vspace{0.3cm}
\begin{tabular}{""" + col_spec + r"""}
\toprule
\multirow{2}{*}{\textbf{Region}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{""" + str(n) + r"""}{c}{\textbf{Displacement (mm)}} & \multicolumn{""" + str(n) + r"""}{c}{\textbf{Stress (""" + s_unit + r""")}} \\
\cmidrule(lr){3-""" + str(n + 2) + r"""} \cmidrule(lr){""" + str(n + 3) + "-" + str(2*n + 2) + r"""}
& & """ + " & ".join([fr"\textbf{{{c}}}" for c in conditions]) + " & " + " & ".join([fr"\textbf{{{c}}}" for c in conditions]) + r""" \\
\midrule
"""
    for i, r_name in enumerate(regions):
        for m_idx, m_name in enumerate(["GT", "Ours"]):
            row = (fr"\multirow{{2}}{{*}}{{\textbf{{{r_name}}}}} " if m_idx == 0 else " ") + f"& {m_name} "
            for c in conditions:
                row += f" & {d_data[m_name][c][r_name]:.3f}"
            for c in conditions:
                row += f" & {(s_data[m_name][c][r_name] * s_scale):.3f}"
            row += r" \\"
            if m_idx == 1 and i < len(regions) - 1:
                row += r" \midrule"
            latex_content += row + "\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
\end{document}
"""
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    try:
        import subprocess
        out_dir = output_path.parent
        subprocess.run(["pdflatex", "-interaction=nonstopmode", output_path.name], cwd=out_dir, check=True, capture_output=True)
        # Convert to PNG
        pdf_path = output_path.with_suffix('.pdf')
        png_path = output_path.with_suffix('.png')
        subprocess.run(["sips", "-s", "format", "png", pdf_path.name, "--out", png_path.name], cwd=out_dir, check=True, capture_output=True)
        return True
    except:
        return False

def main():
    script_dir = Path(__file__).parent
    analysis_dir = script_dir if script_dir.name == "data_analysis" else script_dir / "data_analysis"
    
    out_folder = "regional_summary_raw" if USE_RAW_GT else "regional_summary"
    output_dir = analysis_dir / out_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = "raw_" if USE_RAW_GT else ""
    tag_str = "RAW " if USE_RAW_GT else ""

    combos = [
        {
            "tag": "average",
            "main_title": f"Summary of {tag_str}Regional Analysis (Mean Values)",
            "disp": ("Max of Mean Displacement (mm)", f"region_analysis_{prefix}disp_average.csv"),
            "stress": ("Max of Mean Stress", f"region_analysis_{prefix}stress_average.csv"),
            "output": f"summary_{prefix}average_combined"
        },
        {
            "tag": "median",
            "main_title": f"Summary of {tag_str}Regional Analysis (Median Values)",
            "disp": ("Max of Median Displacement (mm)", f"region_analysis_{prefix}disp_median.csv"),
            "stress": ("Max of Median Stress", f"region_analysis_{prefix}stress_median.csv"),
            "output": f"summary_{prefix}median_combined"
        }
    ]
    
    csv_file = "summary_tables_raw.csv" if USE_RAW_GT else "summary_tables.csv"
    csv_out_path = analysis_dir / csv_file
    csv_rows = []
    
    for combo in combos:
        d_parsed = parse_max_values(analysis_dir / combo["disp"][1])
        s_parsed = parse_max_values(analysis_dir / combo["stress"][1])
        
        if d_parsed and s_parsed:
            d_data, d_conds, d_regs = d_parsed
            s_data, s_conds, s_regs = s_parsed
            
            # CSV Rows - Compatibility with statistical analysis parser
            csv_rows.append([combo["disp"][0]])
            generate_csv_rows(d_data, d_conds, d_regs, csv_rows)
            csv_rows.append([])
            csv_rows.append([combo["stress"][0]])
            generate_csv_rows(s_data, s_conds, s_regs, csv_rows)
            csv_rows.append([])

            # Combined PNG
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)
            draw_table_on_ax(ax1, d_data, d_conds, d_regs, combo["disp"][0], panel_label="A")
            draw_table_on_ax(ax2, s_data, s_conds, s_regs, combo["stress"][0], panel_label="B")
            plt.subplots_adjust(wspace=SUBPLOT_WSPACE, top=SUBPLOT_TOP)
            
            pout = output_dir / combo["output"]
            plt.savefig(pout.with_suffix(".png"), dpi=300, bbox_inches='tight')
            plt.savefig(pout.with_suffix(".pdf"), bbox_inches='tight')
            plt.close()
            print(f"Created {tag_str}combined table: {combo['output']}.png and .pdf in {output_dir.name}/")

            # LaTeX Indiv
            tag = combo["tag"]
            tex_d = output_dir / f"regional_summary_{tag}_disp.tex"
            tex_s = output_dir / f"regional_summary_{tag}_stress.tex"
            if generate_latex_table(d_data, d_conds, d_regs, combo["disp"][0], tex_d, is_stress=False):
                print(f"  ✓ Generated LaTeX Displacement table: {tex_d.name}")
            if generate_latex_table(s_data, s_conds, s_regs, combo["stress"][0], tex_s, is_stress=True):
                print(f"  ✓ Generated LaTeX Stress table: {tex_s.name}")

            # LaTeX Combined
            tex_path = output_dir / f"{combo['output']}.tex"
            if generate_combined_latex_table(d_data, s_data, d_conds, d_regs, combo["main_title"], tex_path):
                print(f"  🚀 Created {tag_str}LaTeX Table: {tex_path.name}")

    if csv_rows:
        with open(csv_out_path, 'w', newline='') as f:
            csv.writer(f).writerows(csv_rows)
        print(f"{tag_str}CSV summary saved to {csv_out_path}")

if __name__ == "__main__":
    main()

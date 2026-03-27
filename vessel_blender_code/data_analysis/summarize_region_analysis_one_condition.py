#!/usr/bin/env python3
"""
Summarizes the Region Analysis CSVs into compact tables specifically for ONE condition.
Transposes the layout: Regions as columns, Methods (GT vs Our Method) as rows.
Outputs .csv and .png formats.
"""

import csv
import sys
from pathlib import Path
import collections
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
import numpy as np

def parse_max_values(csv_path):
    """
    Parses the time-series CSV and computes the max value for each vessel/region.
    Returns: (data_dict, sorted_conditions, sorted_regions)
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return None

    # Structure: data[Type][Condition][Region] = max_val
    data = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(float)
        )
    )
    
    conditions = set()
    regions = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
        
        # Parse header to map index -> (Type, Condition, Region)
        col_map = {}
        for idx, col_name in enumerate(header):
            if idx == 0: continue # Skip Frame
            parts = col_name.split('_')
            # Format: Type_Cond_Region (e.g. GT_1mm_R1)
            if len(parts) >= 3:
                v_type = parts[0]
                cond = parts[1]
                region = "_".join(parts[2:]) # In case region name has underscores
                col_map[idx] = (v_type, cond, region)
                conditions.add(cond)
                regions.add(region)
        
        # Read frames and find max
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
    
    # Sort keys for consistent output
    sorted_conds = sorted(list(conditions), key=lambda x: float(x.replace('mm','')) if x.replace('mm','').replace('.','').isdigit() else x)
    
    # Custom sort for regions to ensure R1, R2, ..., R9
    def region_sort_key(r):
        import re
        match = re.search(r'R(\d+)', r)
        if match:
            return int(match.group(1))
        return r
    
    sorted_regions = sorted(list(regions), key=region_sort_key)
    
    return data, sorted_conds, sorted_regions

def generate_csv_rows_one_condition(data, conditions, regions, output_rows):
    if not data or not conditions: return
    
    # We take the first condition if multiple exist, but script is intended for one
    cond = conditions[0]
    
    # Header Row: Region, R1, R2, ...
    output_rows.append(["Region"] + regions)
    
    # Row for Ground Truth
    gt_row = ["Ground Truth"]
    for region in regions:
        val = data["GT"][cond][region]
        if val > 1000:
            gt_row.append(f"{val:.2f}")
        else:
            gt_row.append(f"{val:.4f}")
    output_rows.append(gt_row)
    
    # Row for Our Method
    ours_row = ["Our Method"]
    for region in regions:
        val = data["Ours"][cond][region]
        if val > 1000:
            ours_row.append(f"{val:.2f}")
        else:
            ours_row.append(f"{val:.4f}")
    output_rows.append(ours_row)

def generate_png_table_one_condition(data, conditions, regions, title, output_path):
    """
    Creates a professional PNG table of the data for a single condition.
    """
    if not conditions: return
    cond = conditions[0]
    
    # Determine if we should scale stress to MPa
    is_stress = "Stress" in title
    unit = "mm"
    scale = 1.0
    if is_stress:
        # Check if values are large
        sample_val = data["GT"][cond][regions[0]]
        if sample_val > 1000:
            scale = 1e-6
            unit = "MPa"
        else:
            unit = "Pa"

    # Prepare data for table
    # Rows: Ground Truth, Our Method
    # Columns: R1, R2, ...
    table_data = []
    
    # GT Row
    gt_row = []
    for region in regions:
        val = data["GT"][cond][region] * scale
        gt_row.append(f"{val:.3f}")
    table_data.append(gt_row)
        
    # Ours Row
    ours_row = []
    for region in regions:
        val = data["Ours"][cond][region] * scale
        ours_row.append(f"{val:.3f}")
    table_data.append(ours_row)

    row_labels = ["Ground Truth", "Our Method"]
    col_labels = regions

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')
    ax.axis('tight')

    # Create table
    the_table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        rowColours=["#f2f2f2", "#ffffff"],
        colColours=["#cfe2f3"] * len(regions)
    )

    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 2.0)  # Scaled for better readability

    # Style cells
    cells = the_table.get_celld()
    
    # Corner cell (0, -1)
    if (0, -1) not in cells:
        try:
            w = cells[(1, -1)].get_width()
            h = cells[(0, 0)].get_height()
            the_table.add_cell(0, -1, w, h, text="Region", loc='center')
        except KeyError:
            pass

    for (row, col), cell in cells.items():
        # Column headers (Row 0)
        if row == 0 and col >= 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4a86e8')
        # Row headers (Column -1)
        elif col == -1 and row > 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#eeeeee')
        # Corner cell (Row 0, Column -1)
        elif row == 0 and col == -1:
            cell.get_text().set_text("Region")
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4a86e8')
        
    plt.title(f"{title} ({unit}) - {cond}", fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    script_dir = Path(__file__).parent
    # Check if we are running from data_analysis or project root
    if script_dir.name == "data_analysis":
        analysis_dir = script_dir
    else:
        analysis_dir = script_dir / "data_analysis"
    
    metrics = [
        ("MAX of AVERAGE Displacement", "region_analysis_disp_average.csv"),
        ("MAX of MEDIAN Displacement", "region_analysis_disp_median.csv"),
        ("MAX of AVERAGE Stress", "region_analysis_stress_average.csv"),
        ("MAX of MEDIAN Stress", "region_analysis_stress_median.csv"),
    ]
    
    csv_out_path = analysis_dir / "summary_tables_one_condition.csv"
    csv_rows = []
    
    for title, filename in metrics:
        csv_path = analysis_dir / filename
        if not csv_path.exists():
            continue
            
        print(f"Reading {csv_path.name}...")
        parsed = parse_max_values(csv_path)
        
        if parsed:
            data, conds, regs = parsed
            
            if not conds:
                continue

            # 1. Update CSV Rows
            csv_rows.append([title])
            generate_csv_rows_one_condition(data, conds, regs, csv_rows)
            csv_rows.append([]) # Blank line
            
            # 2. Generate PNG Table
            safe_title = title.lower().replace(" ", "_").replace("max_of_", "summary_")
            png_path = analysis_dir / f"{safe_title}_one_condition.png"
            generate_png_table_one_condition(data, conds, regs, title, png_path)
            print(f"Created table: {png_path.name}")
        
    if not csv_rows:
        print("No data found to summarize.")
        return

    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"CSV summary saved to {csv_out_path}")

if __name__ == "__main__":
    main()
"""
Statistical comparison of displacement measurements across vessel conditions.

Performs separate One-Way ANOVAs for each anatomical location,
comparing the three experimental conditions (Cervical, Terminal, Cavernous).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, shapiro, levene
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(project_root, 'data', 'processed', 'max_region_analysis', 'max_displacements_regions_pivot.csv')

def load_displacement_data(csv_path):
    """
    Load and structure the displacement data from CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary with structure: {location: {condition: [values]}}
    """
    df = pd.read_csv(csv_path)
    
    # The data columns (excluding Trial and Regions columns)
    data_columns = df.columns[2:-1]  # Exclude 'Trial', 'Regions', and 'Average'
    
    # Forward fill the Trial column (condition names appear only once per group)
    df['Trial'] = df['Trial'].fillna(method='ffill')
    
    # Clean up the Regions column - map variations to standard names
    region_mapping = {
        'M1 Convex': 'M1 Convex Region',
        'M1 Convex Region': 'M1 Convex Region',
        'M1/M2 Bift.': 'M1/M2 Bifurcation',
        'M1/M2 Bifurcation': 'M1/M2 Bifurcation',
        'M1M2 Bifurcation': 'M1/M2 Bifurcation',
        'M2 Caudal I': 'M2 Caudal Region',
        'M2 Caudal Region': 'M2 Caudal Region',
        'M2 Caudal': 'M2 Caudal Region'
    }
    
    df['Regions'] = df['Regions'].map(lambda x: region_mapping.get(x, x) if pd.notna(x) else x)
    
    # Structure the data
    data = {}
    
    # Deriving locations from the available mapped region names in the data
    # (limited to the three we care about to keep analysis focused)
    valid_locations = ['M1 Convex Region', 'M1/M2 Bifurcation', 'M2 Caudal Region']
    found_locations = df['Regions'].unique()
    locations = [loc for loc in valid_locations if loc in found_locations]
    
    for location in locations:
        data[location] = {}
        
        # Get rows for each condition at this location
        cervical_row = df[(df['Trial'] == 'Cervical') & (df['Regions'] == location)]
        terminal_row = df[(df['Trial'] == 'Terminal') & (df['Regions'] == location)]
        cavernous_row = df[(df['Trial'] == 'Cavernous') & (df['Regions'] == location)]
        
        # Extract values
        if not cervical_row.empty:
            data[location]['Cervical'] = cervical_row[data_columns].values[0].astype(float)
        if not terminal_row.empty:
            data[location]['Terminal'] = terminal_row[data_columns].values[0].astype(float)
        if not cavernous_row.empty:
            data[location]['Cavernous'] = cavernous_row[data_columns].values[0].astype(float)
    
    return data


def perform_anova_for_location(location_name, cervical, terminal, cavernous):
    """
    Perform comprehensive statistical analysis for a single location.
    
    Args:
        location_name: Name of the anatomical location
        cervical: Array of cervical measurements
        terminal: Array of terminal measurements
        cavernous: Array of cavernous measurements
        
    Returns:
        Dictionary containing all statistical results
    """
    results = {}
    
    print("\n" + "="*80)
    print(f"LOCATION: {location_name}")
    print("="*80)
    
    # Basic descriptive statistics
    print("\nDESCRIPTIVE STATISTICS")
    print("-"*80)
    
    conditions = {
        'Cervical': cervical,
        'Terminal': terminal,
        'Cavernous': cavernous
    }
    
    for name, data in conditions.items():
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        sem = stats.sem(data)
        median = np.median(data)
        
        print(f"\n{name}:")
        print(f"  N = {len(data)}")
        print(f"  Mean ± SD: {mean:.4f} ± {std:.4f} mm")
        print(f"  SEM: {sem:.4f} mm")
        print(f"  Median: {median:.4f} mm")
        print(f"  Range: [{np.min(data):.4f}, {np.max(data):.4f}] mm")
        
        results[f'{name.lower()}_stats'] = {
            'mean': mean,
            'std': std,
            'sem': sem,
            'median': median,
            'min': np.min(data),
            'max': np.max(data),
            'n': len(data)
        }
    
    # Test for normality (Shapiro-Wilk test)
    print("\n" + "-"*80)
    print("NORMALITY TEST (Shapiro-Wilk)")
    print("-"*80)
    
    for name, data in conditions.items():
        stat, p_value = shapiro(data)
        print(f"{name}: W = {stat:.4f}, p = {p_value:.4f}", end="")
        if p_value > 0.05:
            print(" (Normal)")
        else:
            print(" (Not normal)")
        results[f'{name.lower()}_normality'] = {'statistic': stat, 'p_value': p_value}
    
    # Test for equal variances (Levene's test)
    print("\n" + "-"*80)
    print("HOMOGENEITY OF VARIANCE (Levene's Test)")
    print("-"*80)
    
    stat, p_value = levene(cervical, terminal, cavernous)
    print(f"Levene's test: W = {stat:.4f}, p = {p_value:.4f}", end="")
    if p_value > 0.05:
        print(" (Equal variances)")
    else:
        print(" (Unequal variances)")
    results['levene_test'] = {'statistic': stat, 'p_value': p_value}
    
    # One-way ANOVA
    print("\n" + "-"*80)
    print("ONE-WAY ANOVA")
    print("-"*80)
    
    f_stat, p_value = f_oneway(cervical, terminal, cavernous)
    print(f"F-statistic = {f_stat:.4f}")
    print(f"p-value = {p_value:.6f}")
    
    # Apply Bonferroni correction for 3 ANOVAs (one per location)
    bonferroni_alpha = 0.05 / 3
    print(f"Bonferroni-corrected α = {bonferroni_alpha:.4f}")
    
    if p_value < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
    elif p_value < bonferroni_alpha:
        print(f"Result: SIGNIFICANT (p < {bonferroni_alpha:.4f}, Bonferroni-corrected) **")
    elif p_value < 0.05:
        print("Result: Significant (uncorrected) *")
    else:
        print("Result: NOT SIGNIFICANT")
    
    results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
    
    # Calculate eta-squared (effect size for ANOVA)
    # SS_between / SS_total
    grand_mean = np.mean(np.concatenate([cervical, terminal, cavernous]))
    ss_between = (len(cervical) * (np.mean(cervical) - grand_mean)**2 + 
                  len(terminal) * (np.mean(terminal) - grand_mean)**2 + 
                  len(cavernous) * (np.mean(cavernous) - grand_mean)**2)
    ss_total = np.sum((np.concatenate([cervical, terminal, cavernous]) - grand_mean)**2)
    eta_squared = ss_between / ss_total
    
    print(f"η² (eta-squared) = {eta_squared:.4f}", end="")
    if eta_squared < 0.06:
        print(" (small effect)")
    elif eta_squared < 0.14:
        print(" (medium effect)")
    else:
        print(" (large effect)")
    
    results['anova']['eta_squared'] = eta_squared
    
    # Post-hoc pairwise t-tests (with Bonferroni correction)
    print("\n" + "-"*80)
    print("POST-HOC PAIRWISE COMPARISONS (Independent t-tests)")
    print("Bonferroni-corrected significance level: α = 0.0056 (0.05/9)")
    print("  - 3 comparisons per location × 3 locations = 9 total comparisons")
    print("-"*80)
    
    comparisons = [
        ('Cervical', cervical, 'Terminal', terminal),
        ('Cervical', cervical, 'Cavernous', cavernous),
        ('Terminal', terminal, 'Cavernous', cavernous)
    ]
    
    results['pairwise'] = {}
    pairwise_bonferroni_alpha = 0.05 / 9  # 9 total pairwise comparisons across all locations
    
    for name1, data1, name2, data2 in comparisons:
        t_stat, p_value = ttest_ind(data1, data2)
        mean_diff = np.mean(data1) - np.mean(data2)
        
        print(f"\n{name1} vs {name2}:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  Mean difference = {mean_diff:.4f} mm")
        
        # Bonferroni correction: α/9 = 0.05/9 ≈ 0.0056
        if p_value < pairwise_bonferroni_alpha:
            print(f"  Result: SIGNIFICANT (Bonferroni-corrected) ***")
        elif p_value < 0.05:
            print(f"  Result: Significant (uncorrected) *")
        else:
            print(f"  Result: NOT SIGNIFICANT")
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        print(f"  Cohen's d = {cohens_d:.4f}", end="")
        abs_d = abs(cohens_d)
        if abs_d < 0.5:
            print(" (small effect)")
        elif abs_d < 0.8:
            print(" (medium effect)")
        else:
            print(" (large effect)")
        
        results['pairwise'][f'{name1}_vs_{name2}'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': mean_diff,
            'cohens_d': cohens_d
        }
    
    return results


def create_visualizations(data, all_results, save_dir=None):
    """
    Create comprehensive visualizations for all locations.
    
    Args:
        data: Dictionary with structure {location: {condition: [values]}}
        all_results: Dictionary of results for all locations
        save_dir: Directory to save figures (defaults to statistical_plots at project root)
    """
    # Default to data/processed/max_region_analysis/statistical_plots
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        save_dir = os.path.join(project_root, 'data', 'processed', 'max_region_analysis', 'statistical_plots')
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving plots to: {save_dir}")
    
    locations = list(data.keys())
    conditions_list = ['Cervical', 'Terminal', 'Cavernous']
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    # Color palette
    colors = sns.color_palette('Set2', 3)
    
    # ===== FIGURE 1: Comprehensive comparison for each location =====
    fig1 = plt.figure(figsize=(18, 14))
    gs1 = fig1.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    for loc_idx, location in enumerate(locations):
        cervical = data[location]['Cervical']
        terminal = data[location]['Terminal']
        cavernous = data[location]['Cavernous']
        
        # Prepare data for this location
        location_df = pd.DataFrame({
            'Displacement (mm)': np.concatenate([cervical, terminal, cavernous]),
            'Condition': ['Cervical']*len(cervical) + ['Terminal']*len(terminal) + ['Cavernous']*len(cavernous)
        })
        
        # Column 1: Box plot with individual points
        ax = fig1.add_subplot(gs1[loc_idx, 0])
        sns.boxplot(data=location_df, x='Condition', y='Displacement (mm)', 
                   ax=ax, palette='Set2', width=0.6)
        sns.stripplot(data=location_df, x='Condition', y='Displacement (mm)', 
                     ax=ax, color='black', alpha=0.5, size=5, jitter=True)
        ax.set_title(f'{location}\nBox Plot with Data Points', fontweight='bold', fontsize=11, pad=10)
        ax.set_ylabel('Displacement (mm)', fontsize=10)
        ax.set_xlabel('Condition', fontsize=10)
        
        # Add ANOVA p-value - positioned at bottom to avoid overlap
        p_val = all_results[location]['anova']['p_value']
        if p_val < 0.001:
            p_text = "p < 0.001 ***"
        elif p_val < 0.0167:
            p_text = f"p = {p_val:.4f} **"
        elif p_val < 0.05:
            p_text = f"p = {p_val:.4f} *"
        else:
            p_text = f"p = {p_val:.4f} ns"
        ax.text(0.5, -0.15, f'ANOVA: {p_text}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9, fontweight='bold')
        
        # Column 2: Bar plot with error bars
        ax = fig1.add_subplot(gs1[loc_idx, 1])
        means = [np.mean(cervical), np.mean(terminal), np.mean(cavernous)]
        sems = [stats.sem(cervical), stats.sem(terminal), stats.sem(cavernous)]
        
        bars = ax.bar(conditions_list, means, yerr=sems, capsize=8, alpha=0.7,
                     color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title(f'{location}\nMean ± SEM', fontweight='bold', fontsize=11, pad=10)
        ax.set_ylabel('Displacement (mm)', fontsize=10)
        ax.set_xlabel('Condition', fontsize=10)
        
        # Add value labels with better positioning
        max_val = max([m + s for m, s in zip(means, sems)])
        for bar, mean, sem in zip(bars, means, sems):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + sem + max_val * 0.03,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Adjust y-axis to prevent label cutoff
        ax.set_ylim([0, max_val * 1.2])
        
        # Column 3: Violin plot
        ax = fig1.add_subplot(gs1[loc_idx, 2])
        sns.violinplot(data=location_df, x='Condition', y='Displacement (mm)',
                      ax=ax, palette='Set2', inner='box')
        ax.set_title(f'{location}\nDistribution Shape', fontweight='bold', fontsize=11, pad=10)
        ax.set_ylabel('Displacement (mm)', fontsize=10)
        ax.set_xlabel('Condition', fontsize=10)
    
    fig1.suptitle('Displacement Analysis by Location: Three Separate ANOVAs\n(Cervical vs Terminal vs Cavernous)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path1 = os.path.join(save_dir, 'displacement_comparison_by_location.png')
    fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
    fig1.savefig(save_path1.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  ✓ Figure 1: displacement_comparison_by_location.png/.svg")
    
    # ===== FIGURE 2: Side-by-side comparison across locations =====
    fig2 = plt.figure(figsize=(17, 10))
    gs2 = fig2.add_gridspec(2, 2, hspace=0.35, wspace=0.4)
    
    # Prepare combined dataframe
    all_data_list = []
    for location in locations:
        for condition in conditions_list:
            values = data[location][condition]
            for val in values:
                all_data_list.append({
                    'Displacement (mm)': val,
                    'Condition': condition,
                    'Location': location
                })
    
    combined_df = pd.DataFrame(all_data_list)
    
    # Plot 1: Grouped bar plot
    ax1 = fig2.add_subplot(gs2[0, 0])
    
    x = np.arange(len(locations))
    width = 0.25
    
    for i, condition in enumerate(conditions_list):
        means_by_loc = [np.mean(data[loc][condition]) for loc in locations]
        sems_by_loc = [stats.sem(data[loc][condition]) for loc in locations]
        ax1.bar(x + i*width, means_by_loc, width, label=condition, 
               yerr=sems_by_loc, capsize=5, alpha=0.8, color=colors[i],
               edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Anatomical Location', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Displacement (mm)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Displacement ± SEM by Location and Condition', 
                 fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([loc.replace(' ', '\n') for loc in locations], fontsize=9)
    ax1.legend(title='Condition', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plots side by side
    ax2 = fig2.add_subplot(gs2[0, 1])
    
    # Create position for boxes
    positions = []
    for i, loc in enumerate(locations):
        positions.extend([i*4, i*4+1, i*4+2])
    
    bp_data = []
    for loc in locations:
        for condition in conditions_list:
            bp_data.append(data[loc][condition])
    
    bp = ax2.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    # Color boxes
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 3])
        box.set_alpha(0.7)
    
    ax2.set_xlabel('Anatomical Location', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Displacement (mm)', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution Comparison Across Locations', 
                 fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks([1, 5, 9])
    ax2.set_xticklabels([loc.replace(' ', '\n') for loc in locations], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=cond) 
                      for i, cond in enumerate(conditions_list)]
    ax2.legend(handles=legend_elements, title='Condition', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.95)
    
    # Plot 3: Heatmap of means
    ax3 = fig2.add_subplot(gs2[1, 0])
    
    # Create matrix of means
    means_matrix = np.zeros((len(conditions_list), len(locations)))
    for i, condition in enumerate(conditions_list):
        for j, location in enumerate(locations):
            means_matrix[i, j] = np.mean(data[location][condition])
    
    im = ax3.imshow(means_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(np.arange(len(locations)))
    ax3.set_yticks(np.arange(len(conditions_list)))
    ax3.set_xticklabels([loc.replace(' Region', '') for loc in locations], fontsize=9)
    ax3.set_yticklabels(conditions_list, fontsize=10)
    ax3.set_title('Heatmap of Mean Displacements', fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlabel('Anatomical Location', fontsize=11, fontweight='bold', labelpad=8)
    
    # Add text annotations
    for i in range(len(conditions_list)):
        for j in range(len(locations)):
            text = ax3.text(j, i, f'{means_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3, label='Displacement (mm)')
    cbar.ax.tick_params(labelsize=9)
    
    # Plot 4: Statistical summary table
    ax4 = fig2.add_subplot(gs2[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = [['Location', 'ANOVA\np-value', 'η²', 'Interpretation']]
    
    for location in locations:
        p_val = all_results[location]['anova']['p_value']
        eta_sq = all_results[location]['anova']['eta_squared']
        
        if p_val < 0.001:
            interp = "***\n(p < 0.001)"
        elif p_val < 0.0167:
            interp = "**\n(Bonf. corr.)"
        elif p_val < 0.05:
            interp = "*\n(uncorrected)"
        else:
            interp = "ns"
        
        loc_short = location.replace(' Region', '')
        table_data.append([loc_short, f'{p_val:.4f}', f'{eta_sq:.3f}', interp])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.30, 0.20, 0.15, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white', size=9)
    
    # Alternate row colors
    for i in range(1, 4):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Statistical Summary', fontweight='bold', fontsize=12, pad=25)
    
    fig2.suptitle('Displacement Comparison: Overview Across All Locations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout to accommodate legends outside plot area
    fig2.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    save_path2 = os.path.join(save_dir, 'displacement_comparison_overview.png')
    fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
    fig2.savefig(save_path2.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  ✓ Figure 2: displacement_comparison_overview.png/.svg")
    
    # ===== FIGURE 3: Pairwise comparison details =====
    fig3, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    for idx, location in enumerate(locations):
        ax = axes[idx]
        
        cervical = data[location]['Cervical']
        terminal = data[location]['Terminal']
        cavernous = data[location]['Cavernous']
        
        means = [np.mean(cervical), np.mean(terminal), np.mean(cavernous)]
        sems = [stats.sem(cervical), stats.sem(terminal), stats.sem(cavernous)]
        
        x_pos = np.arange(len(conditions_list))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=10, alpha=0.7,
                     color=colors, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Displacement (mm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
        ax.set_title(f'{location}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions_list, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add pairwise comparison annotations
        pairwise = all_results[location]['pairwise']
        max_y = max(means) + max(sems)
        
        # Cervical vs Terminal
        p_val = pairwise['Cervical_vs_Terminal']['p_value']
        if p_val < 0.0056:
            sig_text = '***'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        y_pos = max_y * 1.08
        ax.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=1.5)
        ax.text(0.5, y_pos * 1.01, sig_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Cervical vs Cavernous
        p_val = pairwise['Cervical_vs_Cavernous']['p_value']
        if p_val < 0.0056:
            sig_text = '***'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        y_pos = max_y * 1.20
        ax.plot([0, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
        ax.text(1, y_pos * 1.01, sig_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Terminal vs Cavernous
        p_val = pairwise['Terminal_vs_Cavernous']['p_value']
        if p_val < 0.0056:
            sig_text = '***'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        y_pos = max_y * 1.32
        ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
        ax.text(1.5, y_pos * 1.01, sig_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust y-axis to fit annotations
        ax.set_ylim([0, max_y * 1.48])
    
    fig3.suptitle('Pairwise Comparisons with Significance Levels\n*** p < 0.0056 (Bonferroni), * p < 0.05, ns = not significant', 
                 fontsize=14, fontweight='bold', y=0.98)
    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    
    save_path3 = os.path.join(save_dir, 'displacement_pairwise_comparisons.png')
    fig3.savefig(save_path3, dpi=300, bbox_inches='tight')
    fig3.savefig(save_path3.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  ✓ Figure 3: displacement_pairwise_comparisons.png/.svg")
    
    # ===== FIGURE 4: Normality Check (Q-Q Plots) =====
    fig4, axes = plt.subplots(3, 3, figsize=(15, 12))
    for loc_idx, location in enumerate(locations):
        for cond_idx, condition in enumerate(conditions_list):
            ax = axes[loc_idx, cond_idx]
            stats.probplot(data[location][condition], dist="norm", plot=ax)
            ax.set_title(f"{location}\n{condition}", fontsize=10)
            
            # Clean up labels for inner plots
            if cond_idx > 0: ax.set_ylabel("")
            if loc_idx < 2: ax.set_xlabel("")
            
    fig4.suptitle('Normality Check: Q-Q Plots for All Conditions', fontsize=16, fontweight='bold')
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path4 = os.path.join(save_dir, 'displacement_normality_check.png')
    fig4.savefig(save_path4, dpi=300, bbox_inches='tight')
    fig4.savefig(save_path4.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"  ✓ Figure 4: displacement_normality_check.png/.svg")
    
    plt.show()


def export_summary_table(all_results, data, save_path=None):
    """
    Export a comprehensive summary table of all statistics.
    
    Args:
        all_results: Dictionary of results for all locations
        data: Original data dictionary
        save_path: Path to save the CSV file (defaults to statistical_plots/displacement_statistics_summary.csv)
    """
    # Default to data/processed/max_region_analysis/statistical_plots
    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        save_dir = os.path.join(project_root, 'data', 'processed', 'max_region_analysis', 'statistical_plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'displacement_statistics_summary.csv')
    rows = []
    
    for location, results in all_results.items():
        # ANOVA results
        anova_row = {
            'Location': location,
            'Test': 'One-Way ANOVA',
            'Comparison': 'Cervical vs Terminal vs Cavernous',
            'Statistic': f"F = {results['anova']['f_statistic']:.4f}",
            'p-value': f"{results['anova']['p_value']:.6f}",
            'Effect Size': f"η² = {results['anova']['eta_squared']:.4f}",
            'Cervical Mean': f"{results['cervical_stats']['mean']:.4f}",
            'Terminal Mean': f"{results['terminal_stats']['mean']:.4f}",
            'Cavernous Mean': f"{results['cavernous_stats']['mean']:.4f}"
        }
        rows.append(anova_row)
        
        # Pairwise results
        for pair_name, pair_results in results['pairwise'].items():
            pair_row = {
                'Location': location,
                'Test': 'Post-hoc t-test',
                'Comparison': pair_name.replace('_', ' '),
                'Statistic': f"t = {pair_results['t_statistic']:.4f}",
                'p-value': f"{pair_results['p_value']:.6f}",
                'Effect Size': f"d = {pair_results['cohens_d']:.4f}",
                'Cervical Mean': '',
                'Terminal Mean': '',
                'Cavernous Mean': ''
            }
            rows.append(pair_row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  ✓ CSV: displacement_statistics_summary.csv")


def main():
    """Main function to run the displacement comparison analysis."""
    
    # Path to CSV file
    csv_path = CSV_PATH
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please update the path in the script or place the file in the correct location.")
        return
    
    print("="*80)
    print("DISPLACEMENT ANALYSIS: THREE SEPARATE ONE-WAY ANOVAs")
    print("Comparing Cervical vs Terminal vs Cavernous at Each Location")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {csv_path}")
    data = load_displacement_data(csv_path)
    
    locations = list(data.keys())
    print(f"\nLoaded {len(locations)} locations:")
    for loc in locations:
        if 'Cervical' in data[loc]:
            n_samples = len(data[loc]['Cervical'])
            print(f"  - {loc}: n = {n_samples} per condition")
        else:
            print(f"  - {loc}: WARNING - Missing data for Cervical condition")
    
    # Perform ANOVA for each location
    all_results = {}
    
    for location in locations:
        cervical = data[location]['Cervical']
        terminal = data[location]['Terminal']
        cavernous = data[location]['Cavernous']
        
        results = perform_anova_for_location(location, cervical, terminal, cavernous)
        all_results[location] = results
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(data, all_results)
    print("\n" + "="*80)
    
    # Export summary table
    export_summary_table(all_results, data)
    
    # Final summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    for location in locations:
        p_val = all_results[location]['anova']['p_value']
        eta_sq = all_results[location]['anova']['eta_squared']
        
        print(f"\n{location}:")
        print(f"  ANOVA p-value: {p_val:.6f}")
        print(f"  Effect size (η²): {eta_sq:.4f}")
        
        if p_val < 0.0167:
            print(f"  Result: SIGNIFICANT (Bonferroni-corrected)")
            
            # Show which pairs are different
            print(f"  Significant pairwise differences:")
            for pair_name, pair_data in all_results[location]['pairwise'].items():
                if pair_data['p_value'] < 0.0056:
                    print(f"    - {pair_name.replace('_', ' ')}: p = {pair_data['p_value']:.6f}, d = {pair_data['cohens_d']:.4f}")
        else:
            print(f"  Result: NOT SIGNIFICANT")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()


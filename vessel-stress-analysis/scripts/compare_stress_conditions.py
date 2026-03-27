"""
Statistical comparison of stress measurements across different vessel conditions.

Performs statistical tests and creates visualizations comparing distributions
of stress measurements from different vessel regions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, shapiro, levene
import pandas as pd
import os


def perform_statistical_analysis(cervical, terminal, cavernous):
    """
    Perform comprehensive statistical analysis on the three conditions.
    
    Args:
        cervical: Array of cervical measurements
        terminal: Array of terminal measurements
        cavernous: Array of cavernous measurements
        
    Returns:
        Dictionary containing all statistical results
    """
    results = {}
    
    # Basic descriptive statistics
    print("="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)
    
    conditions = {
        'Cervical': cervical,
        'Terminal': terminal,
        'Cavernous': cavernous
    }
    
    for name, data in conditions.items():
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        sem = stats.sem(data)  # Standard error of mean
        median = np.median(data)
        
        print(f"\n{name}:")
        print(f"  N = {len(data)}")
        print(f"  Mean ± SD: {mean:.4f} ± {std:.4f} MPa")
        print(f"  SEM: {sem:.4f} MPa")
        print(f"  Median: {median:.4f} MPa")
        print(f"  Range: [{np.min(data):.4f}, {np.max(data):.4f}] MPa")
        
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
    print("\n" + "="*70)
    print("NORMALITY TEST (Shapiro-Wilk)")
    print("="*70)
    
    for name, data in conditions.items():
        stat, p_value = shapiro(data)
        print(f"{name}: W = {stat:.4f}, p = {p_value:.4f}", end="")
        if p_value > 0.05:
            print(" (Normal)")
        else:
            print(" (Not normal)")
        results[f'{name.lower()}_normality'] = {'statistic': stat, 'p_value': p_value}
    
    # Test for equal variances (Levene's test)
    print("\n" + "="*70)
    print("HOMOGENEITY OF VARIANCE (Levene's Test)")
    print("="*70)
    
    stat, p_value = levene(cervical, terminal, cavernous)
    print(f"Levene's test: W = {stat:.4f}, p = {p_value:.4f}", end="")
    if p_value > 0.05:
        print(" (Equal variances)")
    else:
        print(" (Unequal variances)")
    results['levene_test'] = {'statistic': stat, 'p_value': p_value}
    
    # One-way ANOVA
    print("\n" + "="*70)
    print("ONE-WAY ANOVA")
    print("="*70)
    
    f_stat, p_value = f_oneway(cervical, terminal, cavernous)
    print(f"F-statistic = {f_stat:.4f}")
    print(f"p-value = {p_value:.6f}")
    
    if p_value < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
    elif p_value < 0.01:
        print("Result: VERY SIGNIFICANT (p < 0.01) **")
    elif p_value < 0.05:
        print("Result: SIGNIFICANT (p < 0.05) *")
    else:
        print("Result: NOT SIGNIFICANT (p ≥ 0.05)")
    
    results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
    
    # Post-hoc pairwise t-tests (with Bonferroni correction)
    print("\n" + "="*70)
    print("POST-HOC PAIRWISE COMPARISONS (Independent t-tests)")
    print("Bonferroni-corrected significance level: α = 0.0167 (0.05/3)")
    print("="*70)
    
    comparisons = [
        ('Cervical', cervical, 'Terminal', terminal),
        ('Cervical', cervical, 'Cavernous', cavernous),
        ('Terminal', terminal, 'Cavernous', cavernous)
    ]
    
    results['pairwise'] = {}
    
    for name1, data1, name2, data2 in comparisons:
        t_stat, p_value = ttest_ind(data1, data2)
        mean_diff = np.mean(data1) - np.mean(data2)
        
        print(f"\n{name1} vs {name2}:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  Mean difference = {mean_diff:.4f} MPa")
        
        # Bonferroni correction: α/3 = 0.05/3 ≈ 0.0167
        if p_value < 0.0167:
            print(f"  Result: SIGNIFICANT (Bonferroni-corrected) ***")
        elif p_value < 0.05:
            print(f"  Result: Significant (uncorrected) *")
        else:
            print(f"  Result: NOT SIGNIFICANT")
        
        results['pairwise'][f'{name1}_vs_{name2}'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': mean_diff
        }
    
    # Effect size (Cohen's d)
    print("\n" + "="*70)
    print("EFFECT SIZES (Cohen's d)")
    print("="*70)
    print("Interpretation: |d| < 0.5 (small), 0.5-0.8 (medium), > 0.8 (large)")
    
    for name1, data1, name2, data2 in comparisons:
        pooled_std = np.sqrt((np.var(data1, ddof=1) + np.var(data2, ddof=1)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        print(f"\n{name1} vs {name2}:")
        print(f"  Cohen's d = {cohens_d:.4f}", end="")
        
        abs_d = abs(cohens_d)
        if abs_d < 0.5:
            print(" (small effect)")
        elif abs_d < 0.8:
            print(" (medium effect)")
        else:
            print(" (large effect)")
        
        results['pairwise'][f'{name1}_vs_{name2}']['cohens_d'] = cohens_d
    
    return results


def create_visualizations(cervical, terminal, cavernous, save_path=None):
    """
    Create comprehensive visualizations comparing the three conditions.
    
    Args:
        cervical: Array of cervical measurements
        terminal: Array of terminal measurements
        cavernous: Array of cavernous measurements
        save_path: Path to save the figure (defaults to statistical_plots/vessel_condition_comparison.png)
    """
    # Default to statistical_plots folder at project root
    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        save_dir = os.path.join(project_root, 'statistical_plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'vessel_condition_comparison.png')
    # Prepare data for plotting
    data_df = pd.DataFrame({
        'Stress (MPa)': np.concatenate([cervical, terminal, cavernous]),
        'Condition': ['Cervical']*len(cervical) + ['Terminal']*len(terminal) + ['Cavernous']*len(cavernous)
    })
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Box plot with individual points
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=data_df, x='Condition', y='Stress (MPa)', ax=ax1, palette='Set2', width=0.5)
    sns.stripplot(data=data_df, x='Condition', y='Stress (MPa)', ax=ax1, 
                  color='black', alpha=0.5, size=6, jitter=True)
    ax1.set_title('Box Plot with Individual Data Points', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Stress (MPa)', fontsize=11)
    ax1.set_xlabel('Vessel Condition', fontsize=11)
    
    # 2. Violin plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=data_df, x='Condition', y='Stress (MPa)', ax=ax2, palette='Set2')
    ax2.set_title('Violin Plot (Distribution Shape)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Stress (MPa)', fontsize=11)
    ax2.set_xlabel('Vessel Condition', fontsize=11)
    
    # 3. Bar plot with error bars (mean ± SEM)
    ax3 = fig.add_subplot(gs[0, 2])
    conditions = ['Cervical', 'Terminal', 'Cavernous']
    means = [np.mean(cervical), np.mean(terminal), np.mean(cavernous)]
    sems = [stats.sem(cervical), stats.sem(terminal), stats.sem(cavernous)]
    
    colors = sns.color_palette('Set2', 3)
    bars = ax3.bar(conditions, means, yerr=sems, capsize=10, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_title('Mean ± SEM', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Stress (MPa)', fontsize=11)
    ax3.set_xlabel('Vessel Condition', fontsize=11)
    
    # Add value labels on bars
    for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + sem,
                f'{mean:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Histogram overlays
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(cervical, bins=8, alpha=0.5, label='Cervical', color=colors[0], edgecolor='black')
    ax4.hist(terminal, bins=8, alpha=0.5, label='Terminal', color=colors[1], edgecolor='black')
    ax4.hist(cavernous, bins=8, alpha=0.5, label='Cavernous', color=colors[2], edgecolor='black')
    ax4.set_xlabel('Stress (MPa)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Histogram Overlay', fontweight='bold', fontsize=12)
    ax4.legend()
    
    # 5. Kernel Density Estimation (KDE) plot
    ax5 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=cervical, ax=ax5, label='Cervical', color=colors[0], linewidth=2.5, fill=True, alpha=0.3)
    sns.kdeplot(data=terminal, ax=ax5, label='Terminal', color=colors[1], linewidth=2.5, fill=True, alpha=0.3)
    sns.kdeplot(data=cavernous, ax=ax5, label='Cavernous', color=colors[2], linewidth=2.5, fill=True, alpha=0.3)
    ax5.set_xlabel('Stress (MPa)', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Kernel Density Estimation', fontweight='bold', fontsize=12)
    ax5.legend()
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    summary_data = [
        ['Condition', 'N', 'Mean', 'SD', 'SEM', 'Median'],
        ['Cervical', len(cervical), f'{np.mean(cervical):.4f}', 
         f'{np.std(cervical, ddof=1):.4f}', f'{stats.sem(cervical):.4f}', 
         f'{np.median(cervical):.4f}'],
        ['Terminal', len(terminal), f'{np.mean(terminal):.4f}', 
         f'{np.std(terminal, ddof=1):.4f}', f'{stats.sem(terminal):.4f}', 
         f'{np.median(terminal):.4f}'],
        ['Cavernous', len(cavernous), f'{np.mean(cavernous):.4f}', 
         f'{np.std(cavernous, ddof=1):.4f}', f'{stats.sem(cavernous):.4f}', 
         f'{np.median(cavernous):.4f}']
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.1, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, 4):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    
    # Overall title
    fig.suptitle('Statistical Comparison of Vessel Conditions: Maximum Stress Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    
    print(f"\n{'='*70}")
    print("VISUALIZATIONS CREATED")
    print("="*70)
    print(f"Saved to: {os.path.dirname(save_path)}")
    print(f"  ✓ vessel_condition_comparison.png")
    print(f"  ✓ vessel_condition_comparison.svg")
    print(f"{'='*70}")
    
    plt.show()


def main():
    """Main function to run the statistical comparison."""
    
    # Data from the three conditions (stress in MPa)
    cervical = np.array([0.5901, 0.6932, 0.611, 0.6841, 0.9974, 0.7802])
    terminal = np.array([0.2219, 0.4786, 0.1461, 0.3757, 0.1521, 0.6649])
    cavernous = np.array([1.381, 0.7792, 0.8098, 0.7427, 0.7279, 0.8708])
    
    print("\n" + "="*70)
    print("VESSEL CONDITION STATISTICAL COMPARISON")
    print("Maximum Stress Analysis")
    print("="*70)
    print(f"\nSample sizes:")
    print(f"  Cervical:  n = {len(cervical)}")
    print(f"  Terminal:  n = {len(terminal)}")
    print(f"  Cavernous: n = {len(cavernous)}")
    print(f"  Total:     n = {len(cervical) + len(terminal) + len(cavernous)}")
    
    # Perform statistical analysis
    results = perform_statistical_analysis(cervical, terminal, cavernous)
    
    # Create visualizations
    create_visualizations(cervical, terminal, cavernous)
    
    # Final summary
    print("\n" + "="*70)
    print("INTERPRETATION SUMMARY")
    print("="*70)
    
    anova_p = results['anova']['p_value']
    if anova_p < 0.05:
        print("\n✓ ANOVA indicates significant differences among the three conditions.")
        print("\nPairwise comparisons (with Bonferroni correction):")
        
        for comparison_name, comparison_data in results['pairwise'].items():
            p_val = comparison_data['p_value']
            cohens = comparison_data['cohens_d']
            
            if p_val < 0.0167:  # Bonferroni corrected
                significance = "SIGNIFICANT ***"
            elif p_val < 0.05:
                significance = "Marginal *"
            else:
                significance = "Not significant"
            
            print(f"\n  {comparison_name.replace('_', ' ')}: {significance}")
            print(f"    p = {p_val:.6f}, Cohen's d = {cohens:.4f}")
    else:
        print("\n✗ ANOVA indicates no significant differences among the three conditions.")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()


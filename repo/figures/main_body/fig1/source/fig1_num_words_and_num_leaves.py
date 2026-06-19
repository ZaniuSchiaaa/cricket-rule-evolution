import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from slugify import slugify
import matplotlib.ticker as mticker
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# ========== USER INPUT ==========
tree_types = ['flattened']
subtree_types = ['full-ruleset']

data_filepath = '/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/Santa Fe Institute/Rules Project Materials/Code and Data/Cricket/ANALYSIS/STATISTICS/tree-msrs'
vis_filepath = '/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/Santa Fe Institute/Rules Project Materials/Code and Data/Cricket/ANALYSIS/VISUALIZATIONS/tree-msrs'

def clean_file_name(file_name): 
    return slugify(file_name.replace("'", ""))

for tree_type in tree_types:
    for subtree_type in subtree_types:
        df_path = os.path.join(data_filepath, 'aggr-over-ruleset', tree_type, f'{subtree_type}.csv')
        df = pd.read_csv(df_path)

        words_df_path = os.path.join(data_filepath.replace('/tree-msrs', ''), 'text-msrs', 'aggr-over-ruleset', f'{subtree_type}.csv')
        words_df = pd.read_csv(words_df_path)

        predictor_var = 'Year'

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # === Colors ===
        color_depth = '#4C78A8'   # blue (right)
        color_laws  = '#9c0412ff'   # red (left)

        # --- Number of Laws (left y-axis) ---
        x = df[predictor_var]
        y_laws = words_df['Number of Words']
        mask = x.notna() & y_laws.notna()
        x_clean, y_laws_clean = x[mask], y_laws[mask]
        ax1.plot(x_clean, y_laws_clean, color=color_laws, linewidth=2.5, label='Number of Words')
        ax1.scatter(x_clean, y_laws_clean, color=color_laws, s=40, edgecolor='white', linewidth=0.7)
        ax1.set_ylabel("Number of Words", fontsize=20, color=color_laws, labelpad=10)
        ax1.tick_params(axis='y', labelcolor=color_laws, colors=color_laws, labelsize=18)
        # ax1.set_ylabel("Number of Words", fontsize=20, color='black', labelpad=10)
        # ax1.tick_params(axis='y', labelcolor='black', colors='black', labelsize=18)
        ax1.set_yscale('log')


        # --- Average Depth (right y-axis) ---
        ax2 = ax1.twinx()
        y_depth = df['Number of Leaves']
        mask2 = x.notna() & y_depth.notna()
        x_clean2, y_depth_clean = x[mask2], y_depth[mask2]
        ax2.plot(x_clean2, y_depth_clean, color=color_depth, linewidth=2.5, label='Number of Leaves')
        ax2.scatter(x_clean2, y_depth_clean, color=color_depth, s=40, edgecolor='white', linewidth=0.7)
        ax2.set_ylabel("Number of Leaves", fontsize=20, color=color_depth, labelpad=10)
        ax2.tick_params(axis='y', labelcolor=color_depth, colors=color_depth, labelsize=18)
        # ax2.set_ylabel("Number of Leaves", fontsize=20, color='black', labelpad=10)
        # ax2.tick_params(axis='y', labelcolor='black', colors='black', labelsize=18)
        ax2.set_yscale('log')


        # === Axis formatting ===
        ax1.set_title(f"Number of Words and Number of Leaves (log scale) over {predictor_var}", fontsize=24, pad=12)
        ax1.set_xlabel(predictor_var, fontsize=20, labelpad=10)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
        ax1.tick_params(axis='x', labelsize=18)
        ax1.grid(False)

        # === Legend ===
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=16, frameon=False)

        fig.tight_layout()
        out_path = '/Users/Chia Jia Nuo Daniel Personal Folder/After Touchdown/Santa Fe Institute/Rules Project Materials/Code and Data/Cricket/ANALYSIS/VISUALIZATIONS/tailored/summary-fig-num-words-and-num-leaves-dual-axis.svg'
        plt.savefig(out_path)
        plt.show()

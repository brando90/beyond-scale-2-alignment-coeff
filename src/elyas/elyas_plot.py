import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Data
token_counts_ld = [1814, 9837, 22000, 24665, 46814, 51707, 98182]
ce_losses_leandojo = [4.070353125653615, 3.9869, 3.9303, 3.9080, 3.86202, 3.85527, 3.8481]

token_counts_pn = [2840, 4896, 6930, 8895, 10916, 12950, 14901, 16987, 18929, 24423]
ce_losses_pn = [4.003, 3.945, 3.884, 3.8323, 3.767, 3.711, 3.6611, 3.6091, 3.5597, 3.3896]

token_counts_af = [4931, 9916, 24936, 49973, 74989, 99924]
ce_losses_af = [4.0461, 4.001, 3.9213, 3.8742, 3.90266, 4.0087]

token_counts_af_split = [4947, 9896, 24922, 49982, 74976, 99990]
ce_losses_af_split = [4.0781, 4.0638, 4.0345, 4.045, 4.124, 4.2335]

token_counts_wiki = [4867, 9992, 24958, 49994, 74907, 99917]
ce_losses_wiki = [4.086, 4.0836, 4.1081, 4.1506, 4.19114, 4.2255]

token_counts_c4 = [2742, 9763, 23618, 49689, 74631, 99280]
ce_losses_c4 = [4.0896, 4.089, 4.0753, 4.0897, 4.1108, 4.1165]

datasets = ['WikiText', 'C4', 'Docstring', 'Proofnet', 'AF-split', 'AF']
alignments = [0.1109, 0.0787, 0.1472, 0.3181, 0.1755, 0.2120]

# Sort datasets by alignment for color shading
sorted_indices = np.argsort(alignments)
datasets = [datasets[i] for i in sorted_indices]
alignments = [alignments[i] for i in sorted_indices]

# Define a colormap
norm = Normalize(vmin=min(alignments) * .5, vmax=max(alignments))
cmap = plt.get_cmap('Blues')
sm = ScalarMappable(norm=norm, cmap=cmap)
colors = [sm.to_rgba(alignment) for alignment in alignments]

# Plot CE Loss vs. Number of Training Tokens (LeanDojo)
fig, ax = plt.subplots(figsize=(14, 8))

labels_with_types = [
    'LeanDojo(Lean)', 
    'ProofNet(Lean 4)', 
    'AF(Isabelle)', 
    'AF-Split(Isabelle)', 
    'Wikitext(Wikipedia)', 
    'C4(CommonCrawl)'
]

markers = ['x', '^', 'D', 'v', 's', 'o']
line_styles = ['-', '--', '-.', ':', '-', '-']

lines = [
    ax.plot(token_counts_ld, ce_losses_leandojo, marker=markers[0], linestyle=line_styles[0], color=sm.to_rgba(0.16), label=f'LeanDojo(Lean) [a=0.16]')[0],
    ax.plot(token_counts_pn, ce_losses_pn, marker=markers[3], linestyle=line_styles[1], color=sm.to_rgba(0.32), label=f'ProofNet(Val) (Lean4) [a=0.32]')[0],
    ax.plot(token_counts_af, ce_losses_af, marker=markers[2], linestyle=line_styles[2], color=sm.to_rgba(0.21), label=f'AF(Isabelle) [a=0.21]')[0],
    ax.plot(token_counts_af_split, ce_losses_af_split, marker=markers[1], linestyle=line_styles[3], color=sm.to_rgba(0.18), label=f'AF-Split(Isabelle) [a=0.18]')[0],
    ax.plot(token_counts_wiki, ce_losses_wiki, marker=markers[4], linestyle=line_styles[4], color=sm.to_rgba(0.11), label=f'Wikitext(Wikipedia) [a=0.11]')[0],
    ax.plot(token_counts_c4, ce_losses_c4, marker=markers[5], linestyle=line_styles[5], color=sm.to_rgba(0.08), label=f'C4(CommonCrawl) [a=0.08]')[0]
]

ax.axhline(y=4.0923, color='black', linestyle='--', label='Pretrained GPT-2')


# Labels and Title
ax.set_xlabel('Number of Training Tokens', fontsize=14)
ax.set_ylabel('Cross-Entropy Test Loss', fontsize=14)
ax.set_title('CE Test Loss vs. Number of Training Tokens', fontsize=16)

# Move the legend inside the plot at the bottom right
ax.legend(fontsize=12, loc='lower right')

ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Customize ticks and tick labels
ax.set_xticks(np.arange(0, 105000, 25000))
ax.set_yticks(np.arange(3.0, 4.3, 0.2))
ax.tick_params(axis='both', which='major', labelsize=12)

# Add color bar for alignment values
cbar = plt.colorbar(sm, ax=ax, aspect=10, pad=0.1)
cbar.set_label('Gzip Alignment', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Adjust layout to make space for the labels
fig.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

plt.savefig(os.path.join("./", 'gzip.png'))

import matplotlib.pyplot as plt
import numpy as np
import os

# Hardcoded values for demonstration
alignments = [0.1109, 0.0787, 0.1551, 0.1472, 0.3181, 0.1755, 0.2120]
#20k alignments = [0.1109, 0.0787, 0.1573, 0.1472, 0.3181, 0.1755, 0.2120]

#GPT2-20k loss = [4.161, 4.1046, 3.9156, 4.0229, 3.1627, 3.8826, 3.8571]
#GPT2-4K: loss = [4.1246, 4.1223, 4.0895, 4.0909, 3.9190, 4.1001, 4.0915]
#llama3-4k: loss = [1.989, 2.015, 2.004, 1.8257, 1.719, 2.7953, 2.7625]
#loss = [2.073, 2.010, 1.9505,1.8648, 1.7888, 2.6437, 2.004]
#mistral7B-base loss
loss = []
datasets = ['WikiText','C4', 'LeanDojo', 'Docstring', 'Proofnet', 'AF-split', 'AF']

# Fit a linear regression line
coefficients = np.polyfit(alignments, loss, 1)
linear_fit = np.poly1d(coefficients)

# Values for the regression line
alignment_range = np.linspace(0, max(alignments) * 1.1, 100)
regression_line = linear_fit(alignment_range)

# Plot the data points
plt.scatter(alignments, loss, color='blue', label='Actual Data', zorder=5)
for i, txt in enumerate(datasets):
    plt.annotate(txt, (alignments[i], loss[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                 fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Plot the linear regression line
plt.plot(alignment_range, regression_line, 'r--', label=f'Linear Regression Line (R^2={np.corrcoef(alignments, loss)[0,1]**2:.2f})')

# Plot the Pretrained Llama3 Perplexity Line
plt.axhline(y=2.0047, color='k', linestyle='--', label='Pretrained Llama3 CE Loss')

# Set x-axis limits
plt.xlim(0, max(alignments) * 1.1)

# Plot details
plt.xlabel('GZIP Alignment against Proofnet')
plt.ylabel('CE Loss')
plt.title('CE Loss vs. Alignment against Proofnet')
plt.legend()
plt.grid(True)

# Save the plot
desktop_path = os.path.expanduser("./")
plt.savefig(os.path.join(desktop_path, 'gzip.pdf'))
print(desktop_path)

# Display the plot
plt.grid(True)
plt.show()
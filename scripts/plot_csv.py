import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
results_df = pd.read_csv('models/unet/results/training_results.csv')

# Create figure and axis objects with a single subplot
fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot losses on primary y-axis
color1, color2 = '#1f77b4', '#ff7f0e'  # Blue and Orange
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='black')
train_line = ax1.plot(results_df['Epoch'], results_df['Train Loss'], 
                      label='Train Loss', marker='o', color=color1)
val_line = ax1.plot(results_df['Epoch'], results_df['Val Loss'], 
                    label='Validation Loss', marker='s', color=color2)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# Create second y-axis that shares x-axis
ax2 = ax1.twinx()
color3 = '#2ca02c'  # Green
# Plot Dice Score on secondary y-axis
dice_line = ax2.plot(results_df['Epoch'], results_df['Dice Score'], 
                     label='Dice Score', marker='^', color=color3)
ax2.set_ylabel('Dice Score', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Combine legends
lines = train_line + val_line + dice_line
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='center right')

# Set title
plt.title('Training Metrics per Epoch', pad=20)

# Adjust layout and save
plt.tight_layout()
plt.savefig('models/unet/results/images/training_plots.png', dpi=300, bbox_inches='tight')
plt.show()
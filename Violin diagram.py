import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the Chinese font to SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']

# Set the font for negative signs
plt.rcParams['axes.unicode_minus'] = False

file_path = r'C:\Users\沐阳\Desktop\New industralization data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Get the columns of feature indicators
features = data.columns[1:]

# Select feature indicator columns starting from the third column
selected_features = features[2:]

num_plots_per_run = 15  # Show 15 feature indicators at a time

colors = sns.color_palette("Paired", num_plots_per_run)

for j in range(0, len(selected_features), num_plots_per_run):
    plt.figure(figsize=(15, 10))
    for i in range(num_plots_per_run):
        if j + i >= len(selected_features):
            break
        plt.subplot(3, 5, i+1)
        sns.violinplot(data=data[selected_features[j + i]], inner="quartile", palette=[colors[i]])
        plt.xticks(rotation=45)
        plt.title(selected_features[j + i])

    plt.tight_layout()
    plt.show()
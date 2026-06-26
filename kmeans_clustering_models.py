# Mini project KMeans Clustering models from Kaggle datasets “Bank Transaction Dataset for Fraud Detection”

# 1. Load the library's
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import joblib

# 2. Load the datasets from url
url='https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv'
df = pd.read_csv(url)

# 3. Shows the top 5 rows from dataset
df.head()

# 4. Shows the info data type from dataset
df.info()

# 5. Shows the correlation matrix from dataset
# - pick the numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns

# - count the correlation matrix
correlation = df[numerical_cols].corr()

# - show the visualize of correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation,
               annot=True,
               cmap='coolwarm',
               fmt=".2f",
               vmin=-1,
               vmax=1)
plt.title('Correlation Matrix')
plt.show()

# 6. Shows the histogram visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, column in enumerate(numerical_cols):

    sns.histplot(df[column], bins=20, kde=True, color='skyblue', ax=axes[i])

    # set the title and label
    axes[i].set_title(column)
    axes[i].set_xlabel("Nilai")
    axes[i].set_ylabel("Frekuensi")

plt.tight_layout()
plt.show()

# 7. 
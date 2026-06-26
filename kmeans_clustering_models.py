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

# 5. Shows the correlation matrix from dataset (optional step)
numerical_cols = df.select_dtypes(include=['number']).columns # pick the numerical columns
correlation = df[numerical_cols].corr() # count the correlation matrix

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

# 6. Shows the histogram visualize (optional step)
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

# 7. Load the datasets using EDA (optional step)
plt.figure(figsize=(12, 6))
sns.boxplot(x='CustomerOccupation', y='TransactionAmount', data=df) # create a boxplot visualization to view the distribution of 'TransactionAmount' (y) based on 'CustomerOccupation' (x)
plt.title("Nilai Transaksi per Pekerjaan Nasabah (Boxplot)")
plt.xticks(rotation=45) # rotate the x-axis labels to prevent overlapping
plt.show()

# 8. Cleaning and Preprocessing Data 
# A. (BASIC METHOD)
# - Checking the dataset using isnull().sum() and duplicated().sum()
df.isnull().sum()
df.duplicated().sum()

# - Drop null/NaN and duplicate data.
df.dropna(inplace=True) # remove the missing value
df.drop_duplicates(inplace=True) # remove the duplicates

# - Drop all 'id', 'address', and 'date' columns.
cols_to_drop = [col for col in df.columns if
                'id' in col.lower() or
                'ip' in col.lower() or
                'date' in col.lower()]
df = df.drop(columns=cols_to_drop) # remove the rows of 'cols_to_drop'
df.head() # shows the top 5 rows

# - Perform feature encoding using LabelEncoder() for categorical features.
categorical_cols = list(df.select_dtypes(include=['object']).columns) # select all columns of type 'object' (categorical)
encoders = {}
for column in categorical_cols:
    label_encoder = LabelEncoder() # create (instantiate) a LabelEncoder object
    df[column] = label_encoder.fit_transform(df[column]) # fit the encoder to the data and simultaneously transform the data
    encoders[column] = label_encoder # save the encoder
df.head() # shows the top 5 rows

# - Cell code provisions
df.columns.tolist()

# B. (SKILLED METHOD) > optional step
# - Handling outlier using drop method
for col in numerical_cols:
    Q1 = df[col].quantile(0.25) # count the Q1 and Q3
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1 # count the Interquartile Range (IQR)
    lower_bound = Q1 - 1.5 * IQR # determine the lower bound and the upper bound
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)] # filter DataFrame
df.describe() # shows the descriptive statistic after removing the outlier

# - Feature scaling using StandardScaler
scaler = StandardScaler()
# Terapkan (fit) scaler ke data dan sekaligus ubah (transform) data tersebut
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df.head() # shows the top 5 rows

# C. (ADVANCED METHOD) > optional step using bining data
col_to_bin = 'CustomerAge'  # specify the numeric columns you want to group (use 'CustomerAge')
new_col_name = 'AgeGroup' # specify the new category of rows
bin_labels = ['Muda', 'Sedang', 'Tua'] # specify the label for 3 group (Muda, Sedang, Tua)
df[new_col_name] = pd.qcut(df[col_to_bin], q=3, labels=bin_labels, duplicates='drop') # use 'pd.qcut' for dividing the data into 3 groups
label_encoder = LabelEncoder() # apply Label Encoding to this new column to convert it to a numeric format.
df[new_col_name] = label_encoder.fit_transform(df[new_col_name])
encoders[new_col_name] = label_encoder # save the encoder and add the new rows to 'categorical_cols'
categorical_cols.extend([new_col_name])
df.head() # show the top 5 of rows

# 11. Building a Clustering Model: KMeans 
# A. (BASIC METHOD)
df_used = df.copy() # make a copy of 'df' to the variable 'df_used'
df_used.describe() # shows the summary of DataFrame 'df'
# create the ElbowVisualizer
model = KMeans() # create the instance model
visualizer = KElbowVisualizer(model, # instantiate KElbowVisualizer and specify the models
                       k=(2,10), # define the range of cluster counts to test (2 to 10)
                       metric='silhouette', # specify the evaluation 'metric' using silhoutte
                       timings=False)
visualizer.fit(df) # run the visualizer
visualizer.show() # show the plot
# use KMeans Clustering
model = KMeans(n_clusters=3, random_state=42) # specify the n_cluster
model.fit(df) # train the models
joblib.dump(model, "model_clustering.h5") # save the training models

# B. (SKILLED METHOD)
# count the silhoutte score
labels = model.labels_
score = silhouette_score(df, labels)
print("Silhouette Score:", score)
# visualize the cluster using PCA in 2D graphic
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df) # fit PCA to the 'df' data and transform the data
df_pca = pd.DataFrame(data=df_pca, columns=['Principal Component 1', 'Principal Component 2']) # create a new DataFrame 'df_pca' from the transformation results
df_pca['Cluster'] = labels # add 'Cluster' rows to 'df_pca' using 'labels'
# make the scatter plot using Seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Cluster',  
    palette=sns.color_palette("viridis", n_colors=2),
    data=df_pca,
    legend="full",
    alpha=0.8
)
plt.title('Visualisasi Cluster dalam 2D (menggunakan PCA)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
centers = pca.transform(model.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X', label='Centroid')
plt.legend()
plt.show()

# C. (ADVANCED METHOD)
# build a model using PCA (without the visualize)
pca = PCA(n_components=2) # instantiate PCA object with 2 component
df_pca_array = pca.fit_transform(df_used) # fit PCA to the 'df_used' data and transform it
data_final = pd.DataFrame(data=df_pca_array, columns=['PCA1', 'PCA2']) # create a new DataFrame from array PCA result
kmeans_pca = KMeans(n_clusters=3, random_state=42) # instantiate a new KMeans
kmeans_pca.fit(data_final) # train (fit) this NEW KMeans model ONLY on 'data_final'
joblib.dump(kmeans_pca, "PCA_model_clustering.h5") # save the models as PCA model

# 12. Displays descriptive analysis—specifically the mean, minimum, and maximum—for numerical features
df_used['Cluster'] = model.labels_ # add a new 'Cluster' column containing 'labels' (the variable from the previous 'model.labels_')
agg_summary = df_used.groupby('Cluster')[numerical_cols].agg(['mean', 'min', 'max']).round(2).T # groupby 'df_used' based on 'Cluster' and count the agregation for 'numerical_cols'
print(agg_summary) # shows the summary result

# 13. Export the data
# A. (BASIC METHOD)
df_used.rename(columns={"Cluster": "Target"}, inplace=True)
df_used.head() # shows the top 5 rows
df_used.to_csv('data_clustering.csv', index=False) # save the data to CSV file

# B. (SKILLED METHOD) > inverse transform the dataset
df_inverse = df_used.copy() # inverse transform the dataset to the normal range for numerical data
df_inverse[numerical_cols] = scaler.inverse_transform(df_inverse[numerical_cols]) # use the 'scaler' to revert 'numerical_cols' to their original values
# reverse the encoding of the dataset back to its original categories.
for column in categorical_cols:
    encoder = encoders[column] # get the appropriate encoder for 'column' from the 'encoders' dictionary
    df_inverse[column] = encoder.inverse_transform(df_inverse[column].astype(int)) # use the scaler to inverse-transform the column
df_inverse.head() # shows the top 5 rows of result

# groupby 'df_inverse' based on 'Target'  and count the agregation for 'numerical_cols'
agg_summary_num = df_inverse.groupby('Target')[numerical_cols].agg(['mean', 'min', 'max']).round(2).T
# groupby 'df_inverse' by 'Target' and calculate the aggregation for 'categorical_cols'
#  calculate the 'mode' aggregation (the most frequently occurring value).
#  use 'lambda x: x.mode()[0]' to retrieve the first mode value
agg_summary_cat = df_inverse.groupby('Target')[categorical_cols].agg(lambda x: x.mode()[0]).round(2).T
print(agg_summary_num) # show the summary of 2 results
print(agg_summary_cat)

# C. (ADVANCED METHOD) > recheck the inverse data from skilled method and save it
df_inverse.head()
df_inverse.to_csv('data_clustering_inverse.csv', index=False) # save the inverse data to CSV file (for building the Classification Method in next model)
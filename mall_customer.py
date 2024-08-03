import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv(r'C:\Users\astuti\Pictures\DATA MINING_CLUSTER\Mall_Customers.csv')

# Rename columns
df.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'Score'
}, inplace=True)

# Drop unnecessary columns
df1 = df.drop(['CustomerID', 'Gender'], axis=1)

# Streamlit app
st.title('Mall Customer Segmentation')

st.header('Dataset Overview')
st.write(df1.head())

# Calculate inertia for different number of clusters
cluster = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i).fit(df1)
    cluster.append(kmeans.inertia_)

# Elbow Method Plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=cluster, ax=ax)
ax.set_title('Elbow Method')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.annotate('Possible elbow point', 
            xy=(3, cluster[2]), 
            xytext=(5, cluster[2] + 5000), 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red'))

st.pyplot(fig)

# Sidebar for cluster selection
st.sidebar.subheader('K-means Clustering')
n_clusters = st.sidebar.slider('Select number of clusters:', 2, 10, 3)

# K-means clustering function
def kmeans_clustering(n_clust):
    kmeans = KMeans(n_clusters=n_clust).fit(df1)
    df1['Labels'] = kmeans.labels_

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter_plot = sns.scatterplot(x='Income', y='Score', hue='Labels', data=df1, palette='viridis')
    plt.xlabel('Income')
    plt.ylabel('Score')
    plt.title('Clustering Results')
    
    # Add annotations
    for label in df1['Labels'].unique():
        cluster_center = df1[df1['Labels'] == label].mean()
        plt.annotate(f'Cluster-{label}', 
                     xy=(cluster_center['Income'], cluster_center['Score']),
                     xytext=(cluster_center['Income'] + 5, cluster_center['Score'] + 5),
                     horizontalalignment='center', size='medium', color='black', weight='semibold',
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    st.pyplot()
    st.write(df1)

kmeans_clustering(n_clusters)

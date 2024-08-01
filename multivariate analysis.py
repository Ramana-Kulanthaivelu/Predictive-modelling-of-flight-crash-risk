import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/Admin/Desktop/Dissertation/avall (1)/reduced_dataset.csv")

# Define categorical columns
#categorical_columns = ['latitude', 'longitude', 'code', 'Occurrence_Code', 'acft_model', 'light_cond', 'eng_type']

# Encode categorical variables
#label_encoders = {}
#for col in categorical_columns:
#    le = LabelEncoder()
#    data[col] = le.fit_transform(data[col].astype(str))
#    label_encoders[col] = le

# Select features for analysis
features = ['crew_age','latitude','longitude','acft_model', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'altimeter','Occurrence_Code'] 
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Plot PCA
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolor='k', s=50)
plt.title('PCA of Flight Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Correlation Analysis
correlation_matrix = pd.DataFrame(X_scaled, columns=features).corr()

# Plot Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Cluster Analysis
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Plot Clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
plt.title('K-means Clustering of Flight Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Discriminant Analysis
lda = LDA(n_components=2)  # Reduce to 2 dimensions for visualization
X_lda = lda.fit_transform(X_scaled, data['Occurrence_Code'])  # Assuming 'Occurrence_Code' is the target

# Check the shape of X_lda
print("Shape of LDA-transformed data:", X_lda.shape)

# Plot LDA
plt.figure(figsize=(10, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=data['Occurrence_Code'], cmap='coolwarm', edgecolor='k', s=50)
plt.title('LDA of Flight Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()

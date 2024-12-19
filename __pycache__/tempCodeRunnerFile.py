
# # Apply KMeans clustering with 5 clusters
# kmeans = KMeans(n_clusters=5, random_state=42)
# df['Cluster'] = kmeans.fit_predict(X_scaled)

# # Add cluster centers back to the original scale
# cluster_centers = scaler_X.inverse_transform(kmeans.cluster_centers_)

# # Sort clusters by ascending order of cluster center values
# sorted_clusters = np.argsort(cluster_centers.flatten())

# # Map clusters to class labels (A-E) based on sorted order
# cluster_labels = {sorted_clusters[i]: chr(65 + i) for i in range(len(sorted_clusters))}
# df['Class'] = df['Cluster'].map(cluster_labels)

# # Sort cluster_centers for correct display order
# sorted_cluster_centers = cluster_centers[sorted_clusters]

# # Take user input for price
# try:
#     y_value = float(predicted_price_rf)  
# except ValueError:
#     exit()

# # Scale the user input
# y_scaled = scaler_X.transform([[y_value]])

# # Predict the cluster for the input price
# predicted_cluster = kmeans.predict(y_scaled)[0]
# predicted_class = cluster_labels[predicted_cluster]

# print(f"The house price {y_value} belongs to Class {predicted_class}.")

# # Plot the data points and cluster centers
# plt.figure(figsize=(12, 6))

# for cluster in range(5):
#     cluster_data = df[df['Cluster'] == sorted_clusters[cluster]]
#     plt.scatter(
#         cluster_data['Price'],
#         [cluster_labels[sorted_clusters[cluster]]] * len(cluster_data),
#         label=f'Class {cluster_labels[sorted_clusters[cluster]]}',
#         alpha=0.7
#     )

# # Plot cluster centers with labels
# for i, center in enumerate(sorted_cluster_centers):
#     plt.scatter(center, cluster_labels[sorted_clusters[i]], color='red', s=200, marker='x')
#     plt.text(center + 0.5, cluster_labels[sorted_clusters[i]], f'Center {cluster_labels[sorted_clusters[i]]}', fontsize=10, color='black')

# # Highlight the input price as a gold point
# plt.scatter(y_value, predicted_class, color='gold', s=150, label=f'house Price ({y_value})', marker='o')

# # Adding labels and improvements
# plt.xlabel('Price')
# plt.ylabel('Class')
# plt.title('KMeans Clustering of Houses by Price (Ascending Order)')
# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()



# ##! display number for house belong that belong to each cluster
# # Plot cluster distribution
# plt.figure(figsize=(8, 6))
# df['Class'].value_counts().sort_index().plot(kind='bar', color='skyblue')
# plt.xlabel('Cluster/Class')
# plt.ylabel('Number of Houses')
# plt.title('Distribution of Houses Across Clusters')
# plt.xticks(rotation=0)
# plt.show()

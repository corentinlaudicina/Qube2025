import os
import numpy as np 

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import DataStructs, EncodeMutations

def visualize_clusters(patients, n_components=2):
    # Only keep patients with valid embeddings and cluster IDs
    valid_patients = [p for p in patients if hasattr(p, "mutation_embedding") 
                      and p.mutation_embedding is not None 
                      and hasattr(p, "cluster_id")]

    if not valid_patients:
        print("No patients with valid mutation embeddings and cluster IDs.")
        return

    embeddings = np.vstack([p.mutation_embedding for p in valid_patients])
    labels = np.array([p.cluster_id for p in valid_patients])

    print("Performing PCA ... ")

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    return reduced 

dir_train = "train"
dir_test  = "test"

patients = []
DataStructs.read_clinical_data(patients, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients)

mutation_matrix, mutation_ids = EncodeMutations.build_binary_mutation_matrix(patients, 
                                                                             min_gene_freq=5
                                                                             )
X_binary = mutation_matrix.values

autoencoder = EncodeMutations.train_autoencoder(X_binary, 
                                                encoding_dim=16, 
                                                epochs=50, 
                                                batch_size=32)

X_cluster = autoencoder.predict(X_binary)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster)

patient_map = {p.ID: p for p in patients}

for i, pid in enumerate(mutation_ids):
    if pid in patient_map:
        patient_map[pid].mutation_embedding = X_cluster[i]
    else:
        print(f"[Warning] ID {pid} not found in patient_map")

n_embedded = sum(1 for p in patients if hasattr(p, "mutation_embedding") and p.mutation_embedding is not None)
print(f"✅ Mutation embeddings assigned to {n_embedded} patients out of {len(patients)}.")

# Assign cluster IDs
for i, pid in enumerate(mutation_ids):
    if pid in patient_map:
        patient_map[pid].cluster_id = cluster_labels[i]
        patient_map[pid].cluster_id = int(cluster_labels[i])
    else:
        print(f"[Warning] ID {pid} not found in patient_map for clustering")

reduced = visualize_clusters(patients, n_components=2)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], 
                      reduced[:, 1],
                      c=[p.cluster_id for p in patients if hasattr(p, "cluster_id")],
                      cmap="tab10", 
                      alpha=0.7
                      )
plt.title("Patient Mutation Embeddings (PCA) Colored by Cluster ID")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.tight_layout()
plt.show()

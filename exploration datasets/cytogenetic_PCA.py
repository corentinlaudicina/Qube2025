import os, sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import DataStructs, EncodeCytogenetics

dir_train = os.path.join(project_root,"train")
dir_test = os.path.join(project_root,"test")

patients_train = []
DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients_train)
DataStructs.parse_cytogenetics(patients_train)

patients_test = []
DataStructs.read_clinical_data(patients_test, os.path.join(dir_test, "clinical_test.csv"))
DataStructs.add_somatic_mutations(patients_test, os.path.join(dir_test, "molecular_test.csv"))
DataStructs.aggregate_mutation_features(patients_test)
DataStructs.parse_cytogenetics(patients_test)

patients_all = patients_train + patients_test

X_cyto_binary, patient_ids, feature_names = EncodeCytogenetics.build_binary_cytogenetic_matrix(patients_all)

print("Shape of binary matrix:", X_cyto_binary.shape)

X_centered = X_cyto_binary - X_cyto_binary.mean(axis=0)  # center
# X_centered = StandardScaler().fit_transform(X_cyto_binary)  # standardize
pca = PCA(n_components=1500)
X_cyto_pca = pca.fit_transform(X_centered)

plt.figure(figsize=(3, 3))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance (%)")
plt.title("Cytogenetics PCA")
plt.grid()
plt.tight_layout()

# plt.savefig(os.path.join(project_root, "report", "Figures", "cytogenetics_pca.png"), dpi=300)

plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras import Input, Model, regularizers
# from tensorflow.keras.layers import Dense, Dropout



# def compare_pca_autoencoder(patients, max_components=32):
#     # === Prepare binary matrix ===
#     X_binary, _, _ = EncodeCytogenetics.build_binary_cytogenetic_matrix(patients)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_binary)

#     # === PCA cumulative explained variance ===
#     pca = PCA(n_components=max_components)
#     pca.fit(X_scaled)
#     pca_var = np.cumsum(pca.explained_variance_ratio_)

#     # === Autoencoder reconstruction-based "explained variance" ===
#     ae_var = []
#     total_variance = np.var(X_scaled)

#     for k in range(1, max_components + 1):
#         encoder, autoencoder = EncodeCytogenetics.train_autoencoder(X_scaled, encoding_dim=k)
#         X_recon = autoencoder.predict(X_scaled)
#         residual_variance = np.mean((X_scaled - X_recon) ** 2)
#         explained = 1.0 - (residual_variance / total_variance)
#         ae_var.append(explained)

#     # === Plot comparison ===
#     plt.figure(figsize=(8, 4))
#     plt.plot(np.arange(1, max_components + 1), pca_var, marker='o', label='PCA (Cumulative)')
#     plt.plot(np.arange(1, max_components + 1), ae_var, marker='x', label='Autoencoder (Reconstruction-based)')
#     plt.axhline(y=0.95, linestyle='--', color='gray', label='95% Threshold')
#     plt.xlabel("Number of Components")
#     plt.ylabel("Explained Variance")
#     plt.title("PCA vs Autoencoder: Explained Variance")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# dir_train = os.path.join(project_root,"train")
# dir_test = os.path.join(project_root,"test")

# patients_train = []
# DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
# DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
# DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
# DataStructs.aggregate_mutation_features(patients_train)
# DataStructs.parse_cytogenetics(patients_train)

# patients_test = []
# DataStructs.read_clinical_data(patients_test, os.path.join(dir_test, "clinical_test.csv"))
# DataStructs.add_somatic_mutations(patients_test, os.path.join(dir_test, "molecular_test.csv"))
# DataStructs.aggregate_mutation_features(patients_test)
# DataStructs.parse_cytogenetics(patients_test)

# patients_all = patients_train + patients_test


# compare_pca_autoencoder(patients_all)
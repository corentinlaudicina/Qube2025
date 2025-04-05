import numpy as np 
from collections import Counter

from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

from sklearn.cluster import KMeans

import DataStructs  

def build_binary_mutation_matrix(patients, min_gene_freq=5):
    """
    Builds a binary matrix (NumPy array) of shape (n_patients, n_genes)
    indicating presence/absence of gene mutations.

    Returns:
        X_binary: np.ndarray of shape (n_patients, n_genes)
        patient_ids: list of patient IDs in order
        gene_list: list of gene names (column order of X_binary)
    """
    # Count gene occurrences across all patients
    gene_counts = {}
    for p in patients:
        for m in p.somatic_mutations:
            gene_counts[m.GENE] = gene_counts.get(m.GENE, 0) + 1

    # Filter genes by frequency
    selected_genes = sorted([g for g, c in gene_counts.items() if c >= min_gene_freq])
    gene_index = {gene: idx for idx, gene in enumerate(selected_genes)}

    # Initialize binary matrix
    n_patients = len(patients)
    n_genes = len(selected_genes)
    X_binary = np.zeros((n_patients, n_genes), dtype=np.uint8)
    patient_ids = []

    # Fill matrix
    for i, p in enumerate(patients):
        patient_ids.append(p.ID)
        gene_set = {m.GENE for m in p.somatic_mutations}
        for gene in gene_set:
            if gene in gene_index:
                X_binary[i, gene_index[gene]] = 1

    return X_binary, patient_ids, selected_genes

def train_autoencoder(X, encoding_dim=16, epochs=50, batch_size=32):
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    encoded = Dropout(0.3)(encoded)  # Dropout for regularization
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(X, X,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=0)

    encoder = Model(input_layer, encoded)
    return encoder, autoencoder

def get_mutation_autoencoder_embedding(patients, encoding_dim=16, epochs=50, batch_size=32):
    mutation_matrix, patient_ids, selected_genes = build_binary_mutation_matrix(patients, min_gene_freq=5)
    X_binary = mutation_matrix

    encoder, _ = train_autoencoder(X_binary, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
    X_embed = encoder.predict(X_binary)

    pid_map = {p.ID: p for p in patients}
    for i, pid in enumerate(patient_ids):
        if pid in pid_map:
            for j in range(encoding_dim):
                setattr(pid_map[pid], f"mut_embed_{j}", float(X_embed[i][j]))

    print(f"Assigned mutation autoencoder embedding to {len(patient_ids)} patients.")
    return [f"mut_embed_{i}" for i in range(encoding_dim)]

def assign_mutation_cluster(patients, encoding_dim=16, epochs=50, batch_size=32, n_clusters=5):

    mutation_matrix, patient_ids, selected_genes = build_binary_mutation_matrix(patients, min_gene_freq=5)
    X_binary = mutation_matrix

    autoencoder = train_autoencoder(X_binary, encoding_dim=encoding_dim,
                                                    epochs=epochs, batch_size=batch_size)
    X_embed = autoencoder.predict(X_binary)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_embed)

    patient_map = {p.ID: p for p in patients}
    for i, pid in enumerate(patient_ids):
        if pid in patient_map:
            cluster = cluster_labels[i]
            patient_map[pid].cluster_id = int(cluster)
            for c in range(n_clusters):
                setattr(patient_map[pid], f"cluster_{c}", int(c == cluster))
        else:
            print(f"[Warning] ID {pid} not found")

    print(f"One-hot cluster features assigned to {len(patient_ids)} patients.")
    return [f"mutation_cluster_{i}" for i in range(n_clusters)]

def build_top_gene_features(patients, top_n=50):
    # Count gene frequencies
    gene_list = []
    for p in patients:
        for mut in p.somatic_mutations:
            gene_list.append(mut.GENE)
    top_genes = [g for g, _ in Counter(gene_list).most_common(top_n)]
    feature_names = sorted(top_genes)

    patient_ids = []
    data = []
    for p in patients:
        row = [0] * len(feature_names)
        genes = {mut.GENE for mut in p.somatic_mutations}
        for i, gene in enumerate(feature_names):
            if gene in genes:
                row[i] = 1
        data.append(row)
        patient_ids.append(p.ID)

    X = np.array(data, dtype=np.uint8)
    return X, patient_ids, feature_names

def build_top_mutation_features(patients, top_n=5):
    # Count full mutation keys
    key_list = []
    for p in patients:
        for mut in p.somatic_mutations:
            key = f"{mut.GENE}:{mut.PROTEIN_CHANGE}"
            key_list.append(key)
    top_keys = [k for k, _ in Counter(key_list).most_common(top_n)]
    feature_names = sorted(top_keys)

    patient_ids = []
    data = []
    for p in patients:
        row = [0] * len(feature_names)
        keys = {f"{mut.GENE}:{mut.PROTEIN_CHANGE}" for mut in p.somatic_mutations}
        for i, key in enumerate(feature_names):
            if key in keys:
                row[i] = 1
        data.append(row)
        patient_ids.append(p.ID)

    X = np.array(data, dtype=np.uint8)
    return X, patient_ids, feature_names

# dir_train = "train"

# patients_train = []
# DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
# DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
# DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
# DataStructs.aggregate_mutation_features(patients_train)


# df_binary, patient_ids = build_binary_mutation_matrix(patients_train, min_gene_freq=5)

# encoder = train_autoencoder(df_binary.values, encoding_dim=16, epochs=100, batch_size=32)

# X_mutation_embed = encoder.predict(df_binary)  # shape = (n_patients, 16)
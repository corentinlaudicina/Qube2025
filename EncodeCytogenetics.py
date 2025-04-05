from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.cluster import KMeans
import numpy as np

def build_binary_cytogenetic_matrix(patients):
    """
    Builds a binary matrix (NumPy array) of shape (n_patients, n_features)
    indicating presence/absence of cytogenetic features.

    Returns:
        X_binary: np.ndarray (n_patients × n_features)
        patient_ids: list of patient IDs
        feature_names: list of all unique cytogenetic features (column order)
    """
    all_features_set = set()
    patient_feature_sets = []

    # Collect all features
    for p in patients:
        patient_features = set()
        for obs in p.CYTOGENETICS:
            patient_features.update(obs.features)
        patient_feature_sets.append(patient_features)
        all_features_set.update(patient_features)

    # Index mapping
    feature_names = sorted(all_features_set)
    feature_index = {feat: idx for idx, feat in enumerate(feature_names)}

    # Initialize binary matrix
    n_patients = len(patients)
    n_features = len(feature_names)
    X_binary = np.zeros((n_patients, n_features), dtype=np.uint8)
    patient_ids = [p.ID for p in patients]

    for i, feat_set in enumerate(patient_feature_sets):
        for feat in feat_set:
            j = feature_index[feat]
            X_binary[i, j] = 1

    return X_binary, patient_ids, feature_names

def train_autoencoder(X, encoding_dim=16, epochs=50, batch_size=32):
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    encoder = Model(input_layer, encoded)
    return encoder, autoencoder

def get_cytogenetic_autoencoder_embedding(patients, encoding_dim=16, epochs=50, batch_size=32):
    """
    Generates autoencoder-based embeddings for cytogenetic data and assigns them
    as individual attributes (cyto_embed_0, cyto_embed_1, ...) to each patient.

    Returns:
        feature_names: list of strings corresponding to feature names.
    """
    # Get binary matrix and corresponding patient IDs
    X_binary, patient_ids, feature_names = build_binary_cytogenetic_matrix(patients)

    # Train autoencoder
    encoder, _ = train_autoencoder(X_binary, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
    X_embed = encoder.predict(X_binary)

    # Map embeddings back to patients
    id_to_patient = {p.ID: p for p in patients}
    assigned_count = 0

    for i, pid in enumerate(patient_ids):
        patient = id_to_patient.get(pid)
        if patient is not None:
            for j in range(encoding_dim):
                setattr(patient, f"cyto_embed_{j}", float(X_embed[i, j]))
            assigned_count += 1
        else:
            print(f"[Warning] Patient ID {pid} not found in patient list")

    print(f"Assigned cytogenetic autoencoder embeddings to {assigned_count} patients.")
    return [f"cyto_embed_{i}" for i in range(encoding_dim)]

def assign_cytogenetic_cluster(patients, encoding_dim=16, epochs=50, batch_size=32, n_clusters=5):
    """
    Assigns cluster IDs to the patients based on their cytogenetics using autoencoder + KMeans.
    """
    X_binary, _ = build_binary_cytogenetic_matrix(patients)

    encoder = train_autoencoder(X_binary, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
    X_embed = encoder.predict(X_binary)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_embed)

    for p, cluster in zip(patients, cluster_labels):
        p.cytogenetic_cluster_id = int(cluster)
        for c in range(n_clusters):
            setattr(p, f"cytogenetic_cluster_{c}", int(c == cluster))

    print(f"Assigned {n_clusters} cytogenetic clusters to {len(patients)} patients.")
    return [f"cytogenetic_cluster_{i}" for i in range(n_clusters)]

# # Sample usage:

# import os, sys 
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import DataStructs

# dir_train = os.path.join(project_root,"train")
# dir_train = "train"
# patients = []
# DataStructs.read_clinical_data(patients, os.path.join(dir_train, "clinical_train.csv"))
# DataStructs.add_status(patients, os.path.join(dir_train, "target_train.csv"))
# DataStructs.add_somatic_mutations(patients, os.path.join(dir_train, "molecular_train.csv"))
# DataStructs.aggregate_mutation_features(patients)

# cyto_features = assign_cytogenetic_cluster(patients)
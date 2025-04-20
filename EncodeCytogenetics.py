from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.cluster import KMeans
import numpy as np

from autoencoder import train_autoencoder

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

    for p in patients:
        patient_features = set()
        for obs in p.CYTOGENETICS:
            patient_features.update(obs.features)
        patient_feature_sets.append(patient_features)
        all_features_set.update(patient_features)

    feature_names = sorted(all_features_set)
    feature_index = {feat: idx for idx, feat in enumerate(feature_names)}

    n_patients = len(patients)
    n_features = len(feature_names)
    X_binary = np.zeros((n_patients, n_features), dtype=np.uint8)
    patient_ids = [p.ID for p in patients]

    for i, feat_set in enumerate(patient_feature_sets):
        for feat in feat_set:
            j = feature_index[feat]
            X_binary[i, j] = 1

    return X_binary, patient_ids, feature_names

def build_cyto_binary_matrix(patients):
    X_bin, patient_ids, _ = build_binary_cytogenetic_matrix(patients)
    return X_bin, patient_ids

from sklearn.feature_extraction.text import TfidfVectorizer

def build_cytogenetic_tfidf_matrix(patients, max_features=500):
    """
    Computes TF-IDF matrix from patient cytogenetic strings.

    Args:
        patients: list of Patient objects, each with a CYTOGENETICS attribute (list of CytogeneticObservation)
        max_features: limit on vocabulary size (optional)

    Returns:
        X_tfidf: NumPy array of shape (n_patients, n_features)
        feature_names: list of selected cytogenetic tokens (columns)
        patient_ids: list of patient IDs
    """

    cyto_strings = []
    patient_ids = []
    for p in patients:
        obs_strings = []
        for obs in p.CYTOGENETICS:
            obs_strings.extend(obs.features)
        cyto_strings.append(" ".join(obs_strings))  
        patient_ids.append(p.ID)

    vectorizer = TfidfVectorizer(lowercase=True, max_features=max_features)
    X_tfidf = vectorizer.fit_transform(cyto_strings).toarray()
    feature_names = vectorizer.get_feature_names_out()

    return X_tfidf, feature_names, patient_ids

def build_cyto_tfidf_matrix(patients, max_features=500):
    X_tfidf, _, patient_ids = build_cytogenetic_tfidf_matrix(patients, max_features=max_features)
    return X_tfidf, patient_ids

def get_cytogenetic_embedding(patients, 
                                   matrix_builder_fn, 
                                   embed_prefix="embed", 
                                   encoding_dim=16, 
                                   epochs=50, 
                                   batch_size=32):
    """
    Generic function for assigning autoencoder embeddings to patients.

    Args:
        patients: list of Patient objects
        matrix_builder_fn: function that returns (X, patient_ids)
        embed_prefix: prefix for feature names (e.g., "cyto_embed")
        encoding_dim: dimensionality of the embedding
        epochs: training epochs
        batch_size: batch size for training

    Returns:
        List of assigned feature names (e.g. ['cyto_embed_0', ..., 'cyto_embed_{k}'])
    """
    X, patient_ids = matrix_builder_fn(patients)

    encoder, _ = train_autoencoder(X, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
    X_embed = encoder.predict(X)

    pid_map = {p.ID: p for p in patients}
    for i, pid in enumerate(patient_ids):
        patient = pid_map.get(pid)
        if patient:
            for j in range(encoding_dim):
                setattr(patient, f"{embed_prefix}_{j}", float(X_embed[i, j]))


    cyto_feat_names = [f"{embed_prefix}_{j}" for j in range(encoding_dim)]
    print(f"Assigned {embed_prefix} embeddings to {len(patient_ids)} patients.")
    return cyto_feat_names, X_embed, patient_ids

# def get_cytogenetic_autoencoder_embedding(patients, encoding_dim=16, epochs=50, batch_size=32):
#     """
#     Generates autoencoder-based embeddings for cytogenetic data and assigns them
#     as individual attributes (cyto_embed_0, cyto_embed_1, ...) to each patient.

#     Returns:
#         feature_names: list of strings corresponding to feature names.
#     """
#     # Get binary matrix and corresponding patient IDs
#     X_binary, patient_ids, feature_names = build_binary_cytogenetic_matrix(patients)

#     # Train autoencoder
#     encoder, _ = train_autoencoder(X_binary, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
#     X_embed = encoder.predict(X_binary)

#     # Map embeddings back to patients
#     id_to_patient = {p.ID: p for p in patients}
#     assigned_count = 0

#     for i, pid in enumerate(patient_ids):
#         patient = id_to_patient.get(pid)
#         if patient is not None:
#             for j in range(encoding_dim):
#                 setattr(patient, f"cyto_embed_{j}", float(X_embed[i, j]))
#             assigned_count += 1
#         else:
#             print(f"[Warning] Patient ID {pid} not found in patient list")

#     print(f"Assigned cytogenetic autoencoder embeddings to {assigned_count} patients.")
#     return [f"cyto_embed_{i}" for i in range(encoding_dim)]

# def get_cytogenetic_autoencoder_embedding_tfidf(patients, encoding_dim=16, epochs=50, batch_size=32, max_features=500):
#     from sklearn.feature_extraction.text import TfidfVectorizer

#     # --- Step 1: Build cytogenetic TF-IDF matrix ---
#     cyto_strings = []
#     patient_ids = []
#     for p in patients:
#         obs_strings = []
#         for obs in p.CYTOGENETICS:
#             obs_strings.extend(obs.features)
#         cyto_strings.append(" ".join(obs_strings))
#         patient_ids.append(p.ID)

#     vectorizer = TfidfVectorizer(lowercase=True, max_features=max_features)
#     X_tfidf = vectorizer.fit_transform(cyto_strings).toarray()

#     # --- Step 2: Autoencode TF-IDF features ---
#     encoder, _ = train_autoencoder(X_tfidf, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
#     X_embed = encoder.predict(X_tfidf)

#     # --- Step 3: Assign to Patient objects ---
#     pid_map = {p.ID: p for p in patients}
#     for i, pid in enumerate(patient_ids):
#         if pid in pid_map:
#             for j in range(encoding_dim):
#                 setattr(pid_map[pid], f"cyto_embed_{j}", float(X_embed[i][j]))

#     print(f"Assigned cytogenetic TF-IDF embeddings to {len(patient_ids)} patients.")
#     return [f"cyto_embed_{j}" for j in range(encoding_dim)]

# def assign_cytogenetic_cluster(patients, encoding_dim=16, epochs=50, batch_size=32, n_clusters=5):
#     """
#     Assigns cluster IDs to the patients based on their cytogenetics using autoencoder + KMeans.
#     """
#     X_binary, _ = build_binary_cytogenetic_matrix(patients)

#     encoder = train_autoencoder(X_binary, encoding_dim=encoding_dim, epochs=epochs, batch_size=batch_size)
#     X_embed = encoder.predict(X_binary)

#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(X_embed)

#     for p, cluster in zip(patients, cluster_labels):
#         p.cytogenetic_cluster_id = int(cluster)
#         for c in range(n_clusters):
#             setattr(p, f"cytogenetic_cluster_{c}", int(c == cluster))

#     print(f"Assigned {n_clusters} cytogenetic clusters to {len(patients)} patients.")
#     return [f"cytogenetic_cluster_{i}" for i in range(n_clusters)]


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



# # PICKLE 

# import os, sys 
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import DataStructs
# import pickle 

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

# cyto_feat_names, X_cyto_embed, cyto_ids = get_cytogenetic_embedding(patients_all, 
#                                                     build_cyto_tfidf_matrix, 
#                                                     embed_prefix="embed_cyto", 
#                                                     encoding_dim=8, 
#                                                     epochs=50, 
#                                                     batch_size=16
#                                                     )

# with open(os.path.join("encoded_data", "cytogenetic_embeddings.pkl"), "wb") as f:
#     pickle.dump({"ids": cyto_ids, "features": cyto_feat_names, "embeddings": X_cyto_embed}, f)

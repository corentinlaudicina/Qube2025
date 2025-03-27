import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import defaultdict
import numpy as np 
from sksurv.metrics import concordance_index_ipcw

import DataStructs, EncodeMutations, LightGBM_model

def train_stratified_lightgbm(patients, features, cluster_attr="cluster_id", tau=7, test_size=0.3, seed=42):
    """
    Trains separate LightGBM survival models for each subgroup (e.g., cluster).
    Assumes each Patient object has an attribute cluster_attr (e.g., "cluster_id").

    Returns:
        results: list of tuples (cluster_value, n_patients, test_cindex)
        models_by_cluster: dict mapping cluster -> trained model
    """
    clusters = defaultdict(list)
    for p in patients:
        if hasattr(p, cluster_attr):
            cluster_val = getattr(p, cluster_attr)
            clusters[cluster_val].append(p)

    results = []
    models_by_cluster = {}  # 🔧 Now using a dictionary

    for cluster, subgroup in sorted(clusters.items()):
        print(f"\n🧬 Training model for {cluster_attr} = {cluster} ({len(subgroup)} patients)")

        if len(subgroup) < 50:
            print("⚠️ Skipping: too few patients.")
            continue

        # Train model on this subgroup
        ci_test, model = LightGBM_model.train_lightgbm_survival_model(
                                                                    patients=subgroup,
                                                                    features=features,
                                                                    tau=tau,
                                                                    test_size=test_size,
                                                                    seed=seed,
                                                                    lgbm_params=None
                                                                )

        results.append((cluster, len(subgroup), round(ci_test, 4)))
        models_by_cluster[cluster] = model  # ✅ Store model with cluster as key

    print("\n📊 Stratified C-index Summary:")
    for cluster, n, ci in results:
        print(f"{cluster_attr} = {cluster} | n = {n} | C-index = {ci:.4f}")

    return results, models_by_cluster


def meta_ensemble_predict(models_by_cluster, X_test, y_test, test_cluster_ids, y_train, tau=7):
    """
    For each test sample, use the model corresponding to its cluster to predict.
    """
    pred_all = []
    y_test_all = []

    for cluster_id in sorted(models_by_cluster.keys()):
        model = models_by_cluster[cluster_id]
        indices = [i for i, cid in enumerate(test_cluster_ids) if cid == cluster_id]

        if not indices:
            continue  # Skip empty clusters

        X_sub = X_test[indices]
        y_sub = y_test[indices]

        preds = -model.predict(X_sub)  # Use negative risk score for C-index

        pred_all.extend(preds)
        y_test_all.extend(y_sub)

    y_test_all = np.array(y_test_all, dtype=[('event', '?'), ('time', '<f8')])
    pred_all = np.array(pred_all)

    global_ci = concordance_index_ipcw(y_train, y_test_all, pred_all, tau=tau)[0]
    print(f"\n📈 Meta-Ensemble Global C-index: {global_ci:.4f}")
    return global_ci

dir_train = os.path.join(project_root,"train")
dir_test  = os.path.join(project_root,"test")

base_features = [
                'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
                'mutation_count', 'mean_VAF', 'max_VAF',
                'has_TP53_mutation'
                ]
                # , 'log_WBC', 'anc_monocyte_ratio',
                # 'plt_hb_ratio'
                # ]

patients_train = []
DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients_train)

gene_df, gene_ids, gene_cols = EncodeMutations.build_top_gene_features(patients_train, top_n=10)

gene_map = gene_df.to_dict(orient="index")
for p in patients_train:
    if p.ID in gene_map:
        for key, val in gene_map[p.ID].items():
            setattr(p, f"mut_{key}", val)

cluster_features = EncodeMutations.assign_mutation_cluster(patients_train)

features = base_features + [f"mut_{g}" for g in gene_cols] + cluster_features

results, models_by_cluster = train_stratified_lightgbm(patients_train, 
                                    features, 
                                    cluster_attr="cluster_id")

X_full, y_numeric, y_structured, id_list = DataStructs.build_training_data(patients_train, features)
X_train, X_test, y_train, y_test, train_idx, test_idx = DataStructs.split_train_test(X_full, y_structured, id_list, test_size=0.3, seed=42)

train_ids = [id_list[i] for i in train_idx]
DataStructs.impute_median_train_test(X_train, X_test, patient_ids_train=train_ids)

test_patients = [patients_train[i] for i in test_idx]
test_cluster_ids = [getattr(p, "cluster_id", None) for p in test_patients]

meta_ensemble_predict(models_by_cluster, X_test, y_test, test_cluster_ids, y_train, tau=7)
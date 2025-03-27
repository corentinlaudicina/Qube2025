import os, sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

import DataStructs, EncodeMutations

def train_cox_survival_model(patients, features, tau=7, test_size=0.3, seed=42, lgbm_params=None):

    # --- Build & treat dataset ---
    X_full, y_numeric, y_full, id_list = DataStructs.build_training_data(patients, features)

    X_train, X_test, y_train, y_test, train_idx, test_idx = DataStructs.split_train_test(
        X_full, y_full, id_list, test_size=test_size, seed=seed
    )

    patients_test = [patients[i] for i in test_idx]

    id_train = [id_list[i] for i in train_idx]

    medians, imputed_ids = DataStructs.impute_median_train_test(X_train, X_test, patient_ids_train=id_train)

    # --- Variance filtering ---
    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train)
    X_test  = selector.transform(X_test)

    # --- Standardization ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- Train Cox Proportional Hazards ---

    cox = CoxPHSurvivalAnalysis(alpha=1e-4)
    cox.fit(X_train, y_train)

    # Evaluate the model using Concordance Index IPCW
    cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
    cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
    print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.4f}")
    print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.4f}")

    return cox_cindex_test, cox

#### MAIN ####

dir_train = os.path.join(project_root,"train")
dir_test  = os.path.join(project_root,"test")

base_features = [
                'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
                'mutation_count', 'mean_VAF', 'max_VAF', 'has_TP53_mutation', 
                ]

patients_train = []
DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients_train)

gene_df, gene_ids, gene_cols = EncodeMutations.build_top_gene_features(patients_train, top_n=10)

# Attach to patients
gene_map = gene_df.to_dict(orient="index")
for p in patients_train:
    if p.ID in gene_map:
        for key, val in gene_map[p.ID].items():
            setattr(p, f"mut_{key}", val)

cluster_features = EncodeMutations.assign_mutation_cluster(patients_train)

features = base_features + [f"mut_{g}" for g in gene_cols] + cluster_features

train_cox_survival_model(
                         patients_train, 
                         features, 
                         tau=7, 
                         test_size=0.3, 
                         seed=42
                        )

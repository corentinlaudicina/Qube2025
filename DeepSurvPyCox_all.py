import os, sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from sklearn.impute import SimpleImputer

import DataStructs, autoencoder

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torchtuples as tt
import torch 

from sklearn.preprocessing import StandardScaler

def build_full_patient_matrix(patients, clinical_features):
    """
    Builds a full feature matrix per patient by combining:
    - Clinical features
    - Cytogenetic binary features (p.CYTOGENETICS[i].features)
    - Mutation binary features (GENE:PROTEIN_CHANGE)
    
    Returns:
        X: Numpy matrix (n_patients x n_features)
        feature_names: List of all column names
        patient_ids: List of patient IDs
    """

    # Gather all cyto and mutation features across patients
    all_cyto_feats = set()
    all_mut_feats = set()
    for p in patients:
        for obs in p.CYTOGENETICS:
            all_cyto_feats.update(obs.features)
        for mut in p.somatic_mutations:
            key = f"{mut.GENE}:{mut.PROTEIN_CHANGE}"
            all_mut_feats.add(key)

    all_cyto_feats = sorted(all_cyto_feats)
    all_mut_feats = sorted(all_mut_feats)

    # Create full feature list
    feature_names = clinical_features + all_cyto_feats + all_mut_feats
    X = []
    patient_ids = []

    for p in patients:
        row = []

        # Clinical features (numerical)
        for f in clinical_features:
            val = getattr(p, f, None)
            row.append(float(val) if val is not None else np.nan)

        # Cytogenetic features (binary)
        patient_cyto_feats = set()
        for obs in p.CYTOGENETICS:
            patient_cyto_feats.update(obs.features)
        for feat in all_cyto_feats:
            row.append(1.0 if feat in patient_cyto_feats else 0.0)

        # Mutation features (binary)
        patient_mut_feats = {f"{mut.GENE}:{mut.PROTEIN_CHANGE}" for mut in p.somatic_mutations}
        for feat in all_mut_feats:
            row.append(1.0 if feat in patient_mut_feats else 0.0)

        X.append(row)
        patient_ids.append(p.ID)

    return np.array(X), feature_names, patient_ids

dir_train = os.path.join(project_root,"train")
dir_test  = os.path.join(project_root,"test")

base_features = [
                'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
                'mutation_count', 'mean_VAF', 'has_TP53_mutation', 
                'is_normal_karyotype', 'abnormality_count', 'has_monosomy_7', 'has_del_5q'
                ]

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

X_full, full_feature_names, patient_ids = build_full_patient_matrix(patients_all, base_features)

imputer = SimpleImputer(strategy="median")
X_full_imputed = imputer.fit_transform(X_full)

encoder, _ = autoencoder.train_autoencoder(X_full_imputed, encoding_dim=32)

X_embed = encoder.predict(X_full_imputed)

for i, p in enumerate(patients_all):
    for j in range(X_embed.shape[1]):
        setattr(p, f"embed_{j}", float(X_embed[i][j]))

embed_features = [f"embed_{j}" for j in range(X_embed.shape[1])]
features = embed_features

X_full, y_numeric, y_full, id_list = DataStructs.build_training_data(patients_train, features)

test_size = 0.33
seed = 42 
X_train, X_test, y_train, y_test, train_idx, test_idx = DataStructs.split_train_test(
    X_full, y_full, id_list, test_size=test_size, seed=seed
)

_ , _ = DataStructs.impute_median_train_test(X_train, X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

T_train = np.log1p(y_train['time']).astype(np.float32)
E_train = y_train['event'].astype(int)

T_test = np.log1p(y_test['time']).astype(np.float32)
E_test = y_test['event'].astype(int)

y_train_pyc = (T_train, E_train)

net = tt.practical.MLPVanilla(
                            in_features=X_train.shape[1],
                            out_features=1,
                            num_nodes=[32, 16],
                            activation=torch.nn.ReLU,
                            dropout=0.3
                            )

model = CoxPH(net, tt.optim.Adam(lr=1e-4))

model.fit(X_train_scaled, y_train_pyc, batch_size=16, epochs=100, val_data=(X_test_scaled, (T_test, E_test)))
model.compute_baseline_hazards()

surv_test = model.predict_surv_df(X_test_scaled)
surv_train = model.predict_surv_df(X_train_scaled)

ev_train = EvalSurv(surv_train, T_train, E_train, censor_surv='km')
print("C-index train:", ev_train.concordance_td())

ev_test = EvalSurv(surv_test, T_test, E_test, censor_surv='km')
print("C-index test:", ev_test.concordance_td())

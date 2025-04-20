import os, sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torchtuples as tt
import torch 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import DataStructs, EncodeMutations, EncodeCytogenetics

dir_train = os.path.join(project_root,"train")
dir_test = os.path.join(project_root,"test")

base_features = [
                'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
                'mutation_count', 'mean_VAF', 
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

import pickle
pkl_path = os.path.join("encoded_data", "mutation_embeddings.pkl")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)
mut_ids = data["ids"]
mut_feat_names = data["features"]
mutation_embed_features_all = data["embeddings"]
# mutation_embed_features_all, _ , _ = EncodeMutations.get_mutations_embedding(patients_all, 
#                                                                       EncodeMutations.build_mut_tfidf_matrix, 
#                                                                       embed_prefix="embed_mut_", 
#                                                                       encoding_dim=8, 
#                                                                       epochs=20, 
#                                                                       batch_size=32
#                                                                       )

pkl_path = os.path.join("encoded_data", "cytogenetic_embeddings.pkl")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)
cyto_ids = data["ids"]
cyto_feat_names = data["features"]
cyto_embed_features_all = data["embeddings"]
# cyto_embed_features_all = EncodeCytogenetics.get_cytogenetic_embedding(patients_all, 
#                                                                        EncodeCytogenetics.build_cyto_tfidf_matrix, 
#                                                                        embed_prefix="embed_cyto_", 
#                                                                        encoding_dim=8, 
#                                                                        epochs=20, 
#                                                                        batch_size=32
#                                                                        )

# Assign mutation embeddings
for i, pid in enumerate(mut_ids):
    patient = next((p for p in patients_all if p.ID == pid), None)
    if patient:
        for j in range(mutation_embed_features_all.shape[1]):
            setattr(patient, f"mut_embed_{j}", float(mutation_embed_features_all[i, j]))

# Assign cytogenetic embeddings
for i, pid in enumerate(cyto_ids):
    patient = next((p for p in patients_all if p.ID == pid), None)
    if patient:
        for j in range(cyto_embed_features_all.shape[1]):
            setattr(patient, f"cyto_embed_{j}", float(cyto_embed_features_all[i, j]))

# Feature list
mut_feat_names = [f"mut_embed_{j}" for j in range(mutation_embed_features_all.shape[1])]
cyto_feat_names = [f"cyto_embed_{j}" for j in range(cyto_embed_features_all.shape[1])]

features = base_features + mut_feat_names + cyto_feat_names

X_full, y_numeric, y_full, id_list = DataStructs.build_training_data(patients_train, features)

test_size = 0.33
seed = 314159 
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
                            num_nodes=[64, 64, 32],
                            activation=torch.nn.ReLU,
                            dropout=0.4
                            )
optimizer = tt.optim.Adam(lr=1e-4, weight_decay=1e-4)

model = CoxPH(net, optimizer)

model.fit(X_train_scaled, y_train_pyc, batch_size=16, epochs=100, val_data=(X_test_scaled, (T_test, E_test)))
model.compute_baseline_hazards()

surv_test = model.predict_surv_df(X_test_scaled)
surv_train = model.predict_surv_df(X_train_scaled)

ev_train = EvalSurv(surv_train, T_train, E_train, censor_surv='km')
print("C-index train:", ev_train.concordance_td())

ev_test = EvalSurv(surv_test, T_test, E_test, censor_surv='km')
print("C-index test:", ev_test.concordance_td())

# Inference on test data 

# X_test, _, _, test_ids = DataStructs.build_training_data(patients_test, 
#                                                         features, 
#                                                         skip_missing_time=False
#                                                         )
# _, _ = DataStructs.impute_median_train_test(X_train, X_test)
# X_test_scaled = scaler.transform(X_test).astype(np.float32)
# risk_scores = model.predict(X_test_scaled).flatten()
# print(np.min(risk_scores), np.max(risk_scores), np.mean(risk_scores))
# submission_array = np.column_stack((np.array(test_ids), risk_scores))
# with open("submission.csv", "w") as f:
#     f.write("ID,risk_score\n")
#     for row in submission_array:
#         f.write(f"{row[0]},{row[1]}\n")
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

import DataStructs, EncodeMutations, EncodeCytogenetics

dir_train = os.path.join(project_root,"train")
dir_test = os.path.join(project_root,"test")

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

mutation_embed_features_all = EncodeMutations.get_mutation_autoencoder_embedding(patients_all, encoding_dim=4)

# cyto_embed_features_all = EncodeCytogenetics.get_cytogenetic_autoencoder_embedding(patients_all, encoding_dim=4)

# mutation_cluster_features = EncodeMutations.assign_mutation_cluster(patients_train, encoding_dim = 8)
# cyto_features = EncodeCytogenetics.assign_cytogenetic_cluster(patients_train, encoding_dim = 8)  

# mutation_embed_features = EncodeMutations.get_mutation_autoencoder_embedding(patients_train, encoding_dim = 4)
# cyto_embed_features = EncodeCytogenetics.get_cytogenetic_autoencoder_embedding(patients_train, encoding_dim = 4)

features = base_features + mutation_embed_features_all #+ cyto_embed_features_all

## --- Build dataset from training data---
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
    num_nodes=[16, 16],
    activation=torch.nn.ReLU,
    dropout=0.2
)

model = CoxPH(net, tt.optim.Adam(lr=1e-4))

model.fit(X_train_scaled, y_train_pyc, batch_size=64, epochs=300, val_data=(X_test_scaled, (T_test, E_test)))
model.compute_baseline_hazards()

surv_test = model.predict_surv_df(X_test_scaled)
surv_train = model.predict_surv_df(X_train_scaled)

ev_train = EvalSurv(surv_train, T_train, E_train, censor_surv='km')
print("C-index train:", ev_train.concordance_td())

ev_test = EvalSurv(surv_test, T_test, E_test, censor_surv='km')
print("C-index test:", ev_test.concordance_td())


## Inference on test data 

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
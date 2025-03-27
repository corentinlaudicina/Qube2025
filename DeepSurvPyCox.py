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

base_features = [
                'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
                'mutation_count', 'mean_VAF', 'max_VAF', 'has_TP53_mutation', 
                ]

patients_train = []
DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients_train)

# mutation_cluster_features = EncodeMutations.assign_mutation_cluster(patients_train, encoding_dim = 8)
# cyto_features = EncodeCytogenetics.assign_cytogenetic_cluster(patients_train, encoding_dim = 8)  

mutation_embed_features = EncodeMutations.get_mutation_autoencoder_embedding(patients_train, encoding_dim = 4)
cyto_embed_features = EncodeCytogenetics.get_cytogenetic_autoencoder_embedding(patients_train, encoding_dim = 4)

features = base_features + mutation_embed_features + cyto_embed_features

# # --- Build & treat dataset ---
X_full, y_numeric, y_full, id_list = DataStructs.build_training_data(patients_train, features)

test_size=0.1
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


surv = model.predict_surv_df(X_test_scaled)
ev = EvalSurv(surv, T_test, E_test, censor_surv='km')
print("C-index:", ev.concordance_td())

# import pandas as pd

# dir_test = os.path.join(project_root, "test")

# # --- Load & preprocess test patients ---
# patients_test = []
# DataStructs.read_clinical_data(patients_test, os.path.join(dir_test, "clinical_test.csv"))
# DataStructs.add_somatic_mutations(patients_test, os.path.join(dir_test, "molecular_test.csv"))
# DataStructs.aggregate_mutation_features(patients_test)

# # --- Apply encodings ---
# mutation_embed_features = EncodeMutations.get_mutation_autoencoder_embedding(patients_test, encoding_dim=4)
# cyto_embed_features = EncodeCytogenetics.get_cytogenetic_autoencoder_embedding(patients_test, encoding_dim=4)

# # --- Ensure same feature list as used during training ---
# features = base_features + mutation_embed_features + cyto_embed_features

# # --- Build test feature matrix ---
# X_test_submission, _, _, test_ids = DataStructs.build_training_data(patients_test, 
#                                                                     features, 
#                                                                     skip_missing_target=False
#                                                                     )
# X_test_submission = scaler.transform(X_test_submission)
# X_test_submission = X_test_submission.astype(np.float32)

# # --- Predict risk scores ---
# risk_scores = -model.predict(X_test_submission).flatten()  # Negative for risk -> survival logic

# # --- Create submission file ---
# submission = pd.Series(risk_scores, index=test_ids, name='risk_score')
# submission.to_csv("submission.csv", index_label="ID")
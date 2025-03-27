import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import lightgbm as lgb

from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

import DataStructs, EncodeMutations, EncodeCytogenetics

def train_lightgbm_survival_model(patients, features, tau=7, test_size=0.3, seed=42, lgbm_params=None):

    # --- Build & treat dataset ---
    X_full, y_numeric, y_full, id_list = DataStructs.build_training_data(patients, features)

    X_train, X_test, y_train, y_test, train_idx, test_idx = DataStructs.split_train_test(
        X_full, y_full, id_list, test_size=test_size, seed=seed
    )

    patients_test = [patients[i] for i in test_idx]

    id_train = [id_list[i] for i in train_idx]

    medians, imputed_ids = DataStructs.impute_median_train_test(X_train, X_test, patient_ids_train=id_train)

    # --- Train LightGBM ---
    lgbm_params = lgbm_params or {
        'objective': 'regression',
        'learning_rate': 0.1,
        'max_depth': 3,
        'num_leaves': 15,
        'feature_fraction': 0.8, 
        'bagging_fraction': 0.8,
        'metric': 'l2',
        'verbose': -1
        }
    train_dataset = lgb.Dataset(X_train, label=y_train['time'])
    test_dataset  = lgb.Dataset(X_test,  label=y_test['time'], reference=train_dataset)

    model = lgb.train(
        lgbm_params,
        train_set=train_dataset,
        valid_sets=[train_dataset, test_dataset],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    pred_train = -model.predict(X_train)
    pred_test  = -model.predict(X_test)

    ci_train = concordance_index_ipcw(y_train, y_train, pred_train, tau=tau)[0]
    ci_test  = concordance_index_ipcw(y_train, y_test,  pred_test,  tau=tau)[0]

    print(f"LightGBM Survival Model Concordance Index IPCW on train: {ci_train:.4f}")
    print(f"LightGBM Survival Model Concordance Index IPCW on test: {ci_test:.4f}")


    # # Initialize and train the Cox Proportional Hazards model
    # cox = CoxPHSurvivalAnalysis(alpha=1e-4)
    # cox.fit(X_train, y_train)

    # # Evaluate the model using Concordance Index IPCW
    # cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
    # cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
    # print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.2f}")
    # print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.2f}")

    return ci_test, model

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


cyto_features = EncodeCytogenetics.assign_cytogenetic_cluster(patients_train)  

features = base_features + [f"mut_{g}" for g in gene_cols] + cyto_features

train_lightgbm_survival_model(patients_train, 
                              features, 
                              tau=7, 
                              test_size=0.3, 
                              seed=42
                              )


# from itertools import product
# import random 

# def random_lgbm_search(patients_train, base_features, trials=20, tau=7, seed=42):

#     random.seed(seed)

#     # Hyperparameter space
#     space = {
#         "learning_rate": [0.01, 0.05, 0.1],
#         "max_depth": [3, 5, 7],
#         "num_leaves": [15, 31, 63],
#         "feature_fraction": [0.6, 0.8, 1.0],
#         "bagging_fraction": [0.6, 0.8, 1.0],
#         "lambda_l2": [0.0, 1.0, 5.0]
#     }

#     param_keys = list(space.keys())
#     param_combinations = list(product(*space.values()))
#     random.shuffle(param_combinations)

#     best_score = -1.0
#     best_params = None

#     print(f"\n Running randomized search over {trials} configs...\n")

#     for i in range(min(trials, len(param_combinations))):
#         trial_params = dict(zip(param_keys, param_combinations[i]))

#         lgbm_params = {
#             "objective": "regression",
#             "metric": "l2",
#             "verbose": -1,
#             **trial_params
#         }

#         print(f"\n Trial {i+1}/{trials} — Params: {trial_params}")

#         ci_test, model = train_lightgbm_survival_model(
#             patients=patients_train,
#             features=base_features,
#             tau=tau,
#             test_size=0.3,
#             seed=seed,
#             lgbm_params=lgbm_params
#         )

#         if ci_test > best_score:
#             best_score = ci_test
#             best_params = trial_params

#     print(f"\n Best C-index: {best_score:.4f}")
#     print(f"Best Params: {best_params}")

# random_lgbm_search(patients_train, base_features, trials=20, tau=7, seed=42)

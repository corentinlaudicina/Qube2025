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

    X_full, y_numeric, y_full, id_list = DataStructs.build_training_data(patients, features)

    X_train, X_test, y_train, y_test, train_idx, test_idx = DataStructs.split_train_test(
        X_full, y_full, id_list, test_size=test_size, seed=seed
    )

    patients_test = [patients[i] for i in test_idx]

    id_train = [id_list[i] for i in train_idx]

    medians, imputed_ids = DataStructs.impute_median_train_test(X_train, X_test, patient_ids_train=id_train)

    lgbm_params = lgbm_params or {
        'objective': 'regression',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l2': 5.0,
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

    cox = CoxPHSurvivalAnalysis(alpha=1e-4)
    cox.fit(X_train, y_train)

    cox_cindex_train = concordance_index_ipcw(y_train, y_train, cox.predict(X_train), tau=7)[0]
    cox_cindex_test = concordance_index_ipcw(y_train, y_test, cox.predict(X_test), tau=7)[0]
    print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.4f}")
    print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.4f}")

    return ci_test, model

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

mutation_embed_features_all = EncodeMutations.get_mutations_embedding(patients_all, 
                                                                      EncodeMutations.build_mut_tfidf_matrix, 
                                                                      embed_prefix="embed_mut", 
                                                                      encoding_dim=8, 
                                                                      epochs=50, 
                                                                      batch_size=32
                                                                      )

cyto_embed_features_all = EncodeCytogenetics.get_cytogenetic_embedding(patients_all, 
                                                                       EncodeCytogenetics.build_cyto_tfidf_matrix, 
                                                                       embed_prefix="embed_cyto", 
                                                                       encoding_dim=8, 
                                                                       epochs=50, 
                                                                       batch_size=32
                                                                       )

features = base_features + mutation_embed_features_all + cyto_embed_features_all

ci_test, model = train_lightgbm_survival_model(patients_train, 
                              features, 
                              tau=7, 
                              test_size=0.33, 
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

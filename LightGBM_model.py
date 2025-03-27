import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sksurv.metrics import concordance_index_ipcw
# from sksurv.linear_model import CoxPHSurvivalAnalysis
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import StandardScaler

import DataStructs  

def build_training_data(patients, features):
    """
    patients: list of Patient objects.
    features: list of feature names, e.g. ["BM_BLAST", "HB", "PLT"].
    
    Returns:
        X: a NumPy array (n_samples x n_features) with missing values as np.nan.
        y_numeric: a NumPy array (n_samples,) of OS_YEARS for training.
        y_structured: a structured NumPy array with fields 'event' and 'time' for evaluation.
        id_list: a list of patient IDs corresponding to each row.
    """
    X_list = []
    y_numeric_list = []
    y_structured_list = []
    id_list = []
    count_skip = 0 
    for p in patients:
        # Skip record if target or event indicator is missing.
        if p.OS_YEARS is None or p.OS_STATUS is None:
            count_skip += 1
            continue

        vals = np.empty(len(features), dtype=float)
        for i, f in enumerate(features):
            value = getattr(p, f)
            if value is not None:
                vals[i] = float(value)
            else:
                vals[i] = np.nan
        X_list.append(vals)
        y_numeric_list.append(float(p.OS_YEARS))
        # Build a tuple (event, time); event is boolean.
        y_structured_list.append((bool(p.OS_STATUS), float(p.OS_YEARS)))
        id_list.append(p.ID)
    
    X = np.array(X_list)
    y_numeric = np.array(y_numeric_list, dtype=np.float32)
    y_structured = np.array(y_structured_list, dtype=[('event','?'), ('time','<f8')])
    print(f"Skipped {count_skip} / {len(patients)} records with missing target or OS_STATUS")
    return X, y_numeric, y_structured, id_list

def impute_median_train_test(X_train, X_test, patient_ids_train=None):
    """
    Impute missing values in X_train with the median computed on X_train,
    and use those medians to impute missing values in X_test.
    
    Optionally, patient_ids_train is a list of patient IDs corresponding to rows in X_train.
    
    Returns:
        medians: 1D NumPy array of medians for each column.
        imputed_ids: list of patient IDs (from the training set) that had at least one missing value.
    """
    medians = np.nanmedian(X_train, axis=0)
    imputed_ids = []
    
    # Impute training set and record which rows were imputed.
    for j in range(X_train.shape[1]):
        missing_train = np.isnan(X_train[:, j])
        if np.any(missing_train):
            if patient_ids_train is not None:
                for i in range(X_train.shape[0]):
                    if missing_train[i]:
                        imputed_ids.append(patient_ids_train[i])
            X_train[missing_train, j] = medians[j]
    
    # Impute test set using the medians computed from training data.
    for j in range(X_test.shape[1]):
        X_test[np.isnan(X_test[:, j]), j] = medians[j]
    
    print(f"Imputed {len(imputed_ids)} patients (training set) with missing values.")
    return medians, imputed_ids

def split_train_test(X, y, ids, test_size=0.3, seed=42):
    """
    Splits X, y, and ids into training and testing sets using fixed random seed.

    Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array (structured or numeric).
        ids : list
            Patient IDs corresponding to rows in X.
        test_size : float
            Fraction of data to reserve for testing.
        seed : int
            Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, id_train, id_test
    """
    np.random.seed(seed)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split_idx = int(round((1 - test_size) * n))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx, :]
    X_test = X[test_idx, :]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # id_train = [ids[i] for i in train_idx]
    # id_test = [ids[i] for i in test_idx]

    return X_train, X_test, y_train, y_test, train_idx, test_idx

def get_feature_importance(model, features, X_subset):
    # Compute SHAP-like importance: how much each feature contributes
    importance = model.predict(X_subset, pred_contrib=True)[:, :-1]  # last col = bias term
    abs_mean_importance = np.mean(np.abs(importance), axis=0)
    return pd.Series(abs_mean_importance, index=features).sort_values(ascending=False)

def train_lightgbm_survival_model(patients, features, tau=7, test_size=0.3, seed=42):

    # --- Build & treat dataset ---
    X_full, y_numeric, y_full, id_list = build_training_data(patients, features)

    X_train, X_test, y_train, y_test, train_idx, test_idx = split_train_test(
        X_full, y_full, id_list, test_size=test_size, seed=seed
    )

    patients_test = [patients[i] for i in test_idx]

    id_train = [id_list[i] for i in train_idx]

    medians, imputed_ids = impute_median_train_test(X_train, X_test, patient_ids_train=id_train)

    # --- Train LightGBM ---
    lgbm_params = {
        'objective': 'regression',
        'learning_rate': 0.1,
        'max_depth': 7,
        'num_leaves': 63,
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

    # --- Evaluate model on sub-populations ---
    pred_train = -model.predict(X_train)
    pred_test  = -model.predict(X_test)

    ci_train = concordance_index_ipcw(y_train, y_train, pred_train, tau=tau)[0]
    ci_test  = concordance_index_ipcw(y_train, y_test,  pred_test,  tau=tau)[0]

    print(f"LightGBM Survival Model Concordance Index IPCW on train: {ci_train:.4f}")
    print(f"LightGBM Survival Model Concordance Index IPCW on test: {ci_test:.4f}")

    test_tp53_pos = [i for i, p in enumerate(patients_test) if p.has_TP53_mutation == 1]
    test_tp53_neg = [i for i, p in enumerate(patients_test) if p.has_TP53_mutation == 0]

    X_pos = X_test[test_tp53_pos, :]
    y_pos = y_test[test_tp53_pos]

    X_neg = X_test[test_tp53_neg, :]
    y_neg = y_test[test_tp53_neg]

    pred_pos = -model.predict(X_pos)
    pred_neg = -model.predict(X_neg)

    ci_pos = concordance_index_ipcw(y_train, y_pos, pred_pos, tau=7)[0]
    ci_neg = concordance_index_ipcw(y_train, y_neg, pred_neg, tau=7)[0]

    print(f"TP53+ C-index: {ci_pos:.4f}")
    print(f"TP53− C-index: {ci_neg:.4f}")

    importance_tp53_pos = get_feature_importance(model, features, X_pos)
    importance_tp53_neg = get_feature_importance(model, features, X_neg)
    importance_df = pd.DataFrame({"TP53+": importance_tp53_pos,
                                  "TP53−": importance_tp53_neg
                                 }).round(3)

    print(importance_df)

    # return 0.0 model , X_train, X_test, y_train, y_test, pred_train, pred_test


#### MAIN ####

dir_train = "train"
dir_test  = "test"

features = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
            'mutation_count', 'mean_VAF', 'max_VAF',
            'has_TP53_mutation', 'log_WBC', 'anc_monocyte_ratio',
            'plt_hb_ratio'
            ]

patients_train = []
DataStructs.read_clinical_data(patients_train, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients_train, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients_train, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients_train)


train_lightgbm_survival_model(patients_train, 
                              features, 
                              tau=7, 
                              test_size=0.3, 
                              seed=42
                              )


# # # CPH unstable for singular training data
# selector = VarianceThreshold(threshold=0.0)
# X_train_filtered = selector.fit_transform(X_train)
# X_test_filtered = selector.transform(X_test)

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train_filtered)
# X_test_scaled = scaler.transform(X_test_filtered)

# cox = CoxPHSurvivalAnalysis()
# cox.fit(X_train_scaled, y_train)

# cox_cindex_train = concordance_index_ipcw(y_train, 
#                                           y_train, 
#                                           cox.predict(X_train_scaled), 
#                                           tau=7)[0]

# cox_cindex_test = concordance_index_ipcw(y_train, 
#                                          y_test, 
#                                          cox.predict(X_test_scaled), 
#                                          tau=7)[0]

# print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {cox_cindex_train:.4f}")
# print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {cox_cindex_test:.4f}")

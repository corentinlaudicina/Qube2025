from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

## need to modify for the original data structure. 

def train_cox_survival_model(X_train, X_test, y_train, y_test, tau=7):
    """
    Trains and evaluates a Cox Proportional Hazards model with variance threshold and standardization.

    Returns:
        cox_model: trained CoxPHSurvivalAnalysis model
        X_train_scaled, X_test_scaled: preprocessed feature matrices
        ci_train, ci_test: concordance indices (IPCW)
    """

    # Remove features with zero variance
    selector = VarianceThreshold(threshold=0.0)
    X_train_filtered = selector.fit_transform(X_train)
    X_test_filtered = selector.transform(X_test)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)

    # Fit model
    cox = CoxPHSurvivalAnalysis()
    cox.fit(X_train_scaled, y_train)

    # Predict + Evaluate
    pred_train = -cox.predict(X_train_scaled)
    pred_test  = -cox.predict(X_test_scaled)

    ci_train = concordance_index_ipcw(y_train, y_train, pred_train, tau=tau)[0]
    ci_test  = concordance_index_ipcw(y_train, y_test,  pred_test,  tau=tau)[0]

    print(f"Cox Proportional Hazard Model Concordance Index IPCW on train: {ci_train:.4f}")
    print(f"Cox Proportional Hazard Model Concordance Index IPCW on test: {ci_test:.4f}")

    return cox, X_train_scaled, X_test_scaled, ci_train, ci_test
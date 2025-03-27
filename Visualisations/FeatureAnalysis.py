import matplotlib.pyplot as plt
import numpy as np

import DataStructs  


def plot_feature_histograms(patients, features, log_scale=False):
    """
    Plots histograms for a list of numeric features from Patient objects.

    Parameters:
        patients: list of Patient objects
        features: list of feature names (strings)
        log_scale: bool, whether to use log scale on x-axis
    """
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        values = [getattr(p, feature) for p in patients if getattr(p, feature) is not None]
        values = np.array(values, dtype=float)

        if log_scale:
            values = values[values > 0]  # remove zeros for log scale

        ax = axes[i]
        ax.hist(values, bins=30, color='steelblue', edgecolor='black')
        ax.set_title(feature)
        if log_scale:
            ax.set_xscale('log')
            ax.set_xlabel("log-scale")
        else:
            ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

patients = []  # List to store Patient objects

DataStructs.read_clinical_data(patients, "train/clinical_train.csv")
DataStructs.add_somatic_mutations(patients, "train/molecular_train.csv")

DataStructs.read_clinical_data(patients, "test/clinical_test.csv")
DataStructs.add_somatic_mutations(patients, "test/molecular_test.csv")

DataStructs.aggregate_mutation_features(patients)

features = ['BM_BLAST', 'HB', 'PLT', 'mutation_count', 'tp53_VAF']

tp53_positive = [p for p in patients if p.has_TP53_mutation == 1]
tp53_negative = [p for p in patients if p.has_TP53_mutation == 0]

plot_feature_histograms(tp53_positive, features)
plot_feature_histograms(tp53_negative, features)


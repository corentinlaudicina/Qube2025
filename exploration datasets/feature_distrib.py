import os, sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import DataStructs, EncodeMutations, EncodeCytogenetics

import matplotlib.pyplot as plt
import numpy as np 

def compare_feature_distributions(train_patients, test_patients, feature_list):
    
    num_features = len(feature_list)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    for i, feature in enumerate(feature_list):
        ax = axs[i]
        train_vals = [getattr(p, feature) for p in train_patients if getattr(p, feature) is not None]
        test_vals  = [getattr(p, feature) for p in test_patients  if getattr(p, feature) is not None]

        if len(train_vals) == 0 or len(test_vals) == 0:
            ax.set_visible(False)
            print(f"⚠️ Skipping {feature}: no valid data in one of the sets")
            continue

        ax.hist(train_vals, bins=40, alpha=0.6, label="Train", color='blue', density=True)
        ax.hist(test_vals,  bins=40, alpha=0.6, label="Inference", color='orange', density=True)
        
        if feature in ['has_TP53_mutation', 'is_normal_karyotype', 'has_del_5q']:
            ax.set_xlim(-0.5, 1.5)
        else:
            mean = np.mean(train_vals)
            std = np.std(train_vals)
            ax.set_xlim(0.0, mean + 3*std)


        ax.set_xlabel(feature)
        if i%3 == 0:
            ax.set_ylabel("Density")
        if i == 0:
            ax.legend()

    dir_to_save = os.path.join(project_root, "report", "Figures")
    plt.savefig(os.path.join(dir_to_save, "feature_distributions.png"))

    plt.show()


def compare_embeddings(train_patients, test_patients, embed_prefix, dim):

    num_cols = dim // 2
    num_rows = (dim + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    for i in range(dim):
        feat_name = f"{embed_prefix}_{i}"
        train_vals = [getattr(p, feat_name) for p in train_patients if hasattr(p, feat_name)]
        test_vals  = [getattr(p, feat_name) for p in test_patients  if hasattr(p, feat_name)]

        ax = axs[i]

        ax.hist(train_vals, bins=40, alpha=0.6, label="Train", color='blue', density=True)
        ax.hist(test_vals,  bins=40, alpha=0.6, label="Inference", color='orange', density=True)
        ax.set_xlabel(feat_name)

        if i%2 == 0:
            ax.set_ylabel("Density")
        if i == 0:
            ax.legend()

        # plt.figure(figsize=(6, 3))
        # plt.hist(train_vals, bins=30, alpha=0.6, label="Train", color='blue', density=True)
        # plt.hist(test_vals,  bins=30, alpha=0.6, label="Test",  color='orange', density=True)
        # # plt.title(f"{feat_name} distribution")
        # plt.xlabel(feat_name)
        # plt.ylabel("Density")
        # plt.legend()
        # # plt.tight_layout()

    
    dir_to_save = os.path.join(project_root, "report", "Figures")
    filename = f"{embed_prefix}_distribution.png"
    plt.savefig(os.path.join(dir_to_save, filename))

    plt.show()

dir_train = os.path.join(project_root,"train")
dir_test = os.path.join(project_root,"test")

base_features = [
                'BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT',
                'mutation_count', 'mean_VAF', 'has_TP53_mutation', 
                'is_normal_karyotype', 'has_del_5q' #'abnormality_count', 'has_monosomy_7', 'has_del_5q'
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

encoding_dim = 4

mutation_embed_features_train = EncodeMutations.get_mutation_autoencoder_embedding(patients_train, 
                                                                                   encoding_dim=encoding_dim
                                                                                   )
cyto_embed_features_train = EncodeCytogenetics.get_cytogenetic_autoencoder_embedding(patients_train, 
                                                                                     encoding_dim=encoding_dim
                                                                                     )

mutation_embed_features_test = EncodeMutations.get_mutation_autoencoder_embedding(patients_test, 
                                                                                  encoding_dim=encoding_dim
                                                                                  )
cyto_embed_features_test = EncodeCytogenetics.get_cytogenetic_autoencoder_embedding(patients_test, 
                                                                                    encoding_dim=encoding_dim
                                                                                    )

# features = base_features
# compare_feature_distributions(patients_train, patients_test, features)

compare_embeddings(patients_train, patients_test, embed_prefix="mut_embed", dim=4)
compare_embeddings(patients_train, patients_test, embed_prefix="cyto_embed", dim=4)



# def compute_tp53_rate(patients):
#     total = len(patients)
#     mutated = sum(1 for p in patients if getattr(p, "has_TP53_mutation", 0) == 1)
#     return mutated, total, mutated / total

# mut_train, total_train, rate_train = compute_tp53_rate(patients_train)
# mut_test, total_test, rate_test = compute_tp53_rate(patients_test)

# print(f"Train set: {mut_train}/{total_train} = {rate_train:.2%}")
# print(f"Test  set: {mut_test}/{total_test} = {rate_test:.2%}")


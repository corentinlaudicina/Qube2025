import os, sys 
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import DataStructs, EncodeMutations

def plot_kaplan_by_cluster(patients, max_clusters=10):
    # Only keep patients with status + cluster info
    filtered = [p for p in patients if p.OS_YEARS is not None and 
            p.OS_STATUS is not None and getattr(p, "cluster_id", None) is not None]

    clusters = sorted(set(p.cluster_id for p in filtered))

    from collections import Counter
    print("Cluster sizes:", Counter(p.cluster_id for p in filtered))

    if len(clusters) > max_clusters:
        print(f"⚠️ Too many clusters ({len(clusters)}), showing first {max_clusters}")
        clusters = clusters[:max_clusters]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))

    for cid in clusters:
        group = [p for p in filtered if p.cluster_id == cid]
        durations = [p.OS_YEARS for p in group]
        events    = [p.OS_STATUS for p in group]
        label = f"Cluster {cid} (n={len(group)})"
        kmf.fit(durations, event_observed=events, label=label)
        kmf.plot(ci_show=False)

    plt.title("Kaplan-Meier Curves by Cluster")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

dir_train = os.path.join(project_root,"train")
dir_test  = os.path.join(project_root,"test")

patients = []
DataStructs.read_clinical_data(patients, os.path.join(dir_train, "clinical_train.csv"))
DataStructs.add_status(patients, os.path.join(dir_train, "target_train.csv"))
DataStructs.add_somatic_mutations(patients, os.path.join(dir_train, "molecular_train.csv"))
DataStructs.aggregate_mutation_features(patients)

gene_df, gene_ids, gene_cols = EncodeMutations.build_top_gene_features(patients, top_n=10)

# Attach to patients
gene_map = gene_df.to_dict(orient="index")
for p in patients:
    if p.ID in gene_map:
        for key, val in gene_map[p.ID].items():
            setattr(p, f"mut_{key}", val)

EncodeMutations.assign_mutation_cluster(patients)

n_clustered = sum(1 for p in patients if hasattr(p, "cluster_id"))
print(f"Patients with cluster_id: {n_clustered}")

plot_kaplan_by_cluster(patients, max_clusters=10)
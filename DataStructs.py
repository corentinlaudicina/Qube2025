import csv
import numpy as np

class SomaticMutation:
    def __init__(self, CHR, START, END, REF, ALT, GENE, PROTEIN_CHANGE, EFFECT, VAF):
        self.CHR = CHR
        self.START = START
        self.END = END
        self.REF = REF
        self.ALT = ALT
        self.GENE = GENE
        self.PROTEIN_CHANGE = PROTEIN_CHANGE
        self.EFFECT = EFFECT
        self.VAF = VAF

    def __repr__(self):
        return (f"SomaticMutation(CHR={self.CHR}, START={self.START}, END={self.END}, "
                f"GENE={self.GENE}, VAF={self.VAF})")


class Patient:
    def __init__(self, ID, CENTER, BM_BLAST, WBC, ANC, MONOCYTES, HB, PLT, CYTOGENETICS, 
                 OS_YEARS=None, OS_STATUS=None, somatic_mutations=None):
        self.ID = ID
        self.CENTER = CENTER
        self.BM_BLAST = BM_BLAST
        self.WBC = WBC
        self.ANC = ANC
        self.MONOCYTES = MONOCYTES
        self.HB = HB
        self.PLT = PLT
        self.CYTOGENETICS = CYTOGENETICS

        self.OS_YEARS = OS_YEARS
        self.OS_STATUS = OS_STATUS

        self.somatic_mutations = somatic_mutations if somatic_mutations is not None else []

        ## MUTATION FEATURES
        self.mutation_count = 0
        self.mean_VAF = 0.0
        self.max_VAF = 0.0
        self.high_impact_count = 0
        self.has_TP53_mutation = 0
        self.tp53_VAF = 0.0

        ## CYTOGENETICS FEATURE
        self.log_WBC = None
        self.anc_monocyte_ratio = None

        ## INTERACTION FEATURES 
        self.plt_hb_ratio = 0

        

    def __repr__(self):
        return f"Patient(ID={self.ID}, CENTER={self.CENTER}, BM_BLAST={self.BM_BLAST}, ...)"

def safe_parse_float(value: str):
    if value == "" or value.upper() == "NA" or value.upper() == "NAN":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def read_clinical_data(patients, filename):
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        print("Processing patients ...")
        for row in reader:
            patient_id = row[0]
            center = row[1]
            bm_blast = safe_parse_float(row[2])
            wbc = safe_parse_float(row[3])
            anc = safe_parse_float(row[4])
            monocytes = safe_parse_float(row[5])
            hb = safe_parse_float(row[6])
            plt_val = safe_parse_float(row[7])
            # Cytogenetics: split by '/' unless empty or "NA"
            if row[8] == "" or row[8].upper() == "NA":
                cytogenetics = []
            else:
                cytogenetics = row[8].split("/")
            # Create a Patient instance and append it to the list
            patients.append(Patient(patient_id, center, bm_blast, wbc, anc, monocytes, hb, plt_val, cytogenetics))

def add_status(patients, filename):
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for target_row in reader:
            target_id = target_row[0]
            OS_years_str = target_row[1]
            status_str = target_row[2]
            # Find matching patient by ID and update status fields
            for patient in patients:
                if patient.ID == target_id:
                    patient.OS_YEARS = safe_parse_float(OS_years_str)
                    if status_str == "1.0":
                        patient.OS_STATUS = True
                    elif status_str == "0.0":
                        patient.OS_STATUS = False
                    else:
                        patient.OS_STATUS = None
                    break  # assuming IDs are unique

def add_somatic_mutations(patients, filename):
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            patient_id = row[0]
            CHR = row[1]
            START = safe_parse_float(row[2])
            END = safe_parse_float(row[3])
            REF = row[4]
            ALT = row[5]
            GENE = row[6]
            PROTEIN_CHANGE = row[7]
            EFFECT = row[8]
            VAF = safe_parse_float(row[9])
            DEPTH = safe_parse_float(row[10])

            mutation = SomaticMutation(CHR, START, END, REF, ALT, GENE, PROTEIN_CHANGE, EFFECT, VAF)

            for patient in patients:
                if patient.ID == patient_id:
                    patient.somatic_mutations.append(mutation)
                    break  

def aggregate_mutation_features(patients):
    """
    Adds mutation-based features to each Patient object in-place.
    """
    for p in patients:
        muts = p.somatic_mutations
        if not muts:
            p.mutation_count = 0
            p.mean_VAF = 0.0
            p.max_VAF = 0.0
            p.high_impact_count = 0
            p.has_TP53_mutation = 0
            continue

        VAFs = [m.VAF for m in muts if m.VAF is not None]
        effects = [m.EFFECT for m in muts]
        genes = [m.GENE for m in muts]

        tp53_vafs = [m.VAF for m in muts if m.GENE == "TP53" and m.VAF is not None]

        p.mutation_count = len(muts)
        p.mean_VAF = np.mean(VAFs) if VAFs else 0.0
        p.max_VAF = np.max(VAFs) if VAFs else 0.0
        p.high_impact_count = effects.count("HIGH")
        p.has_TP53_mutation = 1 if "TP53" in genes else 0
        p.tp53_VAF = 1 if p.tp53_VAF > 0.3 else 0 
        
        if p.WBC is not None and p.WBC >= 0:
            p.log_WBC = np.log1p(p.WBC)

        # ANC / MONOCYTE ratio
        if p.ANC is not None and p.MONOCYTES is not None and p.MONOCYTES > 0:
            p.anc_monocyte_ratio = p.ANC / p.MONOCYTES

        # Interaction features
        p.plt_hb_ratio = np.log1p(p.PLT / p.HB) if p.PLT is not None and p.HB is not None and p.HB != 0 else None        


def parse_cytogenetics(patients):
    for p in patients:
        cyto = p.CYTOGENETICS.upper() if p.CYTOGENETICS else ""

        # Normal karyotypes
        if "46,xx" in cyto or "46,xy" in cyto:
            p.is_normal_karyotype = 1

        # abnormalities (naive count of '+'/'-'/'del(')
        p.abnormality_count = cyto.count('+') + cyto.count('-') + cyto.count('DEL(')

        # high-risk markers
        if "-7" in cyto:
            p.has_monosomy_7 = 1
        if "del(5q)" in cyto or "del(5)" in cyto:
            p.has_del_5q = 1

# # Example usage:
# patients = []  # List to store Patient objects
# read_clinical_data(patients, "train/clinical_train.csv")
# add_status(patients, "train/target_train.csv")
# add_somatic_mutations(patients, "train/molecular_train.csv")

# # Now you can inspect your patients:
# for p in patients:
#     # print(p)
#     print(p.somatic_mutations)
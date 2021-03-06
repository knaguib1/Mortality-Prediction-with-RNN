import os
import pickle
import pandas as pd

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
    """
    :param icd9_object: ICD-9 code (Pandas/Numpy object).
    :return: extracted main digits of ICD-9 code
    """
    icd9_str = str(icd9_object)
    # TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
    # TODO: Read the homework description carefully.
    if icd9_str[0].isalpha() and not icd9_str.startswith('V'):
        x = 4
    else:
        x = 3
        
    if x >= len(icd9_str):
        return icd9_str
        
    return icd9_str[:x] 


def build_codemap(df_icd9, transform):
    """
    :return: Dict of code map {main-digits of ICD9: unique feature ID}
    """
    # TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
    # drop na values
    df_digits = df_icd9['ICD9_CODE'].dropna()
    
    # apply transformation and extract unique values
    digits = df_digits.apply(transform).unique()
    
    # construct dictionary
    # create index for each unique code
    nDigits = len(digits)
    nIndex = list(range(nDigits))
    
    # create dictionary
    codemap = dict(zip(digits, nIndex))
    
    return codemap


def create_dataset(path, codemap, transform):
    """
    :param path: path to the directory contains raw files.
    :param codemap: 3-digit ICD-9 code feature map
    :param transform: e.g. convert_icd9
    :return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
    """
    # TODO: 1. Load data from the three csv files
    # TODO: Loading the mortality file is shown as an example below. Load two other files also.
    df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
    df_diag = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
    df_admin = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))

    # TODO: 2. Convert diagnosis code in to unique feature ID.
    # TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
    df_diag['ICD9_CODE'] = df_diag.ICD9_CODE.transform(transform)
    
    # TODO: 3. Group the diagnosis codes for the same visit.
    df_diag_visit = df_diag.groupby('HADM_ID')
    
    # TODO: 4. Group the visits for the same patient.
    df_pat = df_admin.groupby('SUBJECT_ID')

    # TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
    # TODO: Visits for each patient must be sorted in chronological order.
    

    # TODO: 6. Make patient-id List and label List also.
    # TODO: The order of patients in the three List output must be consistent.
    
    patient_ids = []
    labels = []
    seq_data = []

    for patient, df in df_pat:
        
        # append patient to patient list
        patient_ids.append(patient)
        
        # get patient label
        label = df_mortality.loc[df_mortality.SUBJECT_ID == patient, 'MORTALITY'].values[0]
        labels.append(label)
        
        # sort patient visits in chronological order
        df.sort_values(by='ADMITTIME', inplace=True)
        
        # placeholder
        seq_temp = []
        
        for idx, row in df.iterrows():
            
            # constuct new dataframe
            df_temp = df_diag_visit.get_group(row['HADM_ID'])
            
            # filter for values in codemap
            df_temp = df_temp[df_temp.ICD9_CODE.isin(codemap.keys())]
            
            # store ICD9 values in a list
            diags = df_temp.ICD9_CODE.to_list()
            
            # map results using codemap
            seq_temp.append([codemap[k] for k in diags])
            
        # store results
        seq_data.append(seq_temp)
    
    return patient_ids, labels, seq_data


def main():
    # Build a code map from the train set
    print("Build feature id map")
    df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
    codemap = build_codemap(df_icd9, convert_icd9)
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Train set
    print("Construct train set")
    train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

    pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Validation set
    print("Construct validation set")
    validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

    pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Test set
    print("Construct test set")
    test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

    pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Complete!")


if __name__ == '__main__':
    main()

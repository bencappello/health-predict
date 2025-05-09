#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Kirshoff Diabetes Readmission Replication
# 
# This script attempts to replicate the baseline Random Forest performance (F1 ~0.76) reported in Kirshoff's Kaggle notebook for predicting hospital readmission.
# The analysis is based on Kirshoff's Kaggle notebook: `https://www.kaggle.com/code/kirshoff/diabetic-patient-readmission-prediction`
# and the summary report: `data/kirshoff results - Diabetes.md`.
#
# **Focus of this version: Robustly ensure numeric features are purely numeric before scaling.**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
import time
from imblearn.over_sampling import SMOTE

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

print("Libraries loaded.")

# %% [markdown]
# ## 1. Load Full Raw Data from S3

# %%
S3_BUCKET_NAME = "health-predict-mlops-f9ac6509"
FULL_RAW_DATA_KEY = "raw_data/diabetic_data.csv"

s3_client = boto3.client('s3')

def load_full_raw_data(bucket, key, s3_client_instance):
    """Loads the full raw CSV file from S3."""
    try:
        response = s3_client_instance.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        print(f"Successfully loaded '{key}' from S3 bucket '{bucket}'. Initial shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading '{key}' from S3: {e}")
        return None

df_raw = load_full_raw_data(S3_BUCKET_NAME, FULL_RAW_DATA_KEY, s3_client)

# %% [markdown]
# ## 2. Data Cleaning and Preprocessing (aligning with Kirshoff's report)

# %%
if df_raw is not None:
    df = df_raw.copy()
    start_time = time.time()

    # --- 2.1 Replace '?' with NaN ---
    print("\n--- Replacing '?' with NaN ---")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('?', np.nan)
    
    # --- 2.2 Filter out 'Unknown/Invalid' gender ---
    if 'gender' in df.columns:
        print(f"\n--- Handling 'gender' ---")
        print(f"Initial gender value counts:\n{df['gender'].value_counts(dropna=False)}")
        df = df[~df['gender'].isin(['Unknown/Invalid'])]
        print(f"Shape after removing 'Unknown/Invalid' gender: {df.shape}")
        print(f"Gender value counts after removal:\n{df['gender'].value_counts(dropna=False)}")

    # --- 2.2.1 Keep only first encounter per patient (aligning with Kirshoff) ---
    if 'patient_nbr' in df.columns:
        print(f"\n--- Deduplicating by 'patient_nbr' (keeping first encounter) ---")
        print(f"Shape before deduplication: {df.shape}")
        # Ensure 'first' is consistent by sorting if encounter_id is present
        # (encounter_id is dropped later, so ensure it exists if used for sorting)
        sort_by_cols = ['patient_nbr']
        if 'encounter_id' in df.columns:
            sort_by_cols.append('encounter_id')
        df.sort_values(by=sort_by_cols, inplace=True) 
        df.drop_duplicates(subset=['patient_nbr'], keep='first', inplace=True)
        print(f"Shape after deduplication: {df.shape}")
    else:
        print("Warning: 'patient_nbr' column not found for deduplication.")

    # --- 2.4 Handle Discharge Disposition (remove hospice/expired) ---
    # Kirshoff implicitly does this by not including these in diag groups or by general cleaning.
    # Apply this *before* dropping discharge_disposition_id.
    if 'discharge_disposition_id' in df.columns:
        print(f"\n--- Filtering by 'discharge_disposition_id' for expired/hospice ---")
        df['discharge_disposition_id'] = pd.to_numeric(df['discharge_disposition_id'], errors='coerce')
        expired_hospice_dispositions = [11, 13, 14, 19, 20, 21]
        initial_rows_discharge = len(df)
        df = df[~df['discharge_disposition_id'].isin(expired_hospice_dispositions)]
        rows_removed_discharge = initial_rows_discharge - len(df)
        print(f"Removed {rows_removed_discharge} rows due to hospice/expired discharge. Current shape: {df.shape}")

    # --- Columns to drop (identifiers, high missingness, Kirshoff's irrelevant IDs) ---
    # patient_nbr will be dropped here after deduplication.
    # encounter_id, weight, payer_code, medical_specialty, admission_type_id, discharge_disposition_id, admission_source_id
    # citoglipton, examide (single value, Kirshoff drops them)
    cols_to_drop_overall = [
        'encounter_id', 'patient_nbr', # Identifiers
        'weight', 'payer_code', 'medical_specialty', # High missing or not useful as per Kirshoff
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id', # Kirshoff's irrelevant IDs
        'citoglipton', 'examide' # Single value columns Kirshoff drops
    ]
    
    print(f"\n--- Dropping specified columns ---")
    # Drop columns that exist in the DataFrame
    cols_actually_in_df_to_drop = [col for col in cols_to_drop_overall if col in df.columns]
    if cols_actually_in_df_to_drop:
        df.drop(columns=cols_actually_in_df_to_drop, inplace=True)
        print(f"Dropped columns: {cols_actually_in_df_to_drop}. Shape after dropping: {df.shape}")
    else:
        print("No columns from the drop list were found in the DataFrame.")

    # --- 2.5 Target Variable: 'readmitted_binary' ---
    print(f"\n--- Engineering target variable 'readmitted_binary' ---")
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].replace(['>30', '<30'], 'YES')
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == 'YES' else 0)
        df.drop(columns=['readmitted'], inplace=True)
        print(f"Target variable 'readmitted_binary' created (YES = 1, NO = 0). Distribution:")
        print(df['readmitted_binary'].value_counts(normalize=True))
    else:
        raise ValueError("Target column 'readmitted' not found.")
    
    df_clean = df.copy() # Start using df_clean from this point for further specific cleaning
    print(f"Shape after initial setup of df_clean: {df_clean.shape}")

    # --- Kirshoff's NaN handling for 'race' (drop rows) ---
    if 'race' in df_clean.columns and df_clean['race'].isnull().any():
        print(f"\n--- Dropping rows with missing 'race' ---")
        print(f"Missing values in 'race' before dropna: {df_clean['race'].isnull().sum()}")
        df_clean.dropna(subset=['race'], inplace=True)
        print(f"Shape after dropping rows with missing 'race': {df_clean.shape}")

    # --- Kirshoff's specific NaN handling for diag_1, diag_2, diag_3 BEFORE grouping ---
    print(f"\n--- Specific NaN handling for diag_1, diag_2, diag_3 ---")
    if 'number_diagnoses' in df_clean.columns:
        df_clean['number_diagnoses'] = pd.to_numeric(df_clean['number_diagnoses'], errors='coerce')

    if 'diag_1' in df_clean.columns:
        original_rows = df_clean.shape[0]
        df_clean.dropna(subset=['diag_1'], inplace=True)
        print(f"Shape after dropping rows with missing diag_1: {df_clean.shape}. Rows removed: {original_rows - df_clean.shape[0]}")

    if 'diag_2' in df_clean.columns and 'diag_3' in df_clean.columns and 'number_diagnoses' in df_clean.columns:
        original_rows = df_clean.shape[0]
        condition_diag2_not_null = df_clean['diag_2'].notna()
        condition_diag2_is_null_and_others_ok = (
            df_clean['diag_2'].isnull() & 
            df_clean['diag_3'].isnull() & 
            (df_clean['number_diagnoses'] <= 1)
        )
        df_clean = df_clean[condition_diag2_not_null | condition_diag2_is_null_and_others_ok]
        print(f"Shape after diag_2 conditional handling: {df_clean.shape}. Rows removed: {original_rows - df_clean.shape[0]}")

    if 'diag_3' in df_clean.columns and 'number_diagnoses' in df_clean.columns:
        original_rows = df_clean.shape[0]
        condition_diag3_not_null = df_clean['diag_3'].notna()
        condition_diag3_is_null_and_others_ok = (
            df_clean['diag_3'].isnull() & 
            (df_clean['number_diagnoses'] <= 2)
        )
        df_clean = df_clean[condition_diag3_not_null | condition_diag3_is_null_and_others_ok]
        print(f"Shape after diag_3 conditional handling: {df_clean.shape}. Rows removed: {original_rows - df_clean.shape[0]}")

    # --- Kirshoff's Diagnosis Code Grouping ---
    print("\n--- Grouping Diagnosis Codes ---")
    diag_cols_to_group = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols_to_group:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str) 
            df_clean[col] = df_clean[col].str.replace('\\.0$', '', regex=True) 
            
            df_clean.loc[df_clean[col].str.contains('^(39[0-9]|4[0-5][0-9]|785)', na=False), col] = 'Circulatory'
            df_clean.loc[df_clean[col].str.contains('^(4[6-9][0-9]|5[0-1][0-9]|786)', na=False), col] = 'Respiratory'
            df_clean.loc[df_clean[col].str.contains('^(5[2-7][0-9]|787)', na=False), col] = 'Digestive'
            df_clean.loc[df_clean[col].str.contains('^250', na=False), col] = 'Diabetes'
            df_clean.loc[df_clean[col].str.contains('^(8[0-9]{2}|9[0-9]{2})', na=False), col] = 'Injury'
            df_clean.loc[df_clean[col].str.contains('^(71[0-9]|72[0-9]|73[0-9])', na=False), col] = 'Musculoskeletal'
            df_clean.loc[df_clean[col].str.contains('^(5[8-9][0-9]|6[0-2][0-9]|788)', na=False), col] = 'Genitourinary'
            df_clean.loc[df_clean[col].str.contains('^(1[4-9][0-9]|2[0-3][0-9])', na=False), col] = 'Neoplasms'
            df_clean.loc[df_clean[col].str.match('^[EV]', na=False), col] = 'Other' # Codes starting with E or V
            
            current_categories = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
            # Also include 'nan' string explicitly if it resulted from astype(str) on NaN, to be changed to 'Other'
            df_clean.loc[~df_clean[col].isin(current_categories) | (df_clean[col].str.lower() == 'nan'), col] = 'Other'

            print(f"Value counts for grouped {col}:")
            print(df_clean[col].value_counts(dropna=False))

    # --- Kirshoff's final fillna('None') after grouping and diag handling ---
    print("\n--- Filling ALL remaining NaNs with 'None' (string) ---")
    numeric_cols_with_nan_before_final_fill = df_clean.select_dtypes(include=np.number).isnull().sum()
    numeric_cols_with_nan_before_final_fill = numeric_cols_with_nan_before_final_fill[numeric_cols_with_nan_before_final_fill > 0]
    if not numeric_cols_with_nan_before_final_fill.empty:
        print(f"Warning: Numeric columns have NaNs before final fillna('None'):\n{numeric_cols_with_nan_before_final_fill}")
        # Impute numeric NaNs with 0 before global fillna('None') to avoid type issues
        for num_col in numeric_cols_with_nan_before_final_fill.index:
            print(f"Imputing NaNs in numeric column {num_col} with 0.")
            df_clean[num_col].fillna(0, inplace=True)
            
    df_clean.fillna('None', inplace=True)
    print("Global fillna('None') applied.")

    # --- Kirshoff's specific feature transformations ---
    print("\n--- Applying Kirshoff's specific feature transformations ---")
    # 1. Age mapping (Kirshoff uses the mapped values, not ordinal 0-9)
    if 'age' in df_clean.columns: # Original 'age' column with ranges like '[0-10)'
        print("Mapping 'age' to numerical midpoints...")
        age_dict = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
            '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
        }
        # Handle if 'None' string is present due to prior fillna
        df_clean['age_mapped'] = df_clean['age'].replace('None', np.nan).map(age_dict).fillna(0).astype(int) 
        df_clean.drop(columns=['age'], inplace=True)
        print(df_clean[['age_mapped']].head())
    elif 'age_ordinal' in df_clean.columns: # If original age was already converted to ordinal
        # This path assumes age_ordinal (0-9) exists and we want Kirshoff's mapping (5, 15, ...)
        # This requires knowing the original bins for age_ordinal, or remapping based on those bins.
        # For now, if age_mapped is not created because 'age' is missing, we assume age_ordinal is the intended feature
        # OR, we should ensure 'age' is present before this step.
        # Safest: drop age_ordinal if age_mapped is created, otherwise use age_ordinal.
        print("'age' column not found for mapping, 'age_ordinal' might be used if present.")
        if 'age' not in df.columns: # Check original df before it became df_clean
            print("ERROR: Original 'age' column with string bins not found for mapping to Kirshoff's values.")
        # If age_ordinal was created earlier (lines 125-132) and original 'age' was dropped,
        # we need to ensure this `age_ordinal` is not used if `age_mapped` is the target.
        if 'age_ordinal' in df_clean.columns and 'age_mapped' in df_clean.columns:
            print("Dropping 'age_ordinal' as 'age_mapped' is created.")
            df_clean.drop(columns=['age_ordinal'], inplace=True, errors='ignore')

    # 2. Medication columns: 'No' / 'None' -> 0, others ('Steady', 'Up', 'Down') -> 1
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
        'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'troglitazone', 'tolazamide', 'insulin', # Removed 'examide', 'citoglipton' as they are dropped
        'glyburide-metformin','glipizide-metformin', 'glimepiride-pioglitazone', 
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    print("\n--- Transforming medication columns to binary (0 or 1) ---")
    # Store original med values before converting to 0/1 for nummed calculation
    df_clean_med_original_values = pd.DataFrame()
    for col in med_cols:
        if col in df_clean.columns:
            df_clean_med_original_values[col] = df_clean[col].copy()
            df_clean[col] = df_clean[col].replace(['No', 'None'], 0)
            df_clean[col] = df_clean[col].replace(['Steady', 'Up', 'Down'], 1)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    # --- Create 'nummed' feature (count of medications patient is on) ---
    print("\n--- Creating 'nummed' feature ---")
    df_clean['nummed'] = 0
    for col in med_cols: # Use the same med_cols list
        if col in df_clean_med_original_values.columns: # Check against the original values DF
            # Increment nummed if the original value was not 'No' and not 'None'
            # This aligns with Kirshoff counting any actual prescription ('Steady', 'Up', 'Down')
            df_clean['nummed'] += (
                (df_clean_med_original_values[col] != 'No') & 
                (df_clean_med_original_values[col] != 'None')
            ).astype(int)
    print(f"'nummed' feature created. Example values:\n{df_clean['nummed'].value_counts().sort_index().head()}")

    # 3. A1Cresult mapping (None->0, >7->1, >8->2, Norm->3)
    if 'A1Cresult' in df_clean.columns:
        print("\n--- Mapping 'A1Cresult' ---")
        a1c_map = {'None': 0, '>7': 1, '>8': 2, 'Norm': 3}
        df_clean['A1Cresult_mapped'] = df_clean['A1Cresult'].map(a1c_map).fillna(0).astype(int)
        df_clean.drop(columns=['A1Cresult'], inplace=True)
        # print(df_clean[['A1Cresult_mapped']].head())

    # 4. max_glu_serum mapping (None->0, >200->1, >300->2, Norm->3)
    if 'max_glu_serum' in df_clean.columns:
        print("\n--- Mapping 'max_glu_serum' ---")
        glu_map = {'None': 0, '>200': 1, '>300': 2, 'Norm': 3}
        df_clean['max_glu_serum_mapped'] = df_clean['max_glu_serum'].map(glu_map).fillna(0).astype(int)
        df_clean.drop(columns=['max_glu_serum'], inplace=True)
        # print(df_clean[['max_glu_serum_mapped']].head())

    # Final check for NaNs (Should be none if fillna('None') and subsequent mappings worked)
    final_nan_check = df_clean.isnull().sum().sum()
    if final_nan_check > 0:
        print(f"WARNING: {final_nan_check} NaNs remain after Kirshoff feature transformations.")
        print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])
    else:
        print("Final NaN check after Kirshoff transformations passed.")

    # Drop all-NaN columns (if any were created)
    df_clean.dropna(axis=1, how='all', inplace=True)
    print(f"Shape after Kirshoff transformations and dropping all-NaN columns: {df_clean.shape}")

    processing_time_total = time.time() - start_time
    print(f"Total data cleaning and preprocessing completed in {processing_time_total:.2f} seconds.")
    
    if df_clean.empty:
        print("DataFrame is empty after cleaning. Aborting.")
        # Ensure df is set to None so downstream processing doesn't run with an empty df_clean
        df = None 
    else:
        df = df_clean # Assign the fully cleaned df back to df for the modeling part

else: # From if df_raw is not None
    print("Raw data not loaded. Halting script.")
    df = None # Ensure df is None if raw data wasn't loaded


# %% [markdown]
# ## 3. Feature Preprocessing for Model

# %%
if df is not None and not df.empty: # Check if df is not None and not empty
    y = df['readmitted_binary']
    X = df.drop(columns=['readmitted_binary'])

    # Identify numerical and categorical features
    # Ensure all columns used for modeling are either numerical or categorical after all transformations.
    
    # Numerical features: Should include time_in_hospital, lab_procs, etc., AND mapped age, A1C, glu, and binary meds
    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'nummed' # Added nummed
    ]
    if 'age_mapped' in X.columns:
        numerical_features.append('age_mapped')
    elif 'age_ordinal' in X.columns: # Fallback if age_mapped wasn't created (should not happen if logic is correct)
        numerical_features.append('age_ordinal')
        
    if 'A1Cresult_mapped' in X.columns:
        numerical_features.append('A1Cresult_mapped')
    if 'max_glu_serum_mapped' in X.columns:
        numerical_features.append('max_glu_serum_mapped')
    
    # Add medication columns (which are now 0/1)
    for med_col in med_cols: # med_cols defined during transformation
        if med_col in X.columns:
            numerical_features.append(med_col)
    
    numerical_features = list(set(numerical_features).intersection(set(X.columns))) # Ensure all exist

    # Categorical features: Should include race, gender, grouped diags, change, diabetesMed
    categorical_features = [
        'race', 'gender', 
        'diag_1', 'diag_2', 'diag_3', # These are now grouped categories
        'change', 'diabetesMed'
    ]
    categorical_features = list(set(categorical_features).intersection(set(X.columns))) # Ensure all exist

    print(f"\nFinal Numerical Features for Preprocessor: {sorted(numerical_features)}")
    print(f"Final Categorical Features for Preprocessor: {sorted(categorical_features)}")
    
    # Sanity check: ensure no overlap and all columns are covered
    processed_cols = set(numerical_features + categorical_features)
    all_X_cols = set(X.columns)
    if processed_cols != all_X_cols:
        print("WARNING: Mismatch between features for preprocessor and columns in X!")
        print(f"Columns in X but not in feature lists: {all_X_cols - processed_cols}")
        print(f"Features in lists but not in X: {processed_cols - all_X_cols}")
        # For robust run, only use columns that are in X for preprocessor lists
        numerical_features = [col for col in numerical_features if col in X.columns]
        categorical_features = [col for col in categorical_features if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop' # Drop any columns not explicitly handled
    )
else:
    preprocessor = None
    X = None
    y = None
    print("DataFrame df is None or empty. Halting before model preprocessing.")


# %% [markdown]
# ## 4. Train-Test Split

# %%
if X is not None and y is not None and not X.empty and not y.empty: # Added checks for empty X,y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    print(f"\nData split into training and testing sets:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    # --- Ensure correct dtypes before ColumnTransformer ---
    print("\n--- Sanitizing feature dtypes before ColumnTransformer ---")
    for col in numerical_features:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0) # Coerce & fill just in case
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
            
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str) # Ensure categorical are string
            X_test[col] = X_test[col].astype(str)

    print("\nFitting preprocessor on X_train...")
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    except Exception as e:
        print(f"Error during preprocessor transformation: {e}")
        # Detailed diagnostics
        for col in numerical_features:
            if col in X_train.columns and X_train[col].apply(type).nunique() > 1:
                print(f"Column {col} in numerical_features has mixed types in X_train: {X_train[col].apply(type).unique()}")
            if col in X_train.columns and X_train[col].isnull().any():
                print(f"Column {col} in numerical_features has NaNs in X_train after coerce/fill.")
        for col in categorical_features:
            if col in X_train.columns and X_train[col].isnull().any():
                print(f"Column {col} in categorical_features has NaNs in X_train after astype(str).")
        # Re-raise to halt if preprocessor fails
        if 'X_train_processed' not in locals(): X_train_processed = None 
        if 'X_test_processed' not in locals(): X_test_processed = None
        # raise 

    if X_train_processed is not None and X_test_processed is not None:
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())
        print(f"X_train_processed_df shape: {X_train_processed_df.shape}")
        print(f"X_test_processed_df shape: {X_test_processed_df.shape}")
    else:
        print("Preprocessor transformation failed or was skipped.")
        X_train_processed_df = None
        X_test_processed_df = None

else: # From if X is not None and y is not None
    X_train_processed = None 
    X_test_processed = None
    X_train_processed_df = None
    X_test_processed_df = None
    print("X or y were None or empty, skipping train-test split and preprocessing.")


# %% [markdown]
# ## 5. Train Random Forest Model

# %%
model = None
if X_train_processed_df is not None and not X_train_processed_df.empty: # Check if DataFrame exists and is not empty
    print("\n--- Applying SMOTE to training data ---")
    smote = SMOTE(random_state=42)
    X_train_input = X_train_processed_df.values 
    X_train_smote, y_train_smote = smote.fit_resample(X_train_input, y_train)
    print(f"Shape after SMOTE: X_train_smote: {X_train_smote.shape}, y_train_smote: {y_train_smote.shape}")
    print(f"Original training target distribution:\n{y_train.value_counts(normalize=True)}")
    y_train_smote_series = pd.Series(y_train_smote) 
    print(f"SMOTE training target distribution:\n{y_train_smote_series.value_counts(normalize=True)}")

    print("\n--- Training Random Forest Model (Kirshoff Replication Attempt with SMOTE) ---")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1  
    )
    
    fit_start_time = time.time()
    model.fit(X_train_smote, y_train_smote) 
    fit_time = time.time() - fit_start_time
    print(f"Model training completed in {fit_time:.2f} seconds.")
else:
    print("Processed training data (X_train_processed_df) not available or empty. Skipping model training.")

# %% [markdown]
# ## 6. Evaluate Model

# %%
if model is not None and X_test_processed_df is not None and not X_test_processed_df.empty: # Check if DataFrame exists and is not empty
    print("\n--- Evaluating Model on Test Set ---")
    X_test_input = X_test_processed_df.values 
    y_pred_test = model.predict(X_test_input)
    y_proba_test = model.predict_proba(X_test_input)[:, 1] 

    print("\nClassification Report (Test Set):")
    # Define target names based on our binary encoding (0: NO, 1: YES for any readmission)
    print(classification_report(y_test, y_pred_test, target_names=['Not Readmitted (NO)', 'Readmitted (YES)']))
    
    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test) 
    auc = roc_auc_score(y_test, y_proba_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    
    print(f"Accuracy Score (Test Set): {accuracy:.4f}")
    print(f"Precision (Positive Class, Test Set): {precision:.4f}")
    print(f"Recall (Positive Class, Test Set): {recall:.4f}")
    print(f"F1 Score (Positive Class, Test Set): {f1:.4f}") 
    print(f"AUC-ROC Score (Test Set): {auc:.4f}")
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Not Readmitted', 'Pred Readmitted'], 
                yticklabels=['Actual Not Readmitted', 'Actual Readmitted'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set (Kirshoff Replication)')
    # Save the plot to a file
    plot_path = "confusion_matrix_replication.png"
    try:
        plt.savefig(plot_path)
        print(f"Confusion matrix plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    plt.show()
else:
    print("Model not trained or processed test data not available/empty. Skipping evaluation.")

# %% [markdown]
# ## 7. Learnings and Report (Placeholder)
# 
# *   Compare results with Kirshoff's reported F1 of 0.76.
# *   Detail differences in preprocessing that might affect the outcome.
# *   Note any challenges during replication.

# %%
print("\nScript execution finished.") 
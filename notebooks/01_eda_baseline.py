# %% [markdown]
# # Health Predict: Initial EDA & Baseline Model
# 
# This notebook performs an initial Exploratory Data Analysis (EDA) on the first 20% of the diabetic dataset and trains a simple baseline model.
# The data used here is `initial_train.csv`, `initial_validation.csv`, and `initial_test.csv` which were derived from the first 20% of the full dataset.

# %%
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import boto3
from io import StringIO

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# MLflow (optional for local EDA, but good practice)
# import mlflow
# import mlflow.sklearn

# Configure pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

print("Initial libraries loaded.")

# %% [markdown]
# ## 1. Setup and Configuration
# 
# - Define S3 bucket and file paths.
# - Initialize Boto3 client for S3 access.

# %%
# S3 Configuration
S3_BUCKET_NAME = "health-predict-mlops-f9ac6509" # From config_variables.md
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"

INITIAL_TRAIN_KEY = f"{PROCESSED_DATA_DIR}/initial_train.csv"
INITIAL_VALIDATION_KEY = f"{PROCESSED_DATA_DIR}/initial_validation.csv"
INITIAL_TEST_KEY = f"{PROCESSED_DATA_DIR}/initial_test.csv"
FUTURE_DATA_KEY = f"{PROCESSED_DATA_DIR}/future_data.csv" # For later use
FULL_RAW_DATA_KEY = f"{RAW_DATA_DIR}/diabetic_data.csv" # For reference if needed

# Initialize S3 client - will use EC2 instance role credentials
s3_client = boto3.client('s3')

print(f"S3 Bucket: {S3_BUCKET_NAME}")
print(f"Initial Train Data S3 Key: {INITIAL_TRAIN_KEY}")
print(f"Initial Validation Data S3 Key: {INITIAL_VALIDATION_KEY}")
print(f"Initial Test Data S3 Key: {INITIAL_TEST_KEY}")

# %% [markdown]
# ## 2. Load Initial Training Data from S3
# 
# We'll load the `initial_train.csv` for our primary EDA and model training.

# %%
def load_df_from_s3(bucket, key, s3_client_instance):
    """Loads a CSV file from S3 into a pandas DataFrame."""
    try:
        response = s3_client_instance.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        print(f"Successfully loaded '{key}' from S3 bucket '{bucket}'. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading '{key}' from S3: {e}")
        return None

df_train_initial = load_df_from_s3(S3_BUCKET_NAME, INITIAL_TRAIN_KEY, s3_client)

if df_train_initial is not None:
    print(df_train_initial.head())

# %% [markdown]
# ---
# *Next steps will involve detailed EDA, preprocessing, baseline model training, and evaluation.*
# ---

# %% [markdown]
# ## 3. Initial Data Exploration (on `df_train_initial`)
# 
# Let's get a first look at the initial training dataset.

# %%
if df_train_initial is not None:
    print("\n--- Data Info ---")
    df_train_initial.info()
    
    print("\n--- Descriptive Statistics (Numerical) ---")
    print(df_train_initial.describe().T)
    
    print("\n--- Descriptive Statistics (Categorical) ---")
    print(df_train_initial.describe(include=['object', 'category']).T)
    
    print("\n--- Missing Values ---")
    missing_values = df_train_initial.isnull().sum()
    missing_percentage = (missing_values / len(df_train_initial)) * 100
    missing_info = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage': missing_percentage})
    print(missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False))

# %% [markdown]
# ### 3.1. Target Variable Analysis (`readmitted`)
# 
# The target variable `readmitted` indicates if a patient was readmitted. Let's examine its distribution.
# - `<30`: Readmitted within 30 days.
# - `>30`: Readmitted after 30 days.
# - `NO`: No readmission.
# 
# For a binary classification baseline, we might simplify this (e.g., readmitted vs. not readmitted). For now, let's see the raw distribution.

# %%
if df_train_initial is not None:
    print("\n--- Target Variable Distribution ('readmitted') ---")
    print(df_train_initial['readmitted'].value_counts(normalize=True) * 100)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='readmitted', data=df_train_initial, order=df_train_initial['readmitted'].value_counts().index)
    plt.title('Distribution of Target Variable: readmitted')
    plt.ylabel('Frequency')
    plt.xlabel('Readmission Status')
    plt.show()

# %% [markdown]
# ### 3.2. Initial Observations & Potential Cleaning Steps:
# 
# *Based on the `.info()`, `.describe()`, and missing values output, we will note down initial observations here.*
# 
# *   **Missing Values:** Columns like `weight`, `medical_specialty`, `payer_code` seem to have a high percentage of missing values. We'll need a strategy for these (e.g., imputation, creating a 'missing' category, or dropping if not useful/too sparse).
# *   **Data Types:** Ensure all columns have appropriate data types. Categorical features might be read as objects and may need explicit conversion to `category` type for efficiency and correct handling in some libraries.
# *   **Identifier Columns:** Columns like `encounter_id` and `patient_nbr` are likely identifiers and might not be useful as direct features for a predictive model but could be useful for tracking or joining data. `patient_nbr` is important because multiple encounters can belong to the same patient.
# *   **Categorical Features with Many Levels:** Some categorical columns might have a very large number of unique values (e.g., `diag_1`, `diag_2`, `diag_3`). This could lead to very high dimensionality if one-hot encoded directly. We might need to group them, use target encoding, or select top N categories.
# *   **Zero Variance / Near Zero Variance:** Check for columns with little to no variation as they won't be predictive.
# *   **Target Variable Imbalance:** The distribution of `readmitted` might be imbalanced, which could affect model training and evaluation. We might need techniques like oversampling, undersampling, or using appropriate metrics (e.g., F1-score, AUC-ROC).
# *   **Special Values:** The dataset description mentions 'Not Available', 'Not Mapped', '?' as special values in some columns. These need to be consistently handled (e.g., converted to `NaN`). `split_data.py` does not currently handle this explicitly for all columns during initial load, so it's something to check in the raw data if issues arise.

# %% [markdown]
# ## 4. Data Cleaning & Preprocessing (Initial Pass)
# 
# Let's start with some basic cleaning based on the observations above.

# %%
# Example: Replace ' ?' with NaN if it exists and wasn't handled earlier (though it should have been by pd.read_csv if it's a standard NA value)
# df_train_initial.replace('?', np.nan, inplace=True)

# We will add more cleaning steps here as we explore further.
print("Placeholder for initial data cleaning steps.")


# %% [markdown]
# ---
# *Further EDA will involve looking at distributions of individual features, relationships between features and the target, and correlations.*
# --- 

# %% [markdown]
# ### 3.3 Further Numerical Feature Exploration

# %%
if df_train_initial is not None:
    numerical_cols = df_train_initial.select_dtypes(include=np.number).columns.tolist()
    # Exclude identifier columns if they were loaded as numbers and not dropped yet
    # For now, assume encounter_id and patient_nbr are not in numerical_cols for plotting
    # Also excluding obviously categorical IDs that might be num_type by mistake
    # A proper list would be derived after confirming true numerical features
    plot_numerical_cols = [col for col in numerical_cols if col not in ['encounter_id', 'patient_nbr', 
                                                                    'admission_type_id', 'discharge_disposition_id', 
                                                                    'admission_source_id']] # Add other IDs if necessary
    
    if not plot_numerical_cols:
        print("No numerical columns selected for plotting after exclusions.")
    else:
        print(f"Plotting histograms for: {plot_numerical_cols}")
        df_train_initial[plot_numerical_cols].hist(bins=30, figsize=(15, 10), layout=(-1, 3))
        plt.suptitle("Histograms of Numerical Features (Initial Train Data)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print(f"Plotting boxplots for: {plot_numerical_cols}")
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(plot_numerical_cols):
            plt.subplot((len(plot_numerical_cols) + 2) // 3, 3, i + 1)
            sns.boxplot(y=df_train_initial[col])
            plt.title(col)
        plt.suptitle("Boxplots of Numerical Features (Initial Train Data)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# %% [markdown]
# ### 3.4 Further Categorical Feature Exploration

# %%
if df_train_initial is not None:
    categorical_cols = df_train_initial.select_dtypes(include=['object', 'category']).columns.tolist()
    # Select a subset of categorical columns for plotting to avoid too many plots
    # Exclude high-cardinality features like diag_1, diag_2, diag_3 for now
    plot_categorical_cols = [col for col in categorical_cols if df_train_initial[col].nunique() < 25 and col not in ['diag_1', 'diag_2', 'diag_3', 'payer_code', 'medical_specialty']] 
    
    if not plot_categorical_cols:
        print("No categorical columns selected for plotting after exclusions/nunique filter.")
    else:
        print(f"Plotting barplots for: {plot_categorical_cols}")
        plt.figure(figsize=(18, len(plot_categorical_cols) * 2)) # Adjusted figure size
        for i, col in enumerate(plot_categorical_cols):
            plt.subplot((len(plot_categorical_cols) + 2) // 3, 3, i + 1)
            sns.countplot(y=df_train_initial[col], order=df_train_initial[col].value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
        plt.suptitle("Distributions of Key Categorical Features (Initial Train Data)")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

# %% [markdown]
# ## 4. Data Cleaning & Preprocessing (Continued)
# 
# Based on initial EDA, we'll perform cleaning and feature engineering.

# %% [markdown]
# ### 4.1 Handling '?' and Special Values
# 
# Many datasets use '?' for missing values. We'll replace these with NaN.

# %%
if df_train_initial is not None:
    # Create copies for cleaning to preserve original loaded data if needed for re-runs
    df_train_processed = df_train_initial.copy()
    
    # Replace '?' with NaN
    # Iterate through columns because df.replace('?', np.nan) might be slow on large mixed-type DFs
    # and can have unintended consequences if '?' is a valid category in some unknown column.
    # For this dataset, '?' is widely known as a missing value indicator.
    print("Replacing '?' with NaN globally...")
    for col in df_train_processed.columns:
        if df_train_processed[col].dtype == 'object':
            df_train_processed[col] = df_train_processed[col].replace('?', np.nan)
    
    # Verify '?' is gone from a sample column known to have them
    if 'race' in df_train_processed.columns:
      print(f"Value counts for 'race' after replacing '?':\n{df_train_processed['race'].value_counts(dropna=False)}")
    if 'payer_code' in df_train_processed.columns:
        print(f"Value counts for 'payer_code' after replacing '?':\n{df_train_processed['payer_code'].value_counts(dropna=False)}")
    if 'medical_specialty' in df_train_processed.columns:
        print(f"Value counts for 'medical_specialty' after replacing '?':\n{df_train_processed['medical_specialty'].value_counts(dropna=False)}")
else:
    print("df_train_initial is not loaded. Skipping cleaning.")
    df_train_processed = None


# %% [markdown]
# ### 4.2 Handling Missing Values (Based on EDA)

# %%
if df_train_processed is not None:
    print("\n--- Initial Missing Values (Post '?' replacement) ---")
    missing_values_post_q = df_train_processed.isnull().sum()
    missing_percentage_post_q = (missing_values_post_q / len(df_train_processed)) * 100
    missing_info_post_q = pd.DataFrame({'Missing Count': missing_values_post_q, 'Missing Percentage': missing_percentage_post_q})
    print(missing_info_post_q[missing_info_post_q['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False))

    # Strategy: Drop columns with very high missing percentage for baseline
    cols_to_drop_missing = []
    if 'weight' in df_train_processed.columns and (df_train_processed['weight'].isnull().sum() / len(df_train_processed)) > 0.9: # Example threshold 90%
        cols_to_drop_missing.append('weight')
    if 'payer_code' in df_train_processed.columns and (df_train_processed['payer_code'].isnull().sum() / len(df_train_processed)) > 0.4: # Example threshold 40%
        cols_to_drop_missing.append('payer_code')
    if 'medical_specialty' in df_train_processed.columns and (df_train_processed['medical_specialty'].isnull().sum() / len(df_train_processed)) > 0.4: # Example threshold 40%
        cols_to_drop_missing.append('medical_specialty')

    if cols_to_drop_missing:
        print(f"\nDropping columns due to high missing values: {cols_to_drop_missing}")
        df_train_processed.drop(columns=cols_to_drop_missing, inplace=True)
    
    # For remaining NaNs in categorical features like diag_1, diag_2, diag_3, race:
    # Impute with a "Missing" category or mode. For baseline, let's use "Missing".
    # This will be handled by OneHotEncoder later if they are still objects/categories.
    # For numerical columns, mean/median imputation might be used, but this dataset has few numerical features with missing values
    # after dropping 'weight'.
    
    # For simplicity in baseline, remaining NaNs in object/category columns will be treated as a separate category by OHE.
    # If any numerical columns had NaNs (other than those dropped), they'd need imputation (e.g., median).
    # Let's check again after drops:
    print("\n--- Missing Values After Dropping High-Missing Columns ---")
    missing_values_after_drop = df_train_processed.isnull().sum()
    missing_percentage_after_drop = (missing_values_after_drop / len(df_train_processed)) * 100
    missing_info_after_drop = pd.DataFrame({'Missing Count': missing_values_after_drop, 'Missing Percentage': missing_percentage_after_drop})
    print(missing_info_after_drop[missing_info_after_drop['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False))
    
    # For this dataset, 'race', 'diag_1', 'diag_2', 'diag_3' might still have NaNs.
    # We'll fill them with a placeholder string 'Missing'
    for col in ['race', 'diag_1', 'diag_2', 'diag_3']:
        if col in df_train_processed.columns and df_train_processed[col].isnull().any():
            print(f"Filling NaNs in '{col}' with 'Missing'")
            df_train_processed[col].fillna('Missing', inplace=True)

    # Add filtering for discharge_disposition_id based on data_summary.md (removing expired/hospice)
    # Typical codes: Expired (11, 19, 20, 21), Hospice (13, 14)
    # These should be confirmed if an ID mapping file is available. Assuming these are standard for now.
    expired_ids = [11, 19, 20, 21]
    hospice_ids = [13, 14]
    discharge_ids_to_remove = expired_ids + hospice_ids

    if 'discharge_disposition_id' in df_train_processed.columns:
        initial_rows = len(df_train_processed)
        df_train_processed = df_train_processed[~df_train_processed['discharge_disposition_id'].isin(discharge_ids_to_remove)]
        rows_removed = initial_rows - len(df_train_processed)
        print(f"Removed {rows_removed} rows due to discharge to hospice or expired.")
    else:
        print("Warning: 'discharge_disposition_id' column not found. Skipping hospice/expired filtering.")


    print("\n--- Final Check for Missing Values ---")
    print(df_train_processed.isnull().sum().sum(), "total missing values remaining.")


# %% [markdown]
# ### 4.3 Feature Engineering

# %%
if df_train_processed is not None:
    # 1. Simplify Target Variable 'readmitted'
    #    <30 OR >30 -> 1 (Any Readmission)
    #    NO -> 0 (No Readmission)
    print("\n--- Simplifying Target Variable 'readmitted' to ANY readmission ---")
    # First, consolidate '<30' and '>30' into 'YES' for any readmission
    df_train_processed['readmitted'] = df_train_processed['readmitted'].replace(['<30', '>30'], 'YES')
    # Then, create the binary target: 1 for 'YES', 0 for 'NO'
    df_train_processed['readmitted_binary'] = df_train_processed['readmitted'].apply(lambda x: 1 if x == 'YES' else 0)
    print(df_train_processed['readmitted_binary'].value_counts(normalize=True))
    
    # 2. Process 'age' column: '[70-80)' -> 75 (midpoint) or ordinal
    # Using ordinal for simplicity and to maintain order
    print("\n--- Processing 'age' column ---")
    if 'age' in df_train_processed.columns:
        age_mapping = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
        df_train_processed['age_ordinal'] = df_train_processed['age'].map(age_mapping)
        print(df_train_processed[['age', 'age_ordinal']].head())
    
    # 3. Drop original 'readmitted' and 'age' columns, and identifiers
    cols_to_drop_engineered = ['readmitted', 'age', 'encounter_id', 'patient_nbr']
    # Also drop diag_1, diag_2, diag_3 for baseline simplicity due to high cardinality and complexity of encoding them properly
    # A more advanced approach would involve feature engineering on these.
    cols_to_drop_engineered.extend(['diag_1', 'diag_2', 'diag_3'])

    existing_cols_to_drop = [col for col in cols_to_drop_engineered if col in df_train_processed.columns]
    print(f"\nDropping columns: {existing_cols_to_drop}")
    df_train_processed.drop(columns=existing_cols_to_drop, inplace=True)
    
    print("\n--- DataFrame after initial Feature Engineering ---")
    print(df_train_processed.head())
    df_train_processed.info()

# %% [markdown]
# ## 5. Preprocessing for Modeling
# 
# - Identify categorical and numerical features.
# - Apply One-Hot Encoding to categorical features and Scaling to numerical features.

# %%
if df_train_processed is not None:
    y_train = df_train_processed['readmitted_binary']
    X_train = df_train_processed.drop(columns=['readmitted_binary'])

    # Identify categorical and numerical columns
    # Ensure 'id' columns that are actually categorical are treated as such
    # For this dataset, many columns like 'admission_type_id' are categorical despite being numbers
    # A full list based on dataset description would be ideal. For now, an approximation:
    potential_categorical_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id'] # Add others if known
    
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col_id in potential_categorical_ids:
        if col_id in X_train.columns and X_train[col_id].dtype != 'object' and X_train[col_id].dtype != 'category':
            if X_train[col_id].nunique() < 30: # Heuristic for ID-like categoricals
                 categorical_features.append(col_id)
                 X_train[col_id] = X_train[col_id].astype('category')


    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    # Remove any numerical features that were re-classified as categorical (like age_ordinal if we want to OHE it, but we'll scale it)
    # or actual categorical_ids that were numeric
    numerical_features = [col for col in numerical_features if col not in categorical_features]
    if 'age_ordinal' in numerical_features: # Keep age_ordinal as numerical for scaling
        pass
    
    # Ensure no overlap and all columns are covered
    categorical_features = list(set(categorical_features)) # Deduplicate
    processed_cols = set(categorical_features + numerical_features)
    all_cols = set(X_train.columns)
    if processed_cols != all_cols:
        print(f"Warning: Column mismatch. Untracked columns: {all_cols - processed_cols}")
        print(f"All: {all_cols}, Processed: {processed_cols}")
        # For any remaining, classify as categorical if object, else numerical
        for rem_col in (all_cols - processed_cols):
            if X_train[rem_col].dtype == 'object':
                categorical_features.append(rem_col)
            else:
                numerical_features.append(rem_col)
        categorical_features = list(set(categorical_features))
        numerical_features = list(set(numerical_features) - set(categorical_features))


    print(f"\nIdentified Numerical Features: {numerical_features}")
    print(f"Identified Categorical Features: {categorical_features}")

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], 
        remainder='passthrough' # Should be empty if all cols are covered
    )

    # Fit and transform the training data
    print("\nFitting preprocessor and transforming training data...")
    X_train_prepared = preprocessor.fit_transform(X_train)
    
    # Get feature names after OHE for creating a DataFrame (optional, but good for inspection)
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + ohe_feature_names.tolist()
        X_train_prepared_df = pd.DataFrame(X_train_prepared.toarray() if hasattr(X_train_prepared, "toarray") else X_train_prepared, columns=all_feature_names) # .toarray() for sparse matrix
        print(f"Training data transformed. Shape: {X_train_prepared_df.shape}")
        print(X_train_prepared_df.head())
    except Exception as e:
        print(f"Could not get OHE feature names or create DataFrame: {e}")
        print(f"Training data transformed (sparse matrix). Shape: {X_train_prepared.shape}")


    # Preprocess Validation and Test data
    print("\nLoading and preprocessing validation and test data...")
    df_val_initial = load_df_from_s3(S3_BUCKET_NAME, INITIAL_VALIDATION_KEY, s3_client)
    df_test_initial = load_df_from_s3(S3_BUCKET_NAME, INITIAL_TEST_KEY, s3_client)

    X_val_prepared = None
    y_val = None
    X_test_prepared = None
    y_test = None

    if df_val_initial is not None:
        df_val_processed = df_val_initial.copy()
        for col in df_val_processed.columns: # Replace '?'
            if df_val_processed[col].dtype == 'object':
                df_val_processed[col] = df_val_processed[col].replace('?', np.nan)
        if cols_to_drop_missing: # Drop same high-missing columns
            df_val_processed.drop(columns=[col for col in cols_to_drop_missing if col in df_val_processed.columns], inplace=True)
        for col in ['race', 'diag_1', 'diag_2', 'diag_3']: # Fill NaNs
            if col in df_val_processed.columns: df_val_processed[col].fillna('Missing', inplace=True)
        
        # Apply discharge disposition filtering
        if 'discharge_disposition_id' in df_val_processed.columns:
            df_val_processed = df_val_processed[~df_val_processed['discharge_disposition_id'].isin(discharge_ids_to_remove)]
            print(f"Validation data rows after hospice/expired filter: {len(df_val_processed)}")

        # Apply the same target variable transformation to validation data
        df_val_processed['readmitted'] = df_val_processed['readmitted'].replace(['<30', '>30'], 'YES')
        df_val_processed['readmitted_binary'] = df_val_processed['readmitted'].apply(lambda x: 1 if x == 'YES' else 0)
        if 'age' in df_val_processed.columns:
             df_val_processed['age_ordinal'] = df_val_processed['age'].map(age_mapping)
        existing_cols_to_drop_val = [col for col in cols_to_drop_engineered if col in df_val_processed.columns]
        df_val_processed.drop(columns=existing_cols_to_drop_val, inplace=True)
        
        y_val = df_val_processed['readmitted_binary']
        X_val = df_val_processed.drop(columns=['readmitted_binary'])
        for col_id in potential_categorical_ids: # Ensure cat type for IDs
            if col_id in X_val.columns and X_val[col_id].dtype != 'object' and X_val[col_id].dtype != 'category':
                 X_val[col_id] = X_val[col_id].astype('category')
        
        X_val_prepared = preprocessor.transform(X_val)
        print(f"Validation data transformed. Shape: {X_val_prepared.shape}")

    if df_test_initial is not None:
        df_test_processed = df_test_initial.copy()
        for col in df_test_processed.columns: # Replace '?'
             if df_test_processed[col].dtype == 'object':
                df_test_processed[col] = df_test_processed[col].replace('?', np.nan)
        if cols_to_drop_missing: # Drop same high-missing columns
            df_test_processed.drop(columns=[col for col in cols_to_drop_missing if col in df_test_processed.columns], inplace=True)
        for col in ['race', 'diag_1', 'diag_2', 'diag_3']: # Fill NaNs
            if col in df_test_processed.columns: df_test_processed[col].fillna('Missing', inplace=True)

        # Apply discharge disposition filtering
        if 'discharge_disposition_id' in df_test_processed.columns:
            df_test_processed = df_test_processed[~df_test_processed['discharge_disposition_id'].isin(discharge_ids_to_remove)]
            print(f"Test data rows after hospice/expired filter: {len(df_test_processed)}")

        # Apply the same target variable transformation to test data
        df_test_processed['readmitted'] = df_test_processed['readmitted'].replace(['<30', '>30'], 'YES')
        df_test_processed['readmitted_binary'] = df_test_processed['readmitted'].apply(lambda x: 1 if x == 'YES' else 0)
        if 'age' in df_test_processed.columns:
            df_test_processed['age_ordinal'] = df_test_processed['age'].map(age_mapping)
        existing_cols_to_drop_test = [col for col in cols_to_drop_engineered if col in df_test_processed.columns]
        df_test_processed.drop(columns=existing_cols_to_drop_test, inplace=True)

        y_test = df_test_processed['readmitted_binary']
        X_test = df_test_processed.drop(columns=['readmitted_binary'])
        for col_id in potential_categorical_ids: # Ensure cat type for IDs
            if col_id in X_test.columns and X_test[col_id].dtype != 'object' and X_test[col_id].dtype != 'category':
                 X_test[col_id] = X_test[col_id].astype('category')

        X_test_prepared = preprocessor.transform(X_test)
        print(f"Test data transformed. Shape: {X_test_prepared.shape}")

# %% [markdown]
# ## 6. Train Baseline Model (Logistic Regression)

# %%
baseline_model = None
if X_train_prepared is not None and y_train is not None:
    print("\n--- Training Baseline Model (Logistic Regression) ---")
    baseline_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # class_weight for imbalance
    baseline_model.fit(X_train_prepared, y_train)
    print("Baseline model trained.")
else:
    print("Training data not prepared. Skipping model training.")

# %% [markdown]
# ## 7. Evaluate Baseline Model

# %%
if baseline_model is not None and X_test_prepared is not None and y_test is not None:
    print("\n--- Evaluating Baseline Model on Test Set ---")
    y_pred_test = baseline_model.predict(X_test_prepared)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    
    print("Accuracy Score (Test Set):")
    print(accuracy_score(y_test, y_pred_test))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Readmitted', 'Readmitted'], 
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.show()
else:
    print("Model not trained or test data not prepared. Skipping evaluation.")

# %% [markdown]
# ## 8. Observations, Conclusions & Next Steps
# 
# *   **EDA Summary:** Briefly summarize key findings from EDA (missing values, important features noted, target distribution).
# *   **Data Quality Issues:** Note any significant data quality problems (e.g., high cardinality in `diag` codes, extensive missingness in `weight`, `payer_code`, `medical_specialty`). The decision to drop `diag_1,2,3` for the baseline simplifies things but loses potentially valuable info. The filtering of expired/hospice patients aligns with the original study's preprocessing.
# *   **Baseline Model Performance:** Discuss the performance of Logistic Regression. Is it better than random chance? What do precision/recall for each class indicate, especially given potential class imbalance?
# *   **Limitations of Baseline:**
#     *   Simple feature engineering (e.g., ordinal for age, dropping complex categoricals like diagnoses).
#     *   Basic missing value handling (dropping columns, 'Missing' category).
#     *   Only one model tried, no hyperparameter tuning.
#     *   Class imbalance potentially not fully addressed by `class_weight='balanced'` alone.
#     *   **First Encounter per Patient:** The original study ("data_summary.md") used only the first encounter per patient for statistical independence. This baseline does not currently implement this; `initial_train.csv` might contain multiple encounters for the same patient. This could be addressed in future iterations for more rigorous analysis.
# *   **Recommendations for Next Iterations:**
#     *   More sophisticated feature engineering for `diag_1, diag_2, diag_3` (e.g., grouping into broader categories based on medical knowledge like ICD-9 codes, using embeddings if applicable, or target encoding with care). The original study noted the importance of primary diagnosis in conjunction with HbA1c testing.
#     *   Explore imputation techniques for moderately missing features instead of dropping or just using 'Missing'.
#     *   Try other models (e.g., Random Forest, Gradient Boosting).
#     *   Perform hyperparameter tuning (e.g., using GridSearchCV or RayTune as planned for later).
#     *   Address class imbalance more robustly (e.g., SMOTE, ADASYN for oversampling, or different model cost functions).
#     *   Feature selection techniques.
#     *   **Patient-Level Features:** Deeper investigation into `patient_nbr` to engineer features related to prior visits or patient history, if appropriate after considering the "first encounter" rule.
#     *   **HbA1c Measurement Feature:** Explicitly engineer a feature for "HbA1c measured vs. not measured", as the original study highlighted the importance of the test being performed, not just the result.
#     *   Log experiments with MLflow.

# %% [markdown]
# ---
# End of Initial EDA and Baseline Model Script.
# Remember to commit this script to Git.
# Consider stopping the EC2 instance if not actively using it to save costs.
# ---

# %% [markdown]
# ## 9. Model Run on Full Dataset (80/20 Split)
# 
# This section loads the full dataset, applies the same preprocessing as the initial 20% run,
# splits it into 80% training and 20% test, and then trains and evaluates the baseline Logistic Regression model.

# %%
print("\\n\\n--- Starting Full Dataset Run (80/20 Split) ---")

# %% [markdown]
# ### 9.1 Load Full Raw Data

# %%
df_full_raw = load_df_from_s3(S3_BUCKET_NAME, FULL_RAW_DATA_KEY, s3_client)

# %% [markdown]
# ### 9.2 Data Cleaning & Preprocessing (Full Dataset)

# %%
df_full_processed = None
if df_full_raw is not None:
    df_full_processed = df_full_raw.copy()
    
    # 4.1 Handling '?' and Special Values (Applied to full dataset)
    print("Replacing '?' with NaN globally (Full Dataset)...")
    for col in df_full_processed.columns:
        if df_full_processed[col].dtype == 'object':
            df_full_processed[col] = df_full_processed[col].replace('?', np.nan)
    
    if 'race' in df_full_processed.columns:
        print(f"Value counts for 'race' after replacing '?' (Full Dataset):\n{df_full_processed['race'].value_counts(dropna=False)}")

    # 4.2 Handling Missing Values (Based on EDA, applied to full dataset)
    print("\\n--- Initial Missing Values (Full Dataset, Post '?' replacement) ---")
    missing_values_full = df_full_processed.isnull().sum()
    missing_percentage_full = (missing_values_full / len(df_full_processed)) * 100
    missing_info_full = pd.DataFrame({'Missing Count': missing_values_full, 'Missing Percentage': missing_percentage_full})
    print(missing_info_full[missing_info_full['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False))

    # Re-define cols_to_drop_missing based on the full dataset's characteristics, or use the same ones from 20% run.
    # For consistency, we'll use the same column names identified from the 20% data for now.
    # These were 'weight', 'payer_code'. 'medical_specialty' was also considered but its threshold might differ.
    # The original script had:
    # cols_to_drop_missing.append('weight')
    # cols_to_drop_missing.append('payer_code')
    # medical_specialty was dropped if > 40% missing. Let's re-evaluate for full data
    
    cols_to_drop_missing_full = []
    if 'weight' in df_full_processed.columns and (df_full_processed['weight'].isnull().sum() / len(df_full_processed)) > 0.9:
        cols_to_drop_missing_full.append('weight')
    if 'payer_code' in df_full_processed.columns and (df_full_processed['payer_code'].isnull().sum() / len(df_full_processed)) > 0.4: # Using 40% as in original script
        cols_to_drop_missing_full.append('payer_code')
    if 'medical_specialty' in df_full_processed.columns and (df_full_processed['medical_specialty'].isnull().sum() / len(df_full_processed)) > 0.4: # Using 40% as in original script
        cols_to_drop_missing_full.append('medical_specialty')

    if cols_to_drop_missing_full:
        print(f"\\nDropping columns due to high missing values (Full Dataset): {cols_to_drop_missing_full}")
        df_full_processed.drop(columns=cols_to_drop_missing_full, inplace=True)
    
    print("\\n--- Missing Values After Dropping High-Missing Columns (Full Dataset) ---")
    missing_values_after_drop_full = df_full_processed.isnull().sum()
    missing_info_after_drop_full = pd.DataFrame({
        'Missing Count': missing_values_after_drop_full[missing_values_after_drop_full > 0],
        'Missing Percentage': (missing_values_after_drop_full[missing_values_after_drop_full > 0] / len(df_full_processed)) * 100
    })
    print(missing_info_after_drop_full.sort_values(by='Missing Percentage', ascending=False))
        
    for col in ['race', 'diag_1', 'diag_2', 'diag_3']:
        if col in df_full_processed.columns and df_full_processed[col].isnull().any():
            print(f"Filling NaNs in '{col}' with 'Missing' (Full Dataset)")
            df_full_processed[col].fillna('Missing', inplace=True)

    # Discharge disposition filtering (same codes as before)
    # expired_ids = [11, 19, 20, 21]
    # hospice_ids = [13, 14]
    # discharge_ids_to_remove = expired_ids + hospice_ids # This was defined in the 20% run scope
    
    # Re-define for clarity or ensure it's accessible; assuming it's the same fixed list.
    discharge_ids_to_remove_full_run = [11, 13, 14, 19, 20, 21] 
    if 'discharge_disposition_id' in df_full_processed.columns:
        initial_rows_full = len(df_full_processed)
        # Ensure column is numeric before filtering
        df_full_processed['discharge_disposition_id'] = pd.to_numeric(df_full_processed['discharge_disposition_id'], errors='coerce')
        df_full_processed = df_full_processed[~df_full_processed['discharge_disposition_id'].isin(discharge_ids_to_remove_full_run)]
        rows_removed_full = initial_rows_full - len(df_full_processed)
        print(f"Removed {rows_removed_full} rows due to discharge to hospice or expired (Full Dataset).")
    
    print("\\n--- Final Check for Missing Values (Full Dataset) ---")
    final_missing_full = df_full_processed.isnull().sum().sum()
    if final_missing_full > 0:
        print(f"{final_missing_full} total missing values remaining before modeling (Full Dataset).")
        print(df_full_processed.isnull().sum()[df_full_processed.isnull().sum() > 0])
    else:
        print("No missing values remaining before modeling (Full Dataset).")

    # 4.3 Feature Engineering (Applied to full dataset)
    # 1. Simplify Target Variable 'readmitted' to ANY readmission
    print("\\n--- Simplifying Target Variable 'readmitted' to ANY readmission (Full Dataset) ---")
    df_full_processed['readmitted'] = df_full_processed['readmitted'].replace(['<30', '>30'], 'YES')
    df_full_processed['readmitted_binary'] = df_full_processed['readmitted'].apply(lambda x: 1 if x == 'YES' else 0)
    print(df_full_processed['readmitted_binary'].value_counts(normalize=True))
    
    # 2. Process 'age' column
    print("\\n--- Processing 'age' column (Full Dataset) ---")
    if 'age' in df_full_processed.columns:
        # age_mapping defined in the 20% run's scope. Make sure it's accessible or redefine.
        age_mapping_full_run = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
        df_full_processed['age_ordinal'] = df_full_processed['age'].map(age_mapping_full_run)
        print(df_full_processed[['age', 'age_ordinal']].head())
    
    # 3. Drop original 'readmitted' and 'age' columns, and identifiers
    # cols_to_drop_engineered defined in 20% run. Redefine for clarity or ensure accessible.
    cols_to_drop_engineered_full_run = ['readmitted', 'age', 'encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3']
    
    existing_cols_to_drop_full = [col for col in cols_to_drop_engineered_full_run if col in df_full_processed.columns]
    print(f"\\nDropping columns (Full Dataset): {existing_cols_to_drop_full}")
    df_full_processed.drop(columns=existing_cols_to_drop_full, inplace=True)
    
    print("\\n--- DataFrame after Feature Engineering (Full Dataset) ---")
    print(df_full_processed.head())
    df_full_processed.info()
else:
    print("Full raw data not loaded. Skipping full dataset run.")

# %% [markdown]
# ### 9.3 Train-Test Split (Full Dataset)

# %%
X_full_train_prepared = None
y_full_train = None
X_full_test_prepared = None
y_full_test = None
preprocessor_full = None

if df_full_processed is not None and not df_full_processed.empty:
    y_full = df_full_processed['readmitted_binary']
    X_full = df_full_processed.drop(columns=['readmitted_binary'])

    X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full # Stratify for imbalanced classes
    )
    print(f"Full dataset split into training and testing sets:")
    print(f"X_full_train shape: {X_full_train.shape}, y_full_train shape: {y_full_train.shape}")
    print(f"X_full_test shape: {X_full_test.shape}, y_full_test shape: {y_full_test.shape}")

    # Identify categorical and numerical features for the full dataset
    # Re-using potential_categorical_ids from the 20% run's scope or redefining.
    potential_categorical_ids_full_run = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    
    categorical_features_full = X_full_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col_id in potential_categorical_ids_full_run:
        if col_id in X_full_train.columns and X_full_train[col_id].dtype != 'object' and X_full_train[col_id].dtype != 'category':
            if X_full_train[col_id].nunique() < 30: 
                 categorical_features_full.append(col_id)
                 X_full_train[col_id] = X_full_train[col_id].astype('category') # Apply to train
                 X_full_test[col_id] = X_full_test[col_id].astype('category')   # Apply to test

    numerical_features_full = X_full_train.select_dtypes(include=np.number).columns.tolist()
    numerical_features_full = [col for col in numerical_features_full if col not in categorical_features_full]
    
    categorical_features_full = list(set(categorical_features_full))
    processed_cols_full = set(categorical_features_full + numerical_features_full)
    all_cols_full = set(X_full_train.columns)

    if processed_cols_full != all_cols_full:
        print(f"Warning (Full Dataset): Column mismatch. Untracked columns: {all_cols_full - processed_cols_full}")
        for rem_col in (all_cols_full - processed_cols_full):
            if X_full_train[rem_col].dtype == 'object':
                categorical_features_full.append(rem_col)
            else:
                numerical_features_full.append(rem_col)
        categorical_features_full = list(set(categorical_features_full))
        numerical_features_full = list(set(numerical_features_full) - set(categorical_features_full))

    print(f"\\nIdentified Numerical Features (Full Dataset): {numerical_features_full}")
    print(f"Identified Categorical Features (Full Dataset): {categorical_features_full}")

    preprocessor_full = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_full),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features_full)
        ], 
        remainder='passthrough'
    )

    print("\\nFitting preprocessor and transforming training data (Full Dataset)...")
    X_full_train_prepared = preprocessor_full.fit_transform(X_full_train)
    X_full_test_prepared = preprocessor_full.transform(X_full_test)
    
    print(f"Full training data transformed. Shape: {X_full_train_prepared.shape}")
    print(f"Full test data transformed. Shape: {X_full_test_prepared.shape}")

else:
    print("Full processed data not available. Skipping model training and evaluation for full dataset.")

# %% [markdown]
# ### 9.4 Train Baseline Model (Logistic Regression - Full Dataset)

# %%
baseline_model_full = None
if X_full_train_prepared is not None and y_full_train is not None:
    print("\\n--- Training Baseline Model (Logistic Regression - Full Dataset) ---")
    baseline_model_full = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    baseline_model_full.fit(X_full_train_prepared, y_full_train)
    print("Baseline model trained (Full Dataset).")
else:
    print("Full training data not prepared. Skipping model training (Full Dataset).")

# %% [markdown]
# ### 9.5 Evaluate Baseline Model (Full Dataset)

# %%
if baseline_model_full is not None and X_full_test_prepared is not None and y_full_test is not None:
    print("\\n--- Evaluating Baseline Model on Test Set (Full Dataset) ---")
    y_pred_full_test = baseline_model_full.predict(X_full_test_prepared)
    
    print("\\nClassification Report (Test Set - Full Dataset):")
    print(classification_report(y_full_test, y_pred_full_test))
    
    print("Accuracy Score (Test Set - Full Dataset):")
    print(accuracy_score(y_full_test, y_pred_full_test))
    
    print("\\nConfusion Matrix (Test Set - Full Dataset):")
    cm_full = confusion_matrix(y_full_test, y_pred_full_test)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Readmitted', 'Readmitted'], 
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set (Full Dataset)')
    plt.show()
else:
    print("Model not trained or full test data not prepared. Skipping evaluation (Full Dataset).")

# %% [markdown]
# ---
# End of Full Dataset Run section.
# ---
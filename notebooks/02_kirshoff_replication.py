#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Kirshoff Diabetes Readmission Replication
# 
# This script attempts to replicate the baseline Random Forest performance (F1 ~0.76) reported in Kirshoff's Kaggle notebook for predicting hospital readmission within 30 days.
# The analysis is based on the summary report: `data/kirshoff results - Diabetes.md`.

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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, f1_score
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
    # The UCI dataset description mentions 3 such records. Kirshoff's report implies these are removed.
    if 'gender' in df.columns:
        print(f"\n--- Handling 'gender' ---")
        print(f"Initial gender value counts:\n{df['gender'].value_counts(dropna=False)}")
        df = df[~df['gender'].isin(['Unknown/Invalid'])]
        print(f"Shape after removing 'Unknown/Invalid' gender: {df.shape}")
        print(f"Gender value counts after removal:\n{df['gender'].value_counts(dropna=False)}")


    # --- 2.3 Drop specified columns (high missingness, identifiers, low variance, complex) ---
    # Based on Kirshoff's report and standard practice
    cols_to_drop = [
        'encounter_id', 'patient_nbr',  # Identifiers
        'weight', 'payer_code', 'medical_specialty', # High missingness as per Kirshoff
        'citoglipton', 'examide', # Low variance as per Kirshoff
        # Diag columns are dropped for simplicity in Kirshoff's baseline RF context as per report
        # This was also a step in our previous EDA/pipeline for baseline simplicity.
        'diag_1', 'diag_2', 'diag_3' 
    ]
    # Ensure columns exist before dropping
    actual_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    print(f"\n--- Dropping columns: {actual_cols_to_drop} ---")
    df.drop(columns=actual_cols_to_drop, inplace=True)
    print(f"Shape after dropping columns: {df.shape}")

    # --- 2.4 Handle Discharge Disposition (remove hospice/expired) ---
    # Codes from UCI data description & common practice. Kirshoff likely did this.
    expired_hospice_dispositions = [11, 13, 14, 19, 20, 21]
    if 'discharge_disposition_id' in df.columns:
        print(f"\n--- Filtering by 'discharge_disposition_id' ---")
        initial_rows_discharge = len(df)
        df = df[~df['discharge_disposition_id'].isin(expired_hospice_dispositions)]
        rows_removed_discharge = initial_rows_discharge - len(df)
        print(f"Removed {rows_removed_discharge} rows due to hospice/expired discharge. Shape: {df.shape}")

    # --- 2.5 Target Variable: 'readmitted_binary' (<30 days) ---
    print(f"\n--- Engineering target variable 'readmitted_binary' ---")
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
        df.drop(columns=['readmitted'], inplace=True)
        print(f"Target variable 'readmitted_binary' created. Distribution:\n{df['readmitted_binary'].value_counts(normalize=True)}")
    else:
        raise ValueError("Target column 'readmitted' not found.")

    # --- 2.6 Feature Engineering: 'age' to ordinal ---
    print(f"\n--- Engineering 'age_ordinal' ---")
    if 'age' in df.columns:
        age_mapping = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
        df['age_ordinal'] = df['age'].map(age_mapping)
        df.drop(columns=['age'], inplace=True)
        print("'age_ordinal' created.")
    
    # --- 2.7 Handle remaining NaNs in categorical features ---
    # Kirshoff's report mentions "imputes missing categorical values with "Unknown" or mode".
    # For simplicity and broad application, we'll fill with "Missing" which OHE will handle.
    print(f"\n--- Filling remaining NaNs in object columns with 'Missing' ---")
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            print(f"Filling NaNs in {col}")
            df[col].fillna('Missing', inplace=True)
            
    # --- 2.8 Final check for NaNs in numerical features (should be few or none after drops) ---
    numerical_nans = df.select_dtypes(include=np.number).isnull().sum()
    print(f"\n--- NaN counts in numerical columns post-cleaning ---")
    print(numerical_nans[numerical_nans > 0])
    # If any remain, simple median imputation
    for col in numerical_nans[numerical_nans > 0].index:
        print(f"Imputing NaNs in numerical column {col} with median.")
        df[col].fillna(df[col].median(), inplace=True)

    print(f"\nFinal dataset shape for modeling: {df.shape}") # Should be around 66k as per Kirshoff
    # Kirshoff report mentioned 66,091 samples. This might vary slightly based on interpretation of cleaning.
    # My prior EDA showed that `discharge_disposition_id` filtering removed ~5k rows from the full dataset.
    # And `gender == 'Unknown/Invalid'` removes 3 rows.
    # The initial dataset has 101766 rows.
    # 101766 - 3 (gender) - ~5000 (discharge) = ~96763.
    # The 66k number from the report might be after more aggressive feature-specific NaN dropping or other criteria.
    # We will proceed with this cleaned set.
    
    # Report final missing values
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"WARNING: {total_missing} missing values remain in the dataset.")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("No missing values remain in the dataset.")

    processing_time = time.time() - start_time
    print(f"Data cleaning and preprocessing completed in {processing_time:.2f} seconds.")

else:
    print("Raw data not loaded. Halting script.")
    df = None


# %% [markdown]
# ## 3. Feature Preprocessing for Model

# %%
if df is not None:
    y = df['readmitted_binary']
    X = df.drop(columns=['readmitted_binary'])

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure 'id' columns that are numeric but represent categories are treated as categorical
    # Example: admission_type_id, discharge_disposition_id, admission_source_id
    potential_cat_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    for col_id in potential_cat_ids:
        if col_id in numerical_features:
            print(f"Treating numeric ID column '{col_id}' as categorical.")
            X[col_id] = X[col_id].astype(str) # Convert to object to be picked by OHE
            numerical_features.remove(col_id)
            if col_id not in categorical_features:
                 categorical_features.append(col_id)
    
    # Re-identify after type conversion
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


    print(f"\nFinal Numerical Features for Preprocessor: {numerical_features}")
    print(f"Final Categorical Features for Preprocessor: {categorical_features}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' 
    )
else:
    preprocessor = None
    X = None
    y = None


# %% [markdown]
# ## 4. Train-Test Split

# %%
if X is not None and y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split into training and testing sets:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    # Fit preprocessor on training data and transform both sets
    print("\nFitting preprocessor on X_train...")
    X_train_processed = preprocessor.fit_transform(X_train)
    print("Transforming X_test...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    try:
        num_feature_names = numerical_features
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        processed_feature_names = list(num_feature_names) + list(cat_feature_names)
        
        # Handle remainder='passthrough' columns if any
        # Get names of remainder columns (should be none if all features are numerical or categorical)
        processed_cols_in_transformer = set()
        for name, trans, cols_in_trans in preprocessor.transformers_:
            if trans != 'drop': # only consider columns that are not dropped
                 processed_cols_in_transformer.update(cols_in_trans)
        
        remainder_cols = [col for col in X_train.columns if col not in processed_cols_in_transformer]
        if remainder_cols:
            print(f"Warning: Remainder columns found: {remainder_cols}")
            processed_feature_names.extend(remainder_cols)

        X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
        print(f"X_train_processed_df shape: {X_train_processed_df.shape}")
        print(f"X_test_processed_df shape: {X_test_processed_df.shape}")

    except Exception as e:
        print(f"Could not get feature names: {e}. Using numpy arrays.")
        X_train_processed_df = X_train_processed # Keep as numpy array
        X_test_processed_df = X_test_processed  # Keep as numpy array
else:
    X_train_processed_df = None
    X_test_processed_df = None


# %% [markdown]
# ## 5. Train Random Forest Model

# %%
model = None
if X_train_processed_df is not None:
    print("\n--- Applying SMOTE to training data ---")
    smote = SMOTE(random_state=42)
    # Ensure we use the numpy array version if DataFrame creation failed
    X_train_input = X_train_processed_df.values if isinstance(X_train_processed_df, pd.DataFrame) else X_train_processed
    X_train_smote, y_train_smote = smote.fit_resample(X_train_input, y_train)
    print(f"Shape after SMOTE: X_train_smote: {X_train_smote.shape}, y_train_smote: {y_train_smote.shape}")
    print(f"Original training target distribution:\n{y_train.value_counts(normalize=True)}")
    # Need pandas Series to use value_counts easily
    y_train_smote_series = pd.Series(y_train_smote) 
    print(f"SMOTE training target distribution:\n{y_train_smote_series.value_counts(normalize=True)}")

    print("\n--- Training Random Forest Model (Kirshoff Replication Attempt with SMOTE) ---")
    # Kirshoff report mentions Random Forest with default params (e.g., 100 trees) and class_weight='balanced'
    # NOTE: When using SMOTE, class_weight='balanced' is typically NOT needed/recommended 
    # as SMOTE already balances the dataset itself. We will remove it.
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    fit_start_time = time.time()
    # <<< Train on SMOTE data >>>
    model.fit(X_train_smote, y_train_smote) 
    fit_time = time.time() - fit_start_time
    print(f"Model training completed in {fit_time:.2f} seconds.")
else:
    print("Processed training data not available. Skipping model training.")

# %% [markdown]
# ## 6. Evaluate Model

# %%
if model is not None and X_test_processed_df is not None:
    print("\n--- Evaluating Model on Test Set ---")
    # <<< Ensure evaluation uses the original, unprocessed test data >>>
    X_test_input = X_test_processed_df.values if isinstance(X_test_processed_df, pd.DataFrame) else X_test_processed
    y_pred_test = model.predict(X_test_input)
    y_proba_test = model.predict_proba(X_test_input)[:, 1] # Probabilities for AUC

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Not Readmitted <30', 'Readmitted <30']))
    
    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test) # F1 for the positive class (1) by default
    auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"Accuracy Score (Test Set): {accuracy:.4f}")
    print(f"F1 Score (Positive Class, Test Set): {f1:.4f}") # Kirshoff reported 0.76
    print(f"AUC-ROC Score (Test Set): {auc:.4f}") # Kirshoff reported 0.74
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Not Readmitted', 'Pred Readmitted'], 
                yticklabels=['Actual Not Readmitted', 'Actual Readmitted'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set (Kirshoff Replication)')
    plt.show()
else:
    print("Model not trained or processed test data not available. Skipping evaluation.")

# %% [markdown]
# ## 7. Learnings and Report (Placeholder)
# 
# *   Compare results with Kirshoff's reported F1 of 0.76.
# *   Detail differences in preprocessing that might affect the outcome.
# *   Note any challenges during replication.

# %%
print("\nScript execution finished.") 
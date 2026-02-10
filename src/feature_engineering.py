"""Feature engineering pipeline for the Health Predict system.

Provides the three-step preprocessing pipeline used by both the
training DAG and the FastAPI prediction service:

  1. ``clean_data()`` — Replace '?' markers with NaN, drop columns with
     excessive missing values (weight, payer_code, medical_specialty)
     and patient identifiers, fill remaining NaN with 'Unknown', and
     filter out expired/hospice patients whose readmission status is
     undefined.
  2. ``engineer_features()`` — Create the binary target variable
     ``readmitted_binary`` (0 = not readmitted, 1 = readmitted) and
     convert the age bracket strings to ordinal integers (0–9).
  3. ``preprocess_data()`` — Apply a fitted ``ColumnTransformer`` that
     StandardScales numeric features and OneHotEncodes categoricals.

Helper functions ``get_preprocessor()``, ``save_preprocessor()``, and
``load_preprocessor()`` manage the sklearn pipeline artifact.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# Columns to drop due to high missing values or other reasons
COLS_TO_DROP_INITIAL = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
# Columns where NaN will be filled with 'Missing'
COLS_FILL_NA_MISSING = ['race', 'diag_1', 'diag_2', 'diag_3'] # In EDA, diag_1, diag_2, diag_3 were dropped for baseline, but keeping here for more general FE pipeline

# Discharge dispositions indicating expired or hospice - to be removed
EXPIRED_HOSPICE_DISPOSITIONS = [11, 13, 14, 19, 20, 21]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps to the dataframe.

    Steps performed:
      - Replace '?' with NaN in all object-type columns.
      - Drop high-missing columns defined in COLS_TO_DROP_INITIAL
        (weight, payer_code, medical_specialty, encounter_id, patient_nbr).
      - Fill remaining NaN in COLS_FILL_NA_MISSING with 'Unknown'.
      - Remove rows where discharge_disposition_id indicates
        expired or hospice patients (IDs 11, 13, 14, 19, 20, 21).
    """
    df_cleaned = df.copy()

    # Replace '?' with NaN globally
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].replace('?', np.nan)

    # Drop specified columns
    cols_to_drop_present = [col for col in COLS_TO_DROP_INITIAL if col in df_cleaned.columns]
    df_cleaned.drop(columns=cols_to_drop_present, inplace=True, errors='ignore')

    # Fill NaNs in specified columns with 'Unknown'
    for col in COLS_FILL_NA_MISSING:
        if col in df_cleaned.columns:
            df_cleaned[col].fillna('Unknown', inplace=True)
            
    # Filter out rows based on discharge_disposition_id
    if 'discharge_disposition_id' in df_cleaned.columns:
        df_cleaned = df_cleaned[~df_cleaned['discharge_disposition_id'].isin(EXPIRED_HOSPICE_DISPOSITIONS)]

    return df_cleaned

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering steps.

    Creates ``readmitted_binary`` (0/1) from the three-class
    ``readmitted`` column and converts the age bracket string
    (e.g., '[70-80)') to an ordinal integer ``age_ordinal`` (0–9).
    Original columns are preserved for downstream flexibility.
    """
    df_featured = df.copy()

    # Create binary target variable 'readmitted_binary'
    if 'readmitted' in df_featured.columns:
        df_featured['readmitted_binary'] = df_featured['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
        # Drop original 'readmitted' as we now have the binary target
        # df_featured.drop(columns=['readmitted'], inplace=True) # Keep original for now, trainer can decide

    # Convert 'age' to ordinal
    if 'age' in df_featured.columns:
        age_mapping = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
        df_featured['age_ordinal'] = df_featured['age'].map(age_mapping)
        # Drop original 'age' column
        # df_featured.drop(columns=['age'], inplace=True) # Keep original for now

    # Drop original diag_1, diag_2, diag_3 as per EDA baseline for simplicity (if they weren't filled with 'Missing')
    # For a more robust pipeline, these would be handled differently (e.g., ICD9 code grouping)
    # cols_to_drop_diag = ['diag_1', 'diag_2', 'diag_3']
    # df_featured.drop(columns=[col for col in cols_to_drop_diag if col in df_featured.columns], inplace=True, errors='ignore')
    
    return df_featured

def get_preprocessor(df: pd.DataFrame, numerical_features: list = None, categorical_features: list = None) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for preprocessing numerical and categorical features.
    Detects feature types if not provided.
    """
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        # Try to exclude target if it was left as numeric
        if 'readmitted_binary' in numerical_features:
            numerical_features.remove('readmitted_binary')
        if 'readmitted' in numerical_features: # if original target is still numeric for some reason
             numerical_features.remove('readmitted')


    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Try to exclude target if it was left as object/category
        if 'readmitted_binary' in categorical_features: # Should not happen if it's binary numeric
            categorical_features.remove('readmitted_binary')
        if 'readmitted' in categorical_features:
            categorical_features.remove('readmitted')


    # Ensure no overlap and features exist
    valid_numerical_features = [col for col in numerical_features if col in df.columns]
    valid_categorical_features = [col for col in categorical_features if col in df.columns and col not in valid_numerical_features]
    
    print(f"Detected/Provided Numerical features for preprocessor: {valid_numerical_features}")
    print(f"Detected/Provided Categorical features for preprocessor: {valid_categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), valid_numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_categorical_features) # sparse_output=False for easier use with pandas
        ],
        remainder='passthrough' # Keep other columns (e.g., target, identifiers if not dropped)
    )
    return preprocessor

def save_preprocessor(preprocessor: ColumnTransformer, file_path: str):
    """Saves the preprocessor to a file."""
    dir_name = os.path.dirname(file_path)
    if dir_name: # Only call makedirs if dirname is not empty
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(preprocessor, file_path)
    print(f"Preprocessor saved to {file_path}")

def load_preprocessor(file_path: str) -> ColumnTransformer:
    """Loads the preprocessor from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
    preprocessor = joblib.load(file_path)
    print(f"Preprocessor loaded from {file_path}")
    return preprocessor

def preprocess_data(df: pd.DataFrame, preprocessor: ColumnTransformer, fit_preprocessor: bool = False) -> pd.DataFrame:
    """
    Applies the preprocessor to the dataframe.
    If fit_preprocessor is True, it fits the preprocessor first.
    Returns a DataFrame with transformed data and original non-transformed columns.
    """
    if fit_preprocessor:
        print("Fitting preprocessor...")
        preprocessor.fit(df)

    print("Transforming data with preprocessor...")
    transformed_data = preprocessor.transform(df)
    
    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception: # Older sklearn versions might not have get_feature_names_out for passthrough
        # Create names manually for num and cat, then add remainder
        num_features = preprocessor.transformers_[0][2]
        cat_features_transformed = preprocessor.named_transformers_['cat'].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_features) + list(cat_features_transformed)
        
        if preprocessor.remainder == 'passthrough':
            # Get names of remainder columns
            processed_cols = set()
            for name, trans, cols in preprocessor.transformers_:
                if trans != 'drop':
                    processed_cols.update(cols)
            
            remainder_cols = [col for col in df.columns if col not in processed_cols]
            feature_names.extend(remainder_cols)


    df_transformed = pd.DataFrame(transformed_data, columns=feature_names, index=df.index)
    print(f"Data transformed. Shape: {df_transformed.shape}")
    return df_transformed


if __name__ == '__main__':
    # Example Usage (mainly for testing the script directly)
    print("Feature engineering script direct execution (for testing)...")

    # Create a dummy dataframe similar to the project's data
    data = {
        'encounter_id': [1, 2, 3, 4, 5, 6],
        'patient_nbr': [101, 102, 103, 104, 105, 106],
        'race': ['Caucasian', 'AfricanAmerican', '?', 'Caucasian', 'Asian', 'Hispanic'],
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'age': ['[70-80)', '[50-60)', '[70-80)', '[80-90)', '[40-50)', '[60-70)'],
        'weight': ['?', '?', '[75-100)', '?', '?', '?'], # High missing
        'admission_type_id': [1, 2, 3, 1, 2, 3],
        'discharge_disposition_id': [1, 1, 11, 1, 1, 13], # 11, 13 are hospice/expired
        'admission_source_id': [7, 7, 7, 7, 7, 7],
        'time_in_hospital': [3, 5, 2, 7, 4, 6],
        'payer_code': ['?', 'MC', '?', 'SP', '?', 'DM'], # High missing
        'medical_specialty': ['?', 'Cardiology', '?', '?', 'InternalMedicine', '?'], # High missing
        'num_lab_procedures': [50, 45, 60, 55, 40, 65],
        'num_procedures': [0, 1, 0, 2, 1, 0],
        'num_medications': [10, 15, 12, 20, 8, 18],
        'diag_1': ['250.83', '428', '?', '786', '401', '250'],
        'diag_2': ['401', '250.01', 'V45', '?', '276', '427'],
        'diag_3': ['250', 'V58.67', 'E878', '414', '428', '707'],
        'readmitted': ['<30', 'NO', '>30', '<30', 'NO', 'NO']
    }
    sample_df = pd.DataFrame(data)
    print(f"\nOriginal sample_df shape: {sample_df.shape}")
    print(sample_df.head(2))

    # 1. Clean data
    df_cleaned = clean_data(sample_df)
    print(f"\ndf_cleaned shape: {df_cleaned.shape}")
    print(df_cleaned.head(2))
    print(f"'race' in cleaned df: {df_cleaned['race'].value_counts(dropna=False)}")
    assert 'weight' not in df_cleaned.columns
    assert len(df_cleaned[df_cleaned['discharge_disposition_id'].isin(EXPIRED_HOSPICE_DISPOSITIONS)]) == 0


    # 2. Engineer features
    df_featured = engineer_features(df_cleaned)
    print(f"\ndf_featured shape: {df_featured.shape}")
    print(df_featured[['age', 'age_ordinal', 'readmitted', 'readmitted_binary']].head())
    assert 'age_ordinal' in df_featured.columns
    assert 'readmitted_binary' in df_featured.columns

    # 3. Define features for preprocessor (excluding target and original complex features)
    # The actual training script will likely pass specific lists based on context
    # For this test, we rely on auto-detection within get_preprocessor, then filter
    
    # For testing, let's prepare a dataframe that only has features intended for the preprocessor
    # and the target variable which preprocessor should passthrough.
    # Columns like 'readmitted' (original categorical target) and 'age' (original categorical age)
    # would be dropped if their engineered versions ('readmitted_binary', 'age_ordinal') are used.
    
    # Identify columns that are not the target or original versions of engineered features
    cols_for_modeling = [col for col in df_featured.columns if col not in ['readmitted', 'age']]
    df_model_input = df_featured[cols_for_modeling].copy()
    
    # Separate features (X) and target (y) for fitting preprocessor
    X_test = df_model_input.drop(columns=['readmitted_binary'], errors='ignore')
    y_test = df_model_input['readmitted_binary'] if 'readmitted_binary' in df_model_input else None
    
    print(f"\nShape of X_test for preprocessor fitting: {X_test.shape}")
    print(X_test.head(2))
    X_test.info()


    # 4. Get and fit preprocessor
    # Explicitly define numerical and categorical columns for the preprocessor
    # Based on df_featured after cleaning and engineering, and excluding target/original cols.
    
    numerical_cols_test = X_test.select_dtypes(include=np.number).columns.tolist()
    if 'age_ordinal' in numerical_cols_test and 'age' in numerical_cols_test : numerical_cols_test.remove('age') # Should already be gone
    
    categorical_cols_test = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'gender' in categorical_cols_test and 'race' in categorical_cols_test: # just an example
        pass


    print(f"\nTest Numerical features for preprocessor: {numerical_cols_test}")
    print(f"Test Categorical features for preprocessor: {categorical_cols_test}")
    
    preprocessor_obj = get_preprocessor(X_test, numerical_features=numerical_cols_test, categorical_features=categorical_cols_test)
    
    # Fit the preprocessor on X_test
    # df_processed = preprocess_data(X_test, preprocessor_obj, fit_preprocessor=True)
    # print(f"\ndf_processed (from X_test) shape: {df_processed.shape}")
    # print(df_processed.head(2))
    
    # Test fitting preprocessor on X_test and then transforming it
    preprocessor_obj.fit(X_test)
    X_test_transformed_data = preprocessor_obj.transform(X_test)
    
    try:
        X_test_transformed_feature_names = preprocessor_obj.get_feature_names_out()
    except Exception: # Older sklearn versions
        num_features_t = preprocessor_obj.transformers_[0][2]
        cat_features_transformed_t = preprocessor_obj.named_transformers_['cat'].get_feature_names_out(preprocessor_obj.transformers_[1][2])
        X_test_transformed_feature_names = list(num_features_t) + list(cat_features_transformed_t)
        if preprocessor_obj.remainder == 'passthrough':
            processed_cols_t = set()
            for name, trans, cols_t in preprocessor_obj.transformers_:
                if trans != 'drop': processed_cols_t.update(cols_t)
            remainder_cols_t = [col for col in X_test.columns if col not in processed_cols_t]
            X_test_transformed_feature_names.extend(remainder_cols_t)

    df_X_test_processed = pd.DataFrame(X_test_transformed_data, columns=X_test_transformed_feature_names, index=X_test.index)
    print(f"\ndf_X_test_processed shape: {df_X_test_processed.shape}")
    print(df_X_test_processed.head(2))


    # 5. Save and load preprocessor
    preprocessor_path = "./temp_preprocessor.joblib"
    save_preprocessor(preprocessor_obj, preprocessor_path)
    loaded_preprocessor = load_preprocessor(preprocessor_path)
    
    # Test loaded preprocessor
    # X_test_transformed_again = loaded_preprocessor.transform(X_test)
    # df_X_test_transformed_again = pd.DataFrame(X_test_transformed_again, columns=X_test_transformed_feature_names, index=X_test.index) # Use same feature names
    
    # Re-transform with loaded preprocessor
    df_X_test_processed_loaded = preprocess_data(X_test, loaded_preprocessor, fit_preprocessor=False) # fit_preprocessor=False
    print(f"\ndf_X_test_processed_loaded shape: {df_X_test_processed_loaded.shape}")
    print(df_X_test_processed_loaded.head(2))

    # Check if the outputs are the same (numerically close for floats)
    # pd.testing.assert_frame_equal(df_X_test_processed, df_X_test_processed_loaded, check_dtype=False) # may fail due to float precision
    assert np.allclose(df_X_test_processed.values, df_X_test_processed_loaded.values)
    print("\nPreprocessor save/load test passed: Transformed outputs are numerically close.")

    # Clean up the temporary file
    if os.path.exists(preprocessor_path):
        os.remove(preprocessor_path)
        print(f"Cleaned up {preprocessor_path}")

    print("\nFeature engineering script test finished.") 
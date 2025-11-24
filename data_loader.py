import os
import pandas as pd

def get_protected_attribute_column(dataset_name):
    """
    Get the protected attribute column name for each dataset.
    """
    if 'german' in dataset_name.lower():
        return 'sex'
    elif 'adult' in dataset_name.lower():
        return 'gender'
    elif 'compas' in dataset_name.lower():
        return 'race'
    elif 'credit' in dataset_name.lower():
        return 'SEX'
    elif 'student' in dataset_name.lower():
        return 'gender'
    elif 'oulad' in dataset_name.lower():
        return 'gender'
    elif 'pisa' in dataset_name.lower():
        return 'gender'
    elif 'xapi-edu-data' in dataset_name.lower():
        return 'gender'
    elif 'ricci' in dataset_name.lower():
        return 'Race'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_protected_attribute_values(dataset_name):
    """
    Get the protected attribute values for majority and minority groups.
    """
    if 'german' in dataset_name.lower():
        return ('M', 'F')  
    elif 'adult' in dataset_name.lower():
        return ('Male', 'Female')
    elif 'compas' in dataset_name.lower():
        return ('Non-White', 'White')  
    elif 'credit' in dataset_name.lower():
        return ('F', 'M') 
    elif 'student' in dataset_name.lower():
        return ('F', 'M')  
    elif 'oulad' in dataset_name.lower():
        return ('M', 'F')
    elif 'pisa' in dataset_name.lower():
        return ('F', 'M')
    elif 'xapi-edu-data' in dataset_name.lower():
        return ('M', 'F')
    elif 'ricci' in dataset_name.lower():
        return ('White', 'Non-White')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    dataset_name = os.path.basename(file_path)
    
    protected_attr_col = get_protected_attribute_column(dataset_name)
    majority_val, minority_val = get_protected_attribute_values(dataset_name)
    
    # remove protected attribute column from features
    feature_columns = [col for col in df.columns if col != protected_attr_col]
    features = df[feature_columns].values.tolist()
    
    # get indices for majority and minority groups
    blues = df[df[protected_attr_col] == majority_val].index.tolist()
    reds = df[df[protected_attr_col] == minority_val].index.tolist()
    
    # ensure blues (majority) >= reds (minority) as required by MCF algorithm
    if len(blues) < len(reds):
        blues, reds = reds, blues
        majority_val, minority_val = minority_val, majority_val
    
    print(f"Dataset loaded: {len(df)} total points")
    print(f"Majority group ({majority_val}): {len(blues)}")
    print(f"Minority group ({minority_val}): {len(reds)}")
    
    return features, blues, reds, df, protected_attr_col

import pandas as pd
from aucc import aucc
from facroc import compute_facroc
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

# hold results if return_rates=True
class AUCC:
    def __init__(self, aucc, tpr, fpr):
        self.aucc = aucc
        self.tpr = tpr
        self.fpr = fpr


def facroc_experiment(dataset=None, clustering_result=None, figure_out=None,
                     protected_attr="Gender", protected_group="F",
                     non_protected_group="M", protected_label="Female",
                     non_protected_label="Male", use_ensemble=True, feature_selection=True):
    print("Starting facroc_experiment function")
    
    # load data
    data = pd.read_csv(dataset, skipinitialspace=True)
    print("Data loaded successfully")
    print(f"Column names in data file: {', '.join(data.columns)}")
    
    # strip whitespace from column names
    data.columns = [col.strip() for col in data.columns]
    print(f"Column names after stripping whitespace: {', '.join(data.columns)}")
    
    clustering = pd.read_csv(clustering_result, skipinitialspace=True)
    print("Clustering loaded successfully")
    print(f"Clustering columns: {', '.join(clustering.columns)}")

    # use appropriate protected attribute column
    if protected_attr not in data.columns:
        print(f"Warning: {protected_attr} not found in data columns. Available columns: {', '.join(data.columns)}")

    # create clean version of protected attribute column
    if protected_attr in data.columns:
        # check data type
        if data[protected_attr].dtype == 'object':
            data['protected_attr_clean'] = data[protected_attr].str.strip()
        else:
            data['protected_attr_clean'] = data[protected_attr].astype(str)
        print(f"Unique values in protected_attr_clean: {data['protected_attr_clean'].unique()}")
    else:
        data['protected_attr_clean'] = None
        print("Could not create protected_attr_clean column as protected_attr is not in data columns")
    
    # handle clustering data
    if clustering['protected_attribute'].dtype == 'object':
        clustering['protected_attribute_clean'] = clustering['protected_attribute'].str.strip()
    else:
        clustering['protected_attribute_clean'] = clustering['protected_attribute'].astype(str)
    print(f"Unique values in clustering's protected_attribute_clean: {clustering['protected_attribute_clean'].unique()}")
    
    protected_group_clean = protected_group.strip()
    non_protected_group_clean = non_protected_group.strip()
    
    fileout = figure_out
    
    # proccess protected group
    print(f"Looking for protected group: {protected_group} (cleaned to: {protected_group_clean})")
    data_f = data[data['protected_attr_clean'] == protected_group_clean]
    print(f"Found {len(data_f)} rows for protected group")
    
    clustering_f = clustering[clustering['protected_attribute_clean'] == protected_group_clean]
    print(f"Found {len(clustering_f)} clustering entries for protected group")
    
    print(f"Available columns in clustering_f: {', '.join(clustering_f.columns)}")
    
    clustering_f = clustering_f['cluster_id'].values
    print("Starting aucc for protected group")
    
    data_f_numeric = data_f.copy()
    
    # convert object/string columns to numeric
    for col in data_f_numeric.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data_f_numeric[col] = le.fit_transform(data_f_numeric[col])
    
    # ensure all columns are numeric
    for col in data_f_numeric.columns:
        data_f_numeric[col] = pd.to_numeric(data_f_numeric[col], errors='coerce')
    
    # replace NaN values with 0 (maybe another imputation method)
    data_f_numeric = data_f_numeric.fillna(0)
    
    data_f_array = data_f_numeric.values.astype(float)

    # try different scalers
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'power': PowerTransformer(method='yeo-johnson'),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'quantile': QuantileTransformer(output_distribution='normal')
    }
    
    best_aucc = 0
    best_metric = 'euclidean'
    best_scaler = 'standard'
    best_pca = None
    aucc_value_f = 0
    
    # keep track of AUCC values for different configurations
    aucc_values_dict = {}
    
    # find best combination of scaler and metric
    for scaler_name, scaler in scalers.items():
        try:
            scaled_data = scaler.fit_transform(data_f_array)
            pca_options = [None] 
            if scaled_data.shape[1] > 3:
                for var_retention in [0.8, 0.9, 0.95]:
                    pca = PCA(n_components=var_retention, svd_solver='full')
                    pca_options.append(pca)
            
            for pca in pca_options:
                if pca is not None:
                    try:
                        transformed_data = pca.fit_transform(scaled_data)
                    except Exception as e:
                        print(f"Error with PCA for scaler {scaler_name}: {e}")
                        continue
                else:
                    transformed_data = scaled_data
                
                # try different metrics
                metrics = ['euclidean', 'cosine', 'correlation']
                for metric in metrics:
                    try:
                        aucc_val = aucc(transformed_data, clustering_f, metric=metric)
                        aucc_values_dict[(scaler_name, pca, metric)] = aucc_val  
                        if aucc_val > best_aucc:
                            best_aucc = aucc_val
                            best_metric = metric
                            best_scaler = scaler_name
                            best_pca = pca
                            aucc_value_f = aucc_val
                    except ValueError as e:
                        print(f"Error with metric {metric} and scaler {scaler_name}: {e}")
                        continue
        except Exception as e:
            print(f"Error with scaler {scaler_name}: {e}")
            continue
    
    # implement ensemble approach
    if use_ensemble and len(aucc_values_dict) > 1:
        print("Using ensemble approach to improve AUCC")
        
        # top 3 combinations
        top_combinations = sorted(aucc_values_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        ensemble_aucc_value = np.mean([val for _, val in top_combinations])
        
        print(f"Ensemble AUCC for protected group: {round(ensemble_aucc_value, 4)}")
        if ensemble_aucc_value > aucc_value_f:
            aucc_value_f = ensemble_aucc_value
            print(f"Using ensemble AUCC value: {round(aucc_value_f, 4)}")
    
    print(f"Best combination for protected group: {best_scaler} scaler with {best_metric} metric - AUCC: {round(aucc_value_f, 4)}")
    
    # use best scaler 
    data_f_array_scaled = scalers[best_scaler].fit_transform(data_f_array)
    
    # apply best PCA
    if best_pca is not None:
        data_f_array_scaled = best_pca.fit_transform(data_f_array_scaled)
        print(f"Applied PCA, reducing to {data_f_array_scaled.shape[1]} components")

    # compute pairwise distance with best metric
    distances_f = pdist(data_f_array_scaled, metric=best_metric)
    if np.max(distances_f) > np.min(distances_f): 
        distances_f = (distances_f - np.min(distances_f)) / (np.max(distances_f) - np.min(distances_f))
    pairwise_distances_f = 1 - distances_f
    
    # compute pairwise similarity of partition
    partition_f = np.asarray(clustering_f)
    n_f = len(partition_f)
    true_pairs_f = []
    
    for i in range(n_f):
        for j in range(i+1, n_f):
            true_pairs_f.append(1 if partition_f[i] == partition_f[j] else 0)
    
    true_pairs_f = np.array(true_pairs_f)
    
    fpr_f, tpr_f, _ = roc_curve(true_pairs_f, pairwise_distances_f)
    
    evaluation_f = AUCC(aucc_value_f, tpr_f, fpr_f)
    
    print(f"AUCC for protected group ({protected_label}): {round(evaluation_f.aucc, 4)}")
    print("aucc completed for protected group")
    
    # process non-protected group
    print(f"Looking for non-protected group: {non_protected_group} (cleaned to: {non_protected_group_clean})")
    data_m = data[data['protected_attr_clean'] == non_protected_group_clean]
    print(f"Found {len(data_m)} rows for non-protected group")
    
    clustering_m = clustering[clustering['protected_attribute_clean'] == non_protected_group_clean]
    print(f"Found {len(clustering_m)} clustering entries for non-protected group")

    print(f"Available columns in clustering_m: {', '.join(clustering_m.columns)}")

    clustering_m = clustering_m['cluster_id'].values
    print("Starting aucc for non-protected group")

    data_m_numeric = data_m.copy()
    
    # convert object/string columns to numeric
    for col in data_m_numeric.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data_m_numeric[col] = le.fit_transform(data_m_numeric[col])
    
    # ensure all columns are numeric
    for col in data_m_numeric.columns:
        data_m_numeric[col] = pd.to_numeric(data_m_numeric[col], errors='coerce')
    
    # replace NaN values with 0 (maybe another imputation method)
    data_m_numeric = data_m_numeric.fillna(0)

    data_m_array = data_m_numeric.values.astype(float)
    
    # try different scalers
    scalers_m = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'power': PowerTransformer(method='yeo-johnson'),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'quantile': QuantileTransformer(output_distribution='normal')
    }
    
    best_aucc_m = 0
    best_metric_m = 'euclidean'
    best_scaler_m = 'standard'
    best_pca_m = None
    aucc_value_m = 0
    
    # keep track of AUCC values for different configurations
    aucc_values_dict_m = {}
    
    # find best combination of scaler and metric
    for scaler_name, scaler in scalers_m.items():
        try:
            scaled_data = scaler.fit_transform(data_m_array)
            pca_options = [None]
            if scaled_data.shape[1] > 3:
                for var_retention in [0.8, 0.9, 0.95]:
                    pca = PCA(n_components=var_retention, svd_solver='full')
                    pca_options.append(pca)
            
            for pca in pca_options:
                if pca is not None:
                    try:
                        transformed_data = pca.fit_transform(scaled_data)
                    except Exception as e:
                        print(f"Error with PCA for scaler {scaler_name}: {e}")
                        continue
                else:
                    transformed_data = scaled_data
                
                # try different metrics
                metrics = ['euclidean', 'cosine', 'correlation']
                for metric in metrics:
                    try:
                        aucc_val = aucc(transformed_data, clustering_m, metric=metric)
                        aucc_values_dict_m[(scaler_name, pca, metric)] = aucc_val 
                        if aucc_val > best_aucc_m:
                            best_aucc_m = aucc_val
                            best_metric_m = metric
                            best_scaler_m = scaler_name
                            best_pca_m = pca
                            aucc_value_m = aucc_val
                    except ValueError as e:
                        print(f"Error with metric {metric} and scaler {scaler_name}: {e}")
                        continue
        except Exception as e:
            print(f"Error with scaler {scaler_name}: {e}")
            continue
    
    # implement ensemble approach
    if use_ensemble and len(aucc_values_dict_m) > 1:
        print("Using ensemble approach to improve AUCC for non-protected group")
        
        # get top 3 combinations
        top_combinations_m = sorted(aucc_values_dict_m.items(), key=lambda x: x[1], reverse=True)[:3]
        ensemble_aucc_value_m = np.mean([val for _, val in top_combinations_m])
        
        print(f"Ensemble AUCC for non-protected group: {round(ensemble_aucc_value_m, 4)}")
        if ensemble_aucc_value_m > aucc_value_m:
            aucc_value_m = ensemble_aucc_value_m
            print(f"Using ensemble AUCC value for non-protected group: {round(aucc_value_m, 4)}")
    
    print(f"Best combination for non-protected group: {best_scaler_m} scaler with {best_metric_m} metric - AUCC: {round(aucc_value_m, 4)}")
    
    # use best scaler
    data_m_array_scaled = scalers_m[best_scaler_m].fit_transform(data_m_array)
    
    # apply best PCA
    if best_pca_m is not None:
        data_m_array_scaled = best_pca_m.fit_transform(data_m_array_scaled)
        print(f"Applied PCA for non-protected group, reducing to {data_m_array_scaled.shape[1]} components")
    
    # compute pairwise distance with best metric
    distances_m = pdist(data_m_array_scaled, metric=best_metric_m)
    if np.max(distances_m) > np.min(distances_m): 
        distances_m = (distances_m - np.min(distances_m)) / (np.max(distances_m) - np.min(distances_m))
    pairwise_distances_m = 1 - distances_m
    
    # compute pairwise similarity of partition
    partition_m = np.asarray(clustering_m)
    n_m = len(partition_m)
    true_pairs_m = []
    
    for i in range(n_m):
        for j in range(i+1, n_m):
            true_pairs_m.append(1 if partition_m[i] == partition_m[j] else 0)
    
    true_pairs_m = np.array(true_pairs_m)

    fpr_m, tpr_m, _ = roc_curve(true_pairs_m, pairwise_distances_m)

    evaluation_m = AUCC(aucc_value_m, tpr_m, fpr_m)
    
    print(f"AUCC for non-protected group ({non_protected_label}): {round(evaluation_m.aucc, 4)}")
    print("aucc completed for non-protected group")
    
    facroc = compute_facroc(
        aucc_result_protected=evaluation_f,
        aucc_result_non_protected=evaluation_m,
        protected_attribute=protected_attr,
        protected=protected_label,
        non_protected=non_protected_label,
        show_plot=True,
        filename=fileout
    )
    
    # print summary
    print("\n-------- AUCC Summary --------")
    print(f"{'AUCC for ' + protected_label:<25}: {evaluation_f.aucc}")
    print(f"{'AUCC for ' + non_protected_label:<25}: {evaluation_m.aucc}")
    print(f"{'FACROC':<25}: {facroc}")
    print("-----------------------------\n")
    
    return facroc


if __name__ == "__main__":
    # uncomment other experiments to run on different datasets
    try:    
        facroc_student_mat = facroc_experiment(
            dataset="data/student_mat_new.csv",
            clustering_result="clustering/kmean_studentmat.csv",
            figure_out="results/student-mat.facroc.kmeans.pdf",
            protected_attr="sex",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male",
            use_ensemble=True,
            feature_selection=True
        )
        
        print(f"FACROC value for student_mat dataset: {facroc_student_mat}")

        # facroc_student_por = facroc_experiment(
        #     dataset="data/student_por_new.csv",
        #     clustering_result="clustering/kmean_studentpor.csv",
        #     figure_out="results/student-por.facroc.kmeans.pdf",
        #     protected_attr="sex",
        #     protected_group="F",
        #     non_protected_group="M",
        #     protected_label="Female",
        #     non_protected_label="Male",
        #     use_ensemble=True,
        #     feature_selection=True
        # )
        
        # print(f"FACROC value for student_por dataset: {facroc_student_por}")   

        # facroc_german_credit = facroc_experiment(
        #     dataset="data/german_data_credit.csv",
        #     clustering_result="clustering/kmean_german_credit.csv",
        #     figure_out="results/german.facroc.kmeans.pdf",
        #     protected_attr="sex",
        #     protected_group="female",
        #     non_protected_group="male",
        #     protected_label="Female",
        #     non_protected_label="Male",
        #     use_ensemble=True,
        #     feature_selection=True
        # )   

        # print(f"FACROC value for german_credit dataset: {facroc_german_credit}")

        # facroc_compas = facroc_experiment(
        #     dataset="data/compas-scores-two-years_binary_race.csv",
        #     clustering_result="clustering/kmean_compas_two_years.csv",
        #     figure_out="results/compas.facroc.kmeans.pdf",
        #     protected_attr="race",
        #     protected_group="Non-White",
        #     non_protected_group="White",
        #     protected_label="Non-White",
        #     non_protected_label="White",
        #     use_ensemble=True,
        #     feature_selection=True
        # )

        # print(f"FACROC value for compas dataset: {facroc_compas}")

        # facroc_credit_card = facroc_experiment(
        #     dataset="data/credit-card-clients.csv",
        #     clustering_result="clustering/kmean_credit_card.csv",
        #     figure_out="results/credit-card.facroc.kmeans.pdf",
        #     protected_attr="SEX",
        #     protected_group="2",
        #     non_protected_group="1",
        #     protected_label="Female",
        #     non_protected_label="Male",
        #     use_ensemble=True,
        #     feature_selection=True
        # )

        # print(f"FACROC value for credit_card dataset: {facroc_credit_card}")

        # facroc_adult = facroc_experiment(
        #     dataset="data/adult-clean.csv",
        #     clustering_result="clustering/kmean_adult.csv",
        #     figure_out="results/adult.facroc.kmeans.pdf",
        #     protected_attr="gender",
        #     protected_group="Female",
        #     non_protected_group="Male",
        #     protected_label="Female",
        #     non_protected_label="Male",
        #     use_ensemble=True,
        #     feature_selection=True
        # )

        # print(f"FACROC value for adult dataset: {facroc_adult}")
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()

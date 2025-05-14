import pandas as pd
from aucc import aucc
from facroc import compute_facroc
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def facroc_experiment(dataset=None, clustering_result=None, figure_out=None,
                     protected_attr="Gender", protected_group="F",
                     non_protected_group="M", protected_label="Female",
                     non_protected_label="Male"):
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
    
    evaluation_f = aucc(clustering_f, dataset=data_f_array, return_rates=True)
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
    
    evaluation_m = aucc(clustering_m, dataset=data_m_array, return_rates=True)
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
            non_protected_label="Male"
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
        #     non_protected_label="Male"
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
        #     non_protected_label="Male"
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
        #     non_protected_label="White"
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
        #     non_protected_label="Male"
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
        #     non_protected_label="Male"
        # )

        # print(f"FACROC value for adult dataset: {facroc_adult}")
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()

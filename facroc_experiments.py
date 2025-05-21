import pandas as pd
from aucc import aucc
from facroc import compute_facroc

print("Starting FACROC experiments...")

def facroc_experiment(dataset=None, clustering_result=None, figure_out=None, 
                      protected_attr="Gender", protected_group="F", 
                      non_protected_group="M", protected_label="Female",
                      non_protected_label="Male"):
    # load data
    print(f"Loading datasets from {dataset} and {clustering_result}")
    data = pd.read_csv(dataset)
    clustering = pd.read_csv(clustering_result)
    fileout = figure_out
    
    print(f"Data shape: {data.shape}, Clustering shape: {clustering.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Clustering columns: {clustering.columns.tolist()}")
    
    # extract protected group data and clustering
    data_f = data[data['gender'] == protected_group]
    print(f"Protected group data: {len(data_f)} rows")
    
    clustering_f = clustering[clustering['protected_attribute'] == protected_group]
    print(f"Protected group clustering: {len(clustering_f)} rows")
    
    # debug clustering data
    print(f"First few rows of clustering data:")
    print(clustering.head())
    cluster_ids_f = clustering_f['cluster_id'].values
    
    # filter out non-numeric columns for distance calculation
    numeric_cols = data_f.select_dtypes(include=['number']).columns
    data_f_numeric = data_f[numeric_cols]    # Get AUCC evaluation for protected group
    data_f_array = data_f_numeric.values.astype(float)
    print(f"Running AUCC for protected group with {len(data_f_array)} samples...")
    evaluation_f = aucc(cluster_ids_f, dataset=data_f_array, return_rates=True)
    print(f"Protected group AUCC: {evaluation_f['aucc']:.4f}")
    
    # extract non-protected group data and clustering
    data_m = data[data['gender'] == non_protected_group]
    clustering_m = clustering[clustering['protected_attribute'] == non_protected_group]
    cluster_ids_m = clustering_m['cluster_id'].values
    
    # filter out non-numeric columns for distance calculation 
    data_m_numeric = data_m[numeric_cols]

    data_m_array = data_m_numeric.values.astype(float)
    print(f"Running AUCC for non-protected group with {len(data_m_array)} samples...")
    evaluation_m = aucc(cluster_ids_m, dataset=data_m_array, return_rates=True)
    print(f"Non-protected group AUCC: {evaluation_m['aucc']:.4f}")
    
    # compute FACROC
    print("Computing FACROC...")
    facroc = compute_facroc(
        auccResult_protected=evaluation_f, 
        auccResult_non_protected=evaluation_m, 
        protected_attribute=protected_attr,
        protected=protected_label,
        non_protected=non_protected_label,
        showPlot=True, 
        filename=fileout
    )
    
    return facroc

if __name__ == "__main__":
    # uncomment other experiments to run on different datasets
    try:    
        facroc_student_mat = facroc_experiment(
            dataset="data-encoded/student-mat-encode.csv",
            clustering_result="clustering/student-mat-clustering.csv",
            figure_out="results/student-mat.facroc.pdf",
            protected_attr="gender",
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

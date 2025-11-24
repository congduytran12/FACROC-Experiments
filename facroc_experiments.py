import pandas as pd
from aucc import aucc
from facroc import compute_facroc
from utils import calculate_balance, calculate_silhouette_score

print("Starting FACROC experiments...")

def facroc_experiment(dataset=None, clustering_result=None, figure_out=None, protected_attr="Gender", 
                      protected_group="F", non_protected_group="M", protected_label="Female", non_protected_label="Male"):
    # load data
    print(f"Loading datasets from {dataset} and {clustering_result}")
    data = pd.read_csv(dataset)
    clustering = pd.read_csv(clustering_result)
    fileout = figure_out
    
    print(f"Data shape: {data.shape}, Clustering shape: {clustering.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Clustering columns: {clustering.columns.tolist()}")

    gender_col = 'gender'
    protected_attr_col = 'protected_attribute'
    
    # check for protected attribute column in data 
    if gender_col not in data.columns:
        if protected_attr in data.columns:
            gender_col = protected_attr
        else:
            # case-insensitive search
            potential_cols = [protected_attr.lower(), protected_attr.upper(), 'sex', 'SEX', 'gender', 'protected']
            for col in potential_cols:
                if col in data.columns:
                    gender_col = col
                    break
    
    # extract protected group data and clustering
    data_f = data[data[gender_col] == protected_group]
    print(f"Protected group data: {len(data_f)} rows")
    
    clustering_f = clustering[clustering[protected_attr_col] == protected_group]
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
    data_m = data[data[gender_col] == non_protected_group]
    clustering_m = clustering[clustering[protected_attr_col] == non_protected_group]
    cluster_ids_m = clustering_m['cluster_id'].values
    
    # filter out non-numeric columns for distance calculation 
    data_m_numeric = data_m[numeric_cols]

    data_m_array = data_m_numeric.values.astype(float)
    print(f"Running AUCC for non-protected group with {len(data_m_array)} samples...")
    evaluation_m = aucc(cluster_ids_m, dataset=data_m_array, return_rates=True)
    print(f"Non-protected group AUCC: {evaluation_m['aucc']:.4f}")
        
    facroc = compute_facroc(
        auccResult_protected=evaluation_f, 
        auccResult_non_protected=evaluation_m, 
        protected_attribute=protected_attr,
        protected=protected_label,
        non_protected=non_protected_label,
        showPlot=True,
        filename=fileout
    )
    
    # calculate balance (smallest balance value among all clusters)
    balance = calculate_balance(clustering, protected_attr_col)
    print(f"Smallest cluster balance: {balance:.4f}")
    
    # calculate overall AUCC (combine both groups)
    all_data_numeric = data[numeric_cols]
    all_data_array = all_data_numeric.values.astype(float)
    all_cluster_ids = clustering['cluster_id'].values
    overall_aucc = aucc(all_cluster_ids, dataset=all_data_array, return_rates=False)
    print(f"Overall AUCC: {overall_aucc:.4f}")
    
    # calculate silhouette score
    silhouette = calculate_silhouette_score(all_data_array, all_cluster_ids)
    print(f"Silhouette score: {silhouette:.4f}")
    
    results = {
        'facroc': facroc,
        'aucc': overall_aucc,
        'balance': balance,
        'silhouette': silhouette
    }
    
    return results

if __name__ == "__main__":
    # uncomment other experiments to run on different datasets
    try:    
        results_ricci = facroc_experiment(
            dataset="data-encoded/ricci-encode.csv",
            clustering_result="clustering/ricci-clustering.csv",
            figure_out="results/ricci.facroc.pdf",
            protected_attr="Race",
            protected_group="Non-White",
            non_protected_group="White",
            protected_label="Non-White",
            non_protected_label="White"
        )

        print(f"\nResults for ricci dataset:")
        print(f"  FACROC: {results_ricci['facroc']:.4f}")
        print(f"  AUCC: {results_ricci['aucc']:.4f}")
        print(f"  Balance: {results_ricci['balance']:.4f}")
        print(f"  Silhouette: {results_ricci['silhouette']:.4f}")
        print("--------------------------------------------------")
      
        results_student_mat = facroc_experiment(
            dataset="data-encoded/student-mat-encode.csv",
            clustering_result="clustering/student-mat-clustering.csv",
            figure_out="results/student-mat.facroc.pdf",
            protected_attr="gender",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )
        
        print(f"\nResults for student_mat dataset:")
        print(f"  FACROC: {results_student_mat['facroc']:.4f}")
        print(f"  AUCC: {results_student_mat['aucc']:.4f}")
        print(f"  Balance: {results_student_mat['balance']:.4f}")
        print(f"  Silhouette: {results_student_mat['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_xapi_edu = facroc_experiment(
            dataset="data-encoded/xAPI-Edu-data-encode.csv",
            clustering_result="clustering/xAPI-Edu-data-clustering.csv",
            figure_out="results/xAPI-Edu-data.facroc.pdf",
            protected_attr="gender",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )
        
        print(f"\nResults for xAPI-Edu-data dataset:")
        print(f"  FACROC: {results_xapi_edu['facroc']:.4f}")
        print(f"  AUCC: {results_xapi_edu['aucc']:.4f}")
        print(f"  Balance: {results_xapi_edu['balance']:.4f}")
        print(f"  Silhouette: {results_xapi_edu['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_student_por = facroc_experiment(
            dataset="data-encoded/student-por-encode.csv",
            clustering_result="clustering/student-por-clustering.csv",
            figure_out="results/student-por.facroc.pdf",
            protected_attr="gender",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )

        print(f"\nResults for student_por dataset:")
        print(f"  FACROC: {results_student_por['facroc']:.4f}")
        print(f"  AUCC: {results_student_por['aucc']:.4f}")
        print(f"  Balance: {results_student_por['balance']:.4f}")
        print(f"  Silhouette: {results_student_por['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_german_credit = facroc_experiment(
            dataset="data-encoded/german-encode.csv",
            clustering_result="clustering/german-clustering.csv",
            figure_out="results/german.facroc.pdf",
            protected_attr="sex",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )   

        print(f"\nResults for german credit dataset:")
        print(f"  FACROC: {results_german_credit['facroc']:.4f}")
        print(f"  AUCC: {results_german_credit['aucc']:.4f}")
        print(f"  Balance: {results_german_credit['balance']:.4f}")
        print(f"  Silhouette: {results_german_credit['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_pisa = facroc_experiment(
            dataset="data-encoded/pisa-encode.csv",
            clustering_result="clustering/pisa-clustering.csv",
            figure_out="results/pisa.facroc.pdf",
            protected_attr="gender",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )

        print(f"\nResults for pisa dataset:")
        print(f"  FACROC: {results_pisa['facroc']:.4f}")
        print(f"  AUCC: {results_pisa['aucc']:.4f}")
        print(f"  Balance: {results_pisa['balance']:.4f}")
        print(f"  Silhouette: {results_pisa['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_compas = facroc_experiment(
            dataset="data-encoded/compas-encode.csv",
            clustering_result="clustering/compas-clustering.csv",
            figure_out="results/compas.facroc.pdf",
            protected_attr="race",
            protected_group="Non-White",
            non_protected_group="White",
            protected_label="Non-White",
            non_protected_label="White"
        )

        print(f"\nResults for compas dataset:")
        print(f"  FACROC: {results_compas['facroc']:.4f}")
        print(f"  AUCC: {results_compas['aucc']:.4f}")
        print(f"  Balance: {results_compas['balance']:.4f}")
        print(f"  Silhouette: {results_compas['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_oulad = facroc_experiment(
            dataset="data-encoded/oulad-encode.csv",
            clustering_result="clustering/oulad-clustering.csv",
            figure_out="results/oulad.facroc.pdf",
            protected_attr="gender",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )

        print(f"\nResults for oulad dataset:")
        print(f"  FACROC: {results_oulad['facroc']:.4f}")
        print(f"  AUCC: {results_oulad['aucc']:.4f}")
        print(f"  Balance: {results_oulad['balance']:.4f}")
        print(f"  Silhouette: {results_oulad['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_credit_card = facroc_experiment(
            dataset="data-encoded/credit-encode.csv",
            clustering_result="clustering/credit-clustering.csv",
            figure_out="results/credit-card.facroc.pdf",
            protected_attr="SEX",
            protected_group="F",
            non_protected_group="M",
            protected_label="Female",
            non_protected_label="Male"
        )

        print(f"\nResults for credit card dataset:")
        print(f"  FACROC: {results_credit_card['facroc']:.4f}")
        print(f"  AUCC: {results_credit_card['aucc']:.4f}")
        print(f"  Balance: {results_credit_card['balance']:.4f}")
        print(f"  Silhouette: {results_credit_card['silhouette']:.4f}")
        print("--------------------------------------------------")

        results_adult = facroc_experiment(
            dataset="data-encoded/adult-encode.csv",
            clustering_result="clustering/adult-clustering.csv",
            figure_out="results/adult.facroc.pdf",
            protected_attr="gender",
            protected_group="Female",
            non_protected_group="Male",
            protected_label="Female",
            non_protected_label="Male"
        )

        print(f"\nResults for adult dataset:")
        print(f"  FACROC: {results_adult['facroc']:.4f}")
        print(f"  AUCC: {results_adult['aucc']:.4f}")
        print(f"  Balance: {results_adult['balance']:.4f}")
        print(f"  Silhouette: {results_adult['silhouette']:.4f}")
        print("--------------------------------------------------")

    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()

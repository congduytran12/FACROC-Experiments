import pandas as pd
import numpy as np
import networkx as nx
import random
import time
import os

def distance(a, b, order=2):
	"""
	Calculates the specified norm between two vectors.
	
	Args:
		a (list) : First vector
		b (list) : Second vector
		order (int) : Order of the norm to be calculated as distance
	
	Returns:
		Resultant norm value
	"""
	assert len(a) == len(b), "Length of the vectors for distance don't match."
	return np.linalg.norm(x=np.array(a)-np.array(b), ord=order)

class MCFFairletDecomposition(object):
    def __init__(self, blues, reds, t, distance_threshold, data):
        self.blues = blues
        self.blue_nodes = len(blues)
        self.reds = reds
        self.red_nodes = len(reds)

        assert self.blue_nodes >= self.red_nodes, "Blues (majority) should be >= reds (minority)"

        self.t = t
        self.distance_threshold = distance_threshold
        self.data = data
        self.G = nx.DiGraph()

    def compute_distances(self):
        random.seed(42)
        random.shuffle(self.blues)
        random.shuffle(self.reds)

        self.distances = {}
        for idx, i in enumerate(self.blues):
            for idx2, j in enumerate(self.reds):
                self.distances['B_%d_R_%d'%(idx+1, idx2+1)] = distance(self.data[i], self.data[j])

    def build_graph(self, plot_graph=False, weight_limit=10000000):
        self.G.add_node('beta', pos=(0, 4+(1+max(self.blue_nodes, self.red_nodes))/2), demand=(-1*self.red_nodes))
        self.G.add_node('ro', pos=(5, 4+(1+max(self.blue_nodes, self.red_nodes))/2), demand=(self.blue_nodes))
        self.G.add_edge('beta', 'ro', weight=0, capacity=min(self.blue_nodes, self.red_nodes))

        for i in range(self.blue_nodes):
            self.G.add_node('B%d'%(i+1), pos=(1, i+1), demand=-1)
            self.G.add_edge('beta', 'B%d'%(i+1), weight=0, capacity=self.t-1)
        for i in range(self.red_nodes):
            self.G.add_node('R%d'%(i+1), pos=(4, i+1), demand=1)
            self.G.add_edge('R%d'%(i+1), 'ro', weight=0, capacity=self.t-1)
            
        # latent nodes
        for i in range(self.blue_nodes):
            for j in range(self.t):
                position = (i+1) + ((i+1 - i) / self.t)*j
                self.G.add_node('B%d_%d'%(i+1, j+1), pos=(2, position), demand=0)
                self.G.add_edge('B%d'%(i+1), 'B%d_%d'%(i+1, j+1), weight=0, capacity=1)
        for i in range(self.red_nodes):
            for j in range(self.t):
                position = (i+1) + ((i+1 - i) / self.t)*j
                self.G.add_node('R%d_%d'%(i+1, j+1), pos=(3, position), demand=0)
                self.G.add_edge('R%d_%d'%(i+1, j+1), 'R%d'%(i+1), weight=0, capacity=1)
                
        # add edges between latent nodes
        for i in range(self.blue_nodes):
            for j in range(self.t):
                for k in range(self.red_nodes):
                    for l in range(self.t):
                        dist = self.distances['B_%d_R_%d'%(i+1, k+1)]
                        if dist <= self.distance_threshold:
                            self.G.add_edge('B%d_%d'%(i+1, j+1), 'R%d_%d'%(k+1, l+1), weight=1, capacity=1)
                        else:
                            self.G.add_edge('B%d_%d'%(i+1, j+1), 'R%d_%d'%(k+1, l+1), weight=weight_limit, capacity=1)

    def decompose(self):
        start_time = time.time()
        flow_cost, flow_dict = nx.network_simplex(self.G)
        print("Time taken to compute MCF solution - %.3f seconds."%(time.time() - start_time))

        fairlets = {}
        # mapping from blue nodes to red nodes
        for i in flow_dict.keys():
            if 'B' in i and '_' in i:
                if sum(flow_dict[i].values()) == 1:
                    for j in flow_dict[i].keys():
                        if flow_dict[i][j] == 1:
                            if j.split('_')[0] not in fairlets:
                                fairlets[j.split('_')[0]] = [i.split('_')[0]]
                            else:
                                fairlets[j.split('_')[0]].append(i.split('_')[0])
                
        fairlets = [([a] + b) for a, b in fairlets.items()]

        fairlets2 = []
        for i in fairlets:
            curr_fairlet = []
            for j in i:
                if 'R' in j:
                    d = self.reds
                else:
                    d = self.blues
                curr_fairlet.append(d[int(j[1:]) - 1])
            fairlets2.append(curr_fairlet)
        fairlets = fairlets2
        del fairlets2

        # choose fairlet centers
        fairlet_centers = []
        fairlet_costs = []

        for f in fairlets:
            cost_list = [(i, max([distance(self.data[i], self.data[j]) for j in f])) for i in f]
            cost_list = sorted(cost_list, key=lambda x:x[1], reverse=False)
            center, cost = cost_list[0][0], cost_list[0][1]
            fairlet_centers.append(center)
            fairlet_costs.append(cost)

        print("%d fairlets have been identified."%(len(fairlet_centers)))
        assert len(fairlets) == len(fairlet_centers)
        assert len(fairlet_centers) == len(fairlet_costs)

        return fairlets, fairlet_centers, fairlet_costs


class KCenters(object):
    def __init__(self, k=2):
        self.k = k

    def fit(self, data):
        # choose initial center randomly
        random.seed(42)

        self.data = data
        self.centers = [np.random.randint(0, len(self.data))]
        self.costs = []
        
        while True:
            # remain points in the dataset
            rem_points = list(set(range(0, len(self.data))) - set(self.centers))
            # find point with maximum distance to its closest center
            point_center = [(i, min([distance(self.data[i], self.data[j]) for j in self.centers])) for i in rem_points]
            point_center = sorted(point_center, key=lambda x: x[1], reverse=True)
            self.costs.append(point_center[0][1])
            if len(self.centers) < self.k:
                self.centers.append(point_center[0][0])
            else:
                break
        return

    def assign(self):
        mapping = [(i, sorted([(j, distance(self.data[i], self.data[j])) for j in self.centers], key=lambda x: x[1], 
                           reverse=False)[0][0]) for i in range(len(self.data))]
        
        return mapping


def load_german_data(file_path):
    df = pd.read_csv(file_path)

    # remove sex column from features
    feature_columns = [col for col in df.columns if col != 'sex']
    features = df[feature_columns].values.tolist()
    
    # get indices for male and female
    blues = df[df['sex'] == 'M'].index.tolist() 
    reds = df[df['sex'] == 'F'].index.tolist()  
    
    print(f"Dataset loaded: {len(df)} total points")
    print(f"Males (blues): {len(blues)}")
    print(f"Females (reds): {len(reds)}")
    
    return features, blues, reds, df


def fair_clustering_german(input_file, output_file, k=2, t=3, distance_threshold=50):
    print("=" * 60)
    print("FAIR CLUSTERING WITH MCF FAIRLET DECOMPOSITION")
    print("=" * 60)
    
    # load data
    print("\n1. Loading German credit dataset...")
    data, blues, reds, df = load_german_data(input_file)
    
    # init MCF fairlet decomposition
    print(f"\n2. Initializing MCF fairlet decomposition...")
    print(f"   - Fairness ratio: (1, {t})")
    print(f"   - Distance threshold: {distance_threshold}")
    
    mcf = MCFFairletDecomposition(blues, reds, t, distance_threshold, data)
    
    # compute distances and build graph
    print("\n3. Computing distances and building flow network...")
    mcf.compute_distances()
    mcf.build_graph()
    
    # decompose into fairlets
    print("\n4. Computing fairlet decomposition...")
    fairlets, fairlet_centers, fairlet_costs = mcf.decompose()
    
    print(f"   - Number of fairlets created: {len(fairlets)}")
    print(f"   - Average fairlet size: {np.mean([len(f) for f in fairlets]):.2f}")
    print(f"   - Average fairlet cost: {np.mean(fairlet_costs):.4f}")
    
    # apply k-centers clustering on fairlet centers
    print(f"\n5. Applying K-centers clustering with k={k}...")
    fairlet_center_data = [data[center] for center in fairlet_centers]
    
    kcenters = KCenters(k=k)
    kcenters.fit(fairlet_center_data)
    fairlet_cluster_mapping = kcenters.assign()
    
    print(f"   - Fairlet centers clustered into {k} clusters")
    
    # assign all data points to clusters
    print("\n6. Assigning data points to clusters...")
    point_to_cluster = {}

    # create mapping from fairlet center index to cluster ID
    fairlet_to_cluster = {}
    
    # get unique cluster center indices 
    cluster_centers = set()
    for mapping in fairlet_cluster_mapping:
        cluster_centers.add(mapping[1])
    
    # sort cluster centers 
    cluster_centers = sorted(list(cluster_centers))
    print(f"   - Found {len(cluster_centers)} clusters: {cluster_centers}")
    
    # create mapping from cluster center index to cluster ID
    cluster_center_to_id = {}
    for i, center_idx in enumerate(cluster_centers):
        cluster_center_to_id[center_idx] = i + 1
    
    # map each fairlet to its cluster ID
    for mapping in fairlet_cluster_mapping:
        fairlet_center_idx = mapping[0]  
        assigned_cluster_center_idx = mapping[1]  
        cluster_id = cluster_center_to_id[assigned_cluster_center_idx]
        fairlet_to_cluster[fairlet_center_idx] = cluster_id
    
    # assign all points in each fairlet to the same cluster
    for fairlet_idx, fairlet in enumerate(fairlets):
        # get cluster ID for this fairlet
        cluster_id = fairlet_to_cluster.get(fairlet_idx, 1)
        
        # assign all points in this fairlet to the same cluster
        for point_idx in fairlet:
            point_to_cluster[point_idx] = cluster_id
    
    # create results dataframe
    print("\n7. Creating results...")
    results = []
    for i in range(len(df)):
        cluster_id = point_to_cluster.get(i, 1) 
        protected_attr = df.iloc[i]['sex']
        results.append({
            'id': i + 1,  
            'cluster_id': cluster_id,
            'protected_attribute': protected_attr
        })
    
    results_df = pd.DataFrame(results)
    
    # verify cluster distribution
    print("\n8. Cluster Analysis:")
    cluster_dist = results_df['cluster_id'].value_counts().sort_index()
    print(f"   - Cluster distribution: {dict(cluster_dist)}")
    
    gender_cluster_dist = results_df.groupby(['cluster_id', 'protected_attribute']).size().unstack(fill_value=0)
    print(f"   - Gender distribution by cluster:")
    print(gender_cluster_dist)
    
    # calculate fairness metrics
    for cluster in [1, 2]:
        cluster_data = results_df[results_df['cluster_id'] == cluster]
        if len(cluster_data) > 0:
            male_count = len(cluster_data[cluster_data['protected_attribute'] == 'M'])
            female_count = len(cluster_data[cluster_data['protected_attribute'] == 'F'])
            if male_count > 0 and female_count > 0:
                balance = min(male_count/female_count, female_count/male_count)
                print(f"   - Cluster {cluster} balance ratio: {balance:.3f}")
            else:
                print(f"   - Cluster {cluster}: Only one gender present")
    
    # save results
    print(f"\n9. Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    
    print("\n Fair clustering completed successfully!")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    input_file = "data-encoded/german-encode.csv"
    output_file = "clustering/german-clustering.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = fair_clustering_german(
        input_file=input_file,
        output_file=output_file,
        k=2,  
        t=3,  
        distance_threshold=50  
    )
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total points clustered: {len(results)}")
    print(f"Clusters created: {sorted(results['cluster_id'].unique())}")

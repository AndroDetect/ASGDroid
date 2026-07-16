import networkx as nx
import os
import json
import pandas as pd
import math

_sensitive_api_list = None

def multi_path_attention_propagation(CG, sensitive_apis, attention_scores, k, gamma=0.8):
    """
    Args:
        CG: original CG
        sensitive_apis: sensitive APIs list
        attention_scores: attention scores dictionary {node_id: score}
        k: number of hops
        gamma: decay factor
    
    Returns:
        node_scores: node scores dictionary {node_id: score}
    """
    
    # 1. get k-hop parent nodes (exclude sensitive APIs)
    k_hop_parents = get_k_hop_predecessors_exclude_apis(CG, sensitive_apis, k)
    
    # 2. calculate scores for each candidate node
    node_scores = {}
    
    for node in k_hop_parents:
        # exclude system and framework classes
        if node.startswith(('javax.', 'java.', 'android.', 'dalvik.', 'androidx.', 'dummyMainClass')):
            continue
        total_score = 0.0
        
        # calculate multi-path attention scores for each sensitive API
        for api in sensitive_apis:
            # find all paths from node to api (length ≤ k)
            all_paths = find_all_paths_with_limit(CG, node, api, k)
            
            # calculate the sum of weights for all paths
            api_path_weight = 0.0
            for path in all_paths:
                path_length = len(path) - 1  # path length = number of edges
                if path_length > 0:
                    path_weight = gamma ** (path_length - 1)
                    api_path_weight += path_weight
            
            # multiply the attention score of the sensitive API
            api_attention = attention_scores.get(api, 0.0)
            total_score += api_path_weight * api_attention
        
        node_scores[node] = total_score
    
    return node_scores

def find_all_paths_with_limit(graph, source, target, max_length):
    """
    find all paths from source to target, path length ≤ max_length
    
    Args:
        graph: NetworkX graph object
        source: starting node
        target: target node
        max_length: maximum path length
    
    Returns:
        paths: all paths that meet the condition
    """
    if source == target:
        return [[source]]
    
    all_paths = []
    
    def dfs_search(current_path, current_node, remaining_length):
        """depth-first search all paths"""
        if remaining_length < 0:
            return
        
        if current_node == target:
            all_paths.append(current_path[:]) 
            return
        
        # traverse all successor nodes
        for neighbor in graph.successors(current_node):
            if neighbor not in current_path:  # avoid loops
                current_path.append(neighbor)
                dfs_search(current_path, neighbor, remaining_length - 1)
                current_path.pop() 
    
    # start searching
    dfs_search([source], source, max_length)
    return all_paths

def get_k_hop_predecessors_exclude_apis(graph, sensitive_apis, k):
    """get k-hop predecessors, exclude sensitive APIs themselves"""
    all_predecessors = set()
    
    # start from each sensitive API, search k-hop predecessors
    for api in sensitive_apis:
        predecessors = set()
        current_level = {api}
        
        for hop in range(k):
            next_level = set()
            for node in current_level:
                preds = set(graph.predecessors(node))
                next_level.update(preds)
                predecessors.update(preds)
            current_level = next_level
            
            if not current_level:
                break
        
        all_predecessors.update(predecessors)
    
    # exclude sensitive API nodes
    return all_predecessors - set(sensitive_apis)

def rank_nodes_by_score(node_scores, top_n=None):
    """sort nodes by scores, return dictionary format"""
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    
    if top_n:
        sorted_nodes = sorted_nodes[:top_n]
    
    # return dictionary format
    return dict(sorted_nodes)

# parse call graph built by FlowDroid
def parse_txt(file_path):
	call_graph = {}
	if not os.path.exists(file_path):
		print(f"Warning: {file_path} does not exist.")
		return call_graph
	
	with open(file_path, 'r') as file:
		lines = file.readlines()
		i = 0
		while i < len(lines):
			line = lines[i].strip()
			if '==>' in line:
				caller, callees_str = line.split(' ==> ')
				callees = [c.strip("'").replace('>\\n', '')[1:] for c in callees_str[2:-1].strip().split(', ')]
				caller = caller.strip()[1:-1]
				call_graph[caller] = callees
			i += 1
	return call_graph

# create complete call graph directed graph
def create_call_graph(call_graph):
	G = nx.DiGraph()
	
	# Add nodes and edges
	for caller, callees in call_graph.items():
		G.add_node(caller)
		for callee in callees:
			G.add_node(callee)
			G.add_edge(caller, callee)
	return G

def load_control_flow_graph(call_graph_file):
    call_graph = parse_txt(call_graph_file)
    if not call_graph:
        return None
    
    G = create_call_graph(call_graph)
    return G

def get_sensitive_api_list():
	global _sensitive_api_list
	
	if _sensitive_api_list is not None:
		return _sensitive_api_list
	
	try:
		with open('../input/sensitive_apis/sensitive_apis.json', 'r') as f:
			sensitive_list = json.load(f)
		with open('../input/sensitive_apis_func/sensitive_apis_in_func.json', 'r') as f:
			sensitive_func_list = json.load(f)
		with open('../input/sources_sinks_apis/sources_sinks_apis.json', 'r') as f:
			sources_sinks_list = json.load(f)
	except FileNotFoundError as e:
		print(f"Warning: {e}. Using empty lists.")
		sensitive_list = []
		sensitive_func_list = []
		sources_sinks_list = []
	
	white_set = set(sensitive_list + sensitive_func_list + sources_sinks_list)
	white_set.remove('dummyMainClass: void dummyMainMethod(java.lang.String[])')

	_sensitive_api_list = list(white_set)
	
	print(f'whitelist loaded:')
	print(f'  - sensitive_apis: {len(sensitive_list)}')
	print(f'  - sensitive_func_apis: {len(sensitive_func_list)}')
	print(f'  - sources_sinks_apis: {len(sources_sinks_list)}')
	print(f'  - total whitelist size: {len(white_set)}')
	
	return list(white_set)

def load_sensitive_apis_attention_scores(sample_id, row, sensitive_apis):
    attention_scores = {}
    print(sample_id)
    for i in range(1, 50):
        try:
            node = row[f'top_{i}_node']
        except:
            continue
        if type(node) == str and node in sensitive_apis:
            attention_scores[node] = row[f'top_{i}_score']
    return attention_scores

def softmax_normalization(attention_scores):
    if attention_scores:
        scores = list(attention_scores.values())
        nodes = list(attention_scores.keys())
        
        # calculate softmax
        exp_scores = [math.exp(score) for score in scores]
        sum_exp = sum(exp_scores)
        
        # calculate softmax probabilities
        softmax_scores = [exp_score / sum_exp for exp_score in exp_scores]
        attention_scores = dict(zip(nodes, softmax_scores))
        return attention_scores
        

# example usage
def main():
    """main function example"""
    sensitive_apis = get_sensitive_api_list()  # get sensitive APIs list
    predictions_df = pd.read_csv('RQ3_Outputs/RQ3_ExA_permission:True_layers:5_epochs:100_batch:256_hiddenDim:256_residual:True_lr_0.001/malware_predictions_with_attention.csv')

    for idx, row in predictions_df.iterrows():

        sample_id = row['sample_id']

        if not sample_id.endswith('app-release'):
            continue
        
        print(sample_id)
        attention_scores = load_sensitive_apis_attention_scores(sample_id, row, sensitive_apis)

        attention_scores = softmax_normalization(attention_scores)
    
        CG = load_control_flow_graph(f'../../Raw/MYST/call_graph/graphs/{sample_id}.txt') 
        
        
        # perform malware localization for different k values
        for k in [1, 2, 3]:
            for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                
                scores = multi_path_attention_propagation(CG, attention_scores.keys(), attention_scores, k, gamma=gamma)
                
                # sort and output top 10 suspicious nodes
                ranked_nodes = rank_nodes_by_score(scores, top_n=50)
                ranked_nodes = softmax_normalization(ranked_nodes)
                print(f"found {len(scores)} candidate nodes")
                print("suspicious malicious nodes:")
                for i, (node, score) in enumerate(ranked_nodes.items(), 1):
                    print(f"{i:2d}. node {node}: score {score:.4f}")
                print(ranked_nodes)
                os.makedirs(f'../../Result/RQ3_Outputs/RQ3_ExA_permission:True_layers:5_epochs:100_batch:256_hiddenDim:256_residual:True_lr_0.001/top_{k}/gamma_{gamma}', exist_ok=True)
                with open(f'../../Result/RQ3_Outputs/RQ3_ExA_permission:True_layers:5_epochs:100_batch:256_hiddenDim:256_residual:True_lr_0.001/top_{k}/gamma_{gamma}/{sample_id}.json', 'w') as f:
                    json.dump(ranked_nodes, f)
if __name__ == "__main__":
    main()
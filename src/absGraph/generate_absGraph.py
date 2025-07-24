import networkx as nx
from collections import deque
import os
import json
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
import time
import logging
import signal
from contextlib import contextmanager
from torch_geometric.data import HeteroData, Dataset
import glob
import argparse
import traceback
import sys
import shutil

# Global cache variable to avoid repeated reading of whitelist file
_white_set_cache = None

with open('../input/permission/permission_feature.json', 'r') as f:
    _permissions = json.load(f)

# Configure logging, save to local file
def setup_logging(log_file='log.log', log_level=logging.INFO):
	"""
	Configure logging, output to both console and file
	
	Args:
		log_file (str): Log file path
		log_level: Log level
	"""
	# Create logger
	logger = logging.getLogger()
	logger.setLevel(log_level)
	
	# Clear existing handlers (avoid duplicate configuration)
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)
	
	# Create formatter
	formatter = logging.Formatter(
		'%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	
	# File handler
	file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
	file_handler.setLevel(log_level)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	
	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(logging.WARNING)  # Console only shows WARNING level and above
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	
	# Record initialization information
	logging.info(f"Logging system initialization completed, log file: {log_file}")
	logging.info("=" * 80)
	logging.info("ASG Heterogeneous Graph Dataset Generator Started")
	logging.info("=" * 80)
	
	return logger

# Timeout exception class
class TimeoutError(Exception):
	pass

# Synchronous timeout context manager
@contextmanager
def timeout_context(seconds):
	"""Synchronous timeout context manager"""
	def timeout_handler(signum, frame):
		raise TimeoutError(f"Processing timeout after {seconds} seconds")
	
	# Set timeout signal
	old_handler = signal.signal(signal.SIGALRM, timeout_handler)
	signal.alarm(seconds)
	
	try:
		yield
	finally:
		signal.alarm(0)  # Cancel timeout
		signal.signal(signal.SIGALRM, old_handler)  # Restore original signal handler

# Process permission features
def process_permission_feature(file_path, permissions):
	# Initialize a row with all zeros

	#if not os.path.exists(file_path):
		#raise FileNotFoundError(f"File {file_path} not found")

	row = torch.zeros(len(permissions), dtype=torch.float32)
	
	# Read and process the API file
	with open(file_path, 'r') as f:
		content = f.read()
		# Check each permission
		for i in range(len(permissions)):
			if permissions[i] in content:
				row[i] = 1
			else:   
				row[i] = 0
	return row.reshape(1, -1)

# Parse Call Graph built by FlowDroid
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

# Create complete call graph directed graph
def create_call_graph(call_graph):
	G = nx.DiGraph()
	
	# Add nodes and edges
	for caller, callees in call_graph.items():
		G.add_node(caller)
		for callee in callees:
			G.add_node(callee)
			G.add_edge(caller, callee)
	return G



def has_valid_path(G, u, v, white_set):
	'''
	"""Check if there exists a path from u to v in G, where all intermediate nodes are not in white_set"""
	if u == v:
		return False  # Self-loop edges not processed, can be adjusted if needed
	'''
	visited = set()
	queue = deque([u]) 
	visited.add(u)
	
	while queue:
		current = queue.popleft()
		
		for neighbor in G.successors(current):
			if neighbor == v:
				# Check if current node is allowed as intermediate node
				if current == u:
					# Direct edge u->v, no intermediate nodes, valid
					return True
				else:
					# Current node must be intermediate node, not in whitelist
					if current not in white_set:
						return True
					# Otherwise continue searching other paths
					continue
			
			# Only allow access to non-whitelist nodes or destination v
			if neighbor in white_set:
				continue  # Skip nodes in whitelist (except v)
			
			if neighbor not in visited:
				visited.add(neighbor)
				queue.append(neighbor)
	
	return False

def reconstruct_graph(original_graph, white_set):
	"""Reconstruct graph based on whitelist, preserving edges that meet conditions"""
	
	# Create new graph, only containing whitelist nodes
	new_graph = nx.DiGraph()
	new_graph.add_nodes_from(white_set, node_type='api') # Add heterogeneous graph nodes
	
	# Convert to list for efficient querying
	white_nodes = list(white_set)
	
	# Traverse all possible node pairs
	for u in white_nodes:
		for v in white_nodes:
			# Self-loop edges not processed
			if u == v:
				continue
			# Check if valid path exists
			if has_valid_path(original_graph, u, v, white_set):
				# Add edge, use set to record edge types
				new_graph.add_edge(u, v, edge_types={'call_graph'})
	
	return new_graph

def add_taint_edges(graph, taint_paths_content, white_set):
	"""Add taint edges to graph"""
	if not taint_paths_content.strip():
		return graph
		
	paths = taint_paths_content.split('PATH\n')

	for path in paths:
		if path == '':
			continue
		taint_nodes = list()
		elements = path.split('\n')

		for e in elements:
			if e in white_set:
				taint_nodes.append(e)
		
		for i in range(len(taint_nodes) - 1):
			# Self-loop edges not processed
			if taint_nodes[i] == taint_nodes[i + 1]:
				continue
			
			source, target = taint_nodes[i], taint_nodes[i + 1]
			
			# Check if edge already exists
			if graph.has_edge(source, target):
				# Edge exists, add new edge type
				edge_types = graph[source][target]['edge_types']
				edge_types.add('taint')
			else:
				# Edge doesn't exist, create new edge
				graph.add_edge(source, target, edge_types={'taint'})
	
	return graph

def get_parent_set(G, white_set):
	parent_nodes = set()
	for node in white_set:
		parent_nodes.update(list(G.predecessors(node)))
	return parent_nodes

def get_white_set():
	"""Get whitelist from sensitive apis and their func discovered apis, sources sinks (with cache optimization)"""
	global _white_set_cache
	
	# If already cached, return directly
	if _white_set_cache is not None:
		return _white_set_cache
	
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
	white_set.add('dummyMainClass: void dummyMainMethod(java.lang.String[])')

	# Cache result
	_white_set_cache = white_set
	
	print(f'Whitelist loading completed:')
	print(f'  - sensitive_apis: {len(sensitive_list)}')
	print(f'  - sensitive_func_apis: {len(sensitive_func_list)}')
	print(f'  - sources_sinks_apis: {len(sources_sinks_list)}')
	print(f'  - Total whitelist size: {len(white_set)}')
	
	return white_set

# Add graph statistics calculation function
def calculate_graph_statistics(graph):
	"""Calculate graph statistics, return the number of different edge types"""
	# Separate different types of edges
	call_graph_edges = []
	taint_edges = []
	mixed_edges = []  # Call graph edge and taint edge
	
	for u, v, data in graph.edges(data=True):
		edge_types = data.get('edge_types', set())
		
		if 'call_graph' in edge_types and 'taint' in edge_types:
			mixed_edges.append((u, v))
		elif 'call_graph' in edge_types:
			call_graph_edges.append((u, v))
		elif 'taint' in edge_types:
			taint_edges.append((u, v))
		else:
			# Process edges without type tags, default to call graph edge
			call_graph_edges.append((u, v))
	
	return {
		'num_nodes': graph.number_of_nodes(),
		'num_call_edges': len(call_graph_edges),
		'num_taint_edges': len(taint_edges),
		'num_mixed_edges': len(mixed_edges),
		'total_edges': len(call_graph_edges) + len(taint_edges) + len(mixed_edges)
	}

def process_single_sample(call_graph_file, taint_path_file, sample_id, white_set, timeout_seconds=30):
	"""Process a single sample, return heterogeneous graph data (optimized: whitelist passed as parameter, with timeout)"""
	try:
		with timeout_context(timeout_seconds):
			# Parse call graph
			call_graph = parse_txt(call_graph_file)
			if not call_graph:
				return None
			
			# Get whitelist (if not passed, get it)
			if white_set is None:
				sensitive_taint_white_set = get_white_set()
			else:
				sensitive_taint_white_set = white_set
			
			# Create complete call graph
			G = create_call_graph(call_graph)
			nodes = set(G.nodes)
			
			# Filter whitelist nodes
			sensitive_taint_white_set = sensitive_taint_white_set & nodes
			if len(sensitive_taint_white_set) == 0:
				return None
			
			# Get parent nodes
			parent_nodes = get_parent_set(G, sensitive_taint_white_set)
			all_white_set = sensitive_taint_white_set | parent_nodes
			
			# Build heterogeneous graph
			H = reconstruct_graph(G, all_white_set)
			abstract_call_graph = H

			# Add taint edges
			if os.path.exists(taint_path_file):
				with open(taint_path_file, 'r') as f:
					taint_content = f.read()
				H = add_taint_edges(H, taint_content, all_white_set)
			else:
				print(f"taint_path_file: {taint_path_file} not found")
			graph_stats = calculate_graph_statistics(H)
		
			logging.info(f'sample_id: {sample_id}, {graph_stats}')
			return {
				'sample_id': sample_id,
				'graph': H,
				'abstract_call_graph': abstract_call_graph,
				'num_nodes': H.number_of_nodes(),
				'num_edges': H.number_of_edges(),
				'num_call_edges': graph_stats['num_call_edges'],
				'num_taint_edges': graph_stats['num_taint_edges'],
				'num_mixed_edges': graph_stats['num_mixed_edges'],
				'total_edges': graph_stats['total_edges']
			}
			
	except TimeoutError:
		logging.warning(f"Sample {sample_id} processing timed out ({timeout_seconds} seconds)")
		print(f"⏱ Sample processing timed out: {sample_id} (over {timeout_seconds} seconds)")
		return None
	except Exception as e:
		logging.error(f"Error processing sample {sample_id}: {e}")
		logging.error(f"Detailed error information: {traceback.format_exc()}")
		print(f"✗ Sample processing exception: {sample_id} - {str(e)}")
		return None

def batch_process_samples_async(call_graph_dir, taint_path_dir, label_value, 
								output_dir='./asg_dataset', 
								timeout_seconds=30,
								codebert_embeddings_path='../nodeRepresentation/codebert_api_embeddings.npz',
								permission_dir=None):
	"""
	Use ASGDataset class to asynchronously batch process samples
	
	Args:
		call_graph_dir (str): Call graph file directory
		taint_path_dir (str): Taint path file directory
		label_value (int): Label value (0: benign, 1: malware)
		output_dir (str): Output directory
		timeout_seconds (int): Processing timeout time (seconds)
		codebert_embeddings_path (str): CodeBERT embedding file path
		permission_dir (str): Permission files directory
	
	Returns:
		dict: Processing result information
	"""
	logging.info(f"Starting ASG dataset batch processing")
	logging.info(f"Call graph directory: {call_graph_dir}")
	logging.info(f"Taint path directory: {taint_path_dir}")
	logging.info(f"Label: {label_value} ({'malware' if label_value == 1 else 'benign'})")
	logging.info(f"Output directory: {output_dir}")
	logging.info(f"Timeout time: {timeout_seconds} seconds")
	logging.info(f"CodeBERT embedding file: {codebert_embeddings_path}")
	logging.info(f"Permission directory: {permission_dir}")
	
	print(f"Using ASGDataset to process samples...")
	print(f"Call graph directory: {call_graph_dir}")
	print(f"Taint path directory: {taint_path_dir}")
	print(f"Label: {label_value} ({'malware' if label_value == 1 else 'benign'})")
	print(f"Output directory: {output_dir}")
	print(f"Timeout time: {timeout_seconds} seconds")
	print(f"CodeBERT embedding file: {codebert_embeddings_path}")
	print(f"Permission directory: {permission_dir}")
	
	# Validate label value
	if label_value not in [0, 1]:
		error_msg = "label_value must be 0 (benign) or 1 (malware)"
		logging.error(error_msg)
		raise ValueError(error_msg)
	
	try:
		logging.info("Creating ASGDataset instance")
		# Create ASG dataset
		dataset = ASGDataset(
			root=output_dir,
			call_graph_dir=call_graph_dir,
			taint_path_dir=taint_path_dir,
			label_value=label_value,
			timeout_seconds=timeout_seconds,
			codebert_embeddings_path=codebert_embeddings_path,
			permission_dir=permission_dir
		)
		
		logging.info(f"ASG dataset created successfully, dataset size: {len(dataset)}")
		print(f"\nASG dataset created successfully!")
		print(f"Dataset size: {len(dataset)} samples")
		
	except Exception as e:
		error_msg = f"ASG dataset processing failed: {str(e)}"
		logging.error(error_msg)
		print(error_msg)
		logging.error(f"Detailed error information: {traceback.format_exc()}")
		traceback.print_exc()
		return None

class CodeBERTEmbeddingLoader:
	"""
	CodeBERT embedding loader, specifically designed to load and use pre-trained CodeBERT embeddings
	"""
	
	def __init__(self, embeddings_path='../nodeRepresentation/codebert_api_embeddings.npz'):
		"""
		Initialize CodeBERT embedding loader
		
		Args:
			embeddings_path (str): CodeBERT embedding file path
		"""
		self.embeddings_path = embeddings_path
		
		# CodeBERT embedding related properties
		self.codebert_embeddings = None
		self.codebert_api_list = None
		self.codebert_api_to_index = None
		self.codebert_feature_dim = None
		
		self._load_codebert_embeddings()
	
	def _load_codebert_embeddings(self):
		"""Load pre-trained CodeBERT embeddings"""
		try:
			# Check if file exists
			if not os.path.exists(self.embeddings_path):
				raise FileNotFoundError(f"CodeBERT embedding file does not exist: {self.embeddings_path}")
			
			# Load embedding data
			data = np.load(self.embeddings_path, allow_pickle=True)
			self.codebert_embeddings = data['embeddings']
			self.codebert_api_list = data['api_list'].tolist()
			self.codebert_api_to_index = data['api_to_index'].item()
			model_name = str(data['model_name'])
			
			self.codebert_feature_dim = self.codebert_embeddings.shape[1]
			
			print(f"✓ Successfully loaded CodeBERT embeddings:")
			print(f"  - Model: {model_name}")
			print(f"  - Number of APIs: {len(self.codebert_api_list)}")
			print(f"  - Embedding dimension: {self.codebert_feature_dim}")
			print(f"  - File path: {self.embeddings_path}")
			
			
		except Exception as e:
			error_msg = f"Could not load CodeBERT embedding file: {e}"
			print(error_msg)
			logging.error(error_msg)
			raise RuntimeError(error_msg)
	
	
	def get_api_embedding(self, api_name):
		"""
		Get embedding vector for a single API
		
		Args:
			api_name (str): API name
		
		Returns:
			torch.Tensor: API feature vector, returns zero vector if API does not exist
		"""
		if api_name in self.codebert_api_to_index:
			index = self.codebert_api_to_index[api_name]
			embedding = self.codebert_embeddings[index]
			return torch.FloatTensor(embedding)
		else:
			# If API is not in CodeBERT embeddings, return zero vector
			print(f"Warning: API '{api_name}' not found in CodeBERT embeddings, using zero vector")
			return torch.zeros(self.codebert_feature_dim, dtype=torch.float32)
	
	def batch_get_api_embeddings(self, abstract_call_graph, api_names):
		"""
		Batch get API embedding vectors
		
		Args:
			api_names (list): List of API names
		
		Returns:
			torch.Tensor: Batch feature matrix
		"""
		if not api_names:
			return torch.FloatTensor([]).reshape(0, self.codebert_feature_dim)
		
		features = []
		found_count = 0
		not_found_count = 0
		avg_count = 0
		
		finished_embedding_dict = {}

		for api_name in api_names:
			if api_name in self.codebert_api_to_index:
				index = self.codebert_api_to_index[api_name]
				embedding = self.codebert_embeddings[index]
				embedding = torch.FloatTensor(embedding)
				features.append(embedding)
				finished_embedding_dict[api_name] = embedding.tolist()
				found_count += 1
			else:
				not_found_count += 1
				avg_list = []
				
				successors = list(abstract_call_graph.successors(api_name))

				#print('**********avg info**********')
				#print(type(embedding))
				#print(f'Fininshed apis: {finished_embedding_dict.keys()}')
				#print(f'Now, processing {api_name}')
				#print(f'Successors of {api_name}: {successors}')
				successors_in_codebert = [successor for successor in successors if successor in self.codebert_api_to_index]
				if len(successors_in_codebert) > 0:
					avg_count += 1
				for successor in successors:
					if successor in self.codebert_api_to_index:
						index = self.codebert_api_to_index[successor]
						embedding = self.codebert_embeddings[index]
						embedding = torch.FloatTensor(embedding)
						avg_list.append(embedding)
					#print(f'child of {api_name}: {successor}')
				if len(avg_list) == 0:
					logging.info(f'{api_name} No embeddable successors found, successors: {successors}, successors_in_codebert: {successors_in_codebert}')
				avg_embedding = torch.mean(torch.stack(avg_list), dim=0)

				finished_embedding_dict[api_name] = avg_embedding.tolist()
				#print(f'avg embedding of {api_name}: {avg_embedding}')
				#print('**********avg info**********')
				features.append(avg_embedding)
		# Output statistics
		if len(api_names) > 0:
			logging.info(f"Batch embedding statistics: Total={len(api_names)}, Found={found_count}, Not found={not_found_count}, Embeddings to average={avg_count}, Finished embeddings={len(finished_embedding_dict)}")
		
		return torch.stack(features)

class ASGDataset(Dataset):
	"""
	ASG (Abstract Syntax Graph) dataset class, inheriting from torch_geometric.data.Dataset
	Uses pre-calculated CodeBERT embeddings for API feature encoding
	"""
	
	def __init__(self, root, call_graph_dir=None, taint_path_dir=None, label_value=None, 
				 transform=None, pre_transform=None,
				 timeout_seconds=30, codebert_embeddings_path='../nodeRepresentation/codebert_api_embeddings.npz',
				 permission_dir=None):
		"""
		Initialize ASG dataset
		
		Args:
			root (str): Dataset root directory
			call_graph_dir (str): Call graph file directory
			taint_path_dir (str): Taint path file directory
			label_value (int): Label value (0: benign, 1: malware)
			transform (callable, optional): Data transformation function
			pre_transform (callable, optional): Pre-processing transformation function
			timeout_seconds (int): Processing timeout time (seconds)
			codebert_embeddings_path (str): CodeBERT embedding file path
			permission_dir (str): Permission files directory
		"""
		logging.info(f"Starting ASGDataset initialization")
		logging.info(f"Parameters - root: {root}, call_graph_dir: {call_graph_dir}, taint_path_dir: {taint_path_dir}")
		logging.info(f"Parameters - codebert_embeddings_path: {codebert_embeddings_path}")
		
		self.call_graph_dir = call_graph_dir
		self.taint_path_dir = taint_path_dir
		self.label_value = label_value
		self.timeout_seconds = timeout_seconds
		self.codebert_embeddings_path = codebert_embeddings_path
		self.permission_dir = permission_dir
		
		# Initialize CodeBERT embedding loader
		logging.info("Initializing CodeBERT embedding loader")
		self.embedding_loader = CodeBERTEmbeddingLoader(
			embeddings_path=codebert_embeddings_path,
		)
		
		# Track processing status
		self._processed_samples = 0
		self._total_samples = 0
		
		# Call parent class initialization, this sets self.root, self.raw_dir, self.processed_dir, etc.
		super().__init__(root, transform, pre_transform)
		
		# Ensure processed_dir exists (inherited from parent class, but we explicitly check)
		if not hasattr(self, 'processed_dir') or self.processed_dir is None:
			self.processed_dir = os.path.join(self.root, 'processed')
		
		
		# Ensure directories exist
		os.makedirs(self.processed_dir, exist_ok=True)
		
		logging.info(f"ASGDataset initialization completed:")
		logging.info(f"  - Dataset root: {root}")
		logging.info(f"  - Processed data directory: {self.processed_dir}")
		logging.info(f"  - Call graph directory: {call_graph_dir}")
		logging.info(f"  - Taint path directory: {taint_path_dir}")
		logging.info(f"  - Label value: {label_value}")
		
		print(f"Creating ASG dataset:")
		print(f"  - Call graph directory: {call_graph_dir}")
		print(f"  - Taint path directory: {taint_path_dir}")
		print(f"  - Permission directory: {permission_dir}")
		print(f"  - Label: {label_value} ({'malware' if label_value == 1 else 'benign'})")
		print(f"  - Output directory: {root}")
		print(f"  - CodeBERT embedding: {codebert_embeddings_path}")
	
	@property
	def raw_file_names(self):
		"""
		Original file name list 
		"""
		if self.call_graph_dir is None or not os.path.exists(self.call_graph_dir):
			return []
		
		# Get all call graph files
		call_graph_files = [f for f in os.listdir(self.call_graph_dir) if f.endswith('.txt')]
		return call_graph_files
	
	@property
	def processed_file_names(self):
		# Check if processed files already exist
		existing_files = glob.glob(os.path.join(self.processed_dir, '*.pt'))
		if existing_files:
			return [os.path.basename(f) for f in existing_files]
		
		# If no processed files, generate file names based on original files
		raw_files = self.raw_file_names
		processed_names = []
		for raw_file in raw_files:
			# Format: filename.pt (remove .txt extension, add .pt)
			base_name = raw_file.replace('.txt', '')
			processed_names.append(f'{base_name}.pt')
		
		return processed_names
	
	def download(self):
		"""
		Download data (if needed), here no download is needed
		"""
		pass
	
	def process(self):
		"""
		Main method for processing data, using asynchronous + multi-process to improve efficiency
		"""
		if not self.call_graph_dir or not os.path.exists(self.call_graph_dir):
			print("Warning: Call graph directory does not exist, skipping processing")
			return
		
		# Pre-load whitelist
		print("Pre-loading whitelist...")
		white_set = get_white_set()
		print(f"Whitelist pre-loaded, total {len(white_set)} sensitive APIs")
		
		# Get original files
		raw_files = self.raw_file_names
		total_files = len(raw_files)
		self._total_samples = total_files
		
		if total_files == 0:
			print("No original files to process")
			return
		
		print(f"Starting multi-process processing of {total_files} samples...")
		print("=" * 80)
		
		# Prepare multi-process parameters
		sample_args_list = []
		for file_name in raw_files:
			sample_id = file_name.replace('.txt', '')
			call_graph_file = os.path.join(self.call_graph_dir, file_name)
			taint_path_file = os.path.join(self.taint_path_dir, file_name) if self.taint_path_dir else ""
			
			# Parameters: (call_graph_file, taint_path_file, sample_id, processed_dir, label_value, embedding_loader, white_set, permission_dir, timeout_seconds)
			sample_args = (call_graph_file, taint_path_file, sample_id, self.processed_dir, 
						  self.label_value, self.embedding_loader, white_set, self.permission_dir, self.timeout_seconds)
			sample_args_list.append(sample_args)
		
		# Process using multi-process
		successful_count, failed_count = self.mp_process_asg_samples(
			sample_args_list, 
			num_processes=None, 
			sample_timeout=self.timeout_seconds
		)
		
		# Save processing information
		self._processed_samples = successful_count

		print(f"\n=== ASG dataset multi-process processing completed ===")
		print(f"Successfully processed: {successful_count}/{total_files} samples")
		print(f"Failed samples: {failed_count} samples")
		print(f"CodeBERT API count: {len(self.embedding_loader.codebert_api_list)}")
		#print(f"Processed data saved in: {self.processed_dir}")
	
	def mp_process_asg_samples(self, sample_args_list, num_processes=None, sample_timeout=30):
		"""
		Process ASG samples using multi-process
		
		Args:
			sample_args_list (list): List of sample parameters
			num_processes (int): Number of processes
			sample_timeout (int): Single sample timeout time
		
		Returns:
			tuple: (successful_count, failed_count)
		"""
		if num_processes is None:
			num_processes = max(1, cpu_count() - 2)
		
		total_samples = len(sample_args_list)
		logging.info(f"Processing {total_samples} ASG samples using {num_processes} processes (each sample timeout: {sample_timeout} seconds)")
		print(f"ASG multi-process mode: Using {num_processes} processes")
		print(f"Sample processing timeout set to: {sample_timeout} seconds")
		print("=" * 80)
		
		start_time = time.time()
		successful_count = 0
		failed_count = 0
		timeout_count = 0
		
		try:
			with Pool(processes=num_processes) as pool:
				# Submit all tasks
				jobs = []
				for args in sample_args_list:
					job = pool.apply_async(process_asg_sample_wrapper, args)
					jobs.append(job)
				
				print(f"Submitted {len(jobs)} ASG tasks to process pool")
				print("Starting to collect ASG processing results...")
				
				# Collect results
				for i, job in enumerate(jobs):
					sample_id = sample_args_list[i][2]  # sample_id is the third parameter
					progress_percent = ((i + 1) / total_samples) * 100
					remaining_count = total_samples - (i + 1)
					
					print(f"\n[ASG multi-process progress {i+1}/{total_samples}] ({progress_percent:.1f}%) Sample: {sample_id}")
					print(f"Processed: {i} samples | Remaining: {remaining_count} samples")
					
					try:
						success = job.get(timeout=sample_timeout + 30)
						if success:
							successful_count += 1
							print(f"✓ ASG sample processing successful: {sample_id}")
						else:
							failed_count += 1
							print(f"✗ ASG sample processing failed: {sample_id}")
						
					except Exception as e:
						failed_count += 1
						if "timeout" in str(e).lower():
							timeout_count += 1
							logging.warning(f"ASG sample {sample_id} multi-process timeout")
							print(f"⏱ ASG sample multi-process timeout: {sample_id} - {str(e)}")
						else:
							logging.error(f"ASG sample {sample_id} processing failed: {str(e)}")
							logging.error(f"Detailed error information: {traceback.format_exc()}")
							print(f"✗ ASG sample processing exception: {sample_id} - {str(e)}")
					
					print("-" * 80)
				
				pool.close()
				pool.join()
				
		except Exception as e:
			logging.error(f"ASG multi-process processing error: {str(e)}")
			logging.error(f"Detailed error information: {traceback.format_exc()}")
			return 0, total_samples
		
		end_time = time.time()
		processing_time = end_time - start_time
		success_rate = (successful_count / total_samples) * 100 if total_samples > 0 else 0
		
		print(f"\n" + "=" * 80)
		print(f"=== ASG multi-process processing completed ===")
		print(f"Total samples: {total_samples}")
		print(f"Successfully processed: {successful_count} samples ({success_rate:.1f}%)")
		print(f"Failed samples: {failed_count} samples")
		print(f"Failed due to timeout: {timeout_count} samples")
		print(f"Processing time: {processing_time:.2f} seconds")
		print(f"Average time per sample: {processing_time/total_samples:.2f} seconds")
		
		logging.info(f"ASG successfully processed {successful_count}/{total_samples} samples, timeout: {timeout_count}, time: {processing_time:.2f} seconds")
		
		return successful_count, failed_count
	
	def len(self):
		"""
		Return dataset size
		"""
		if not hasattr(self, '_len'):
			# Calculate the number of processed files
			pt_files = glob.glob(os.path.join(self.processed_dir, '*.pt'))
			self._len = len(pt_files)
		return self._len
	
	def get(self, idx):
		"""
		Args:
			idx (int): Sample index
		
		Returns:
			HeteroData: Heterogeneous graph data
		"""
		try:
			# Get the list of processed file names
			processed_files = self.processed_file_names
			if idx >= len(processed_files):
				raise IndexError(f"Index {idx} out of bounds, dataset size is {len(processed_files)}")
			
			# Construct file path
			file_name = processed_files[idx]
			data_file = os.path.join(self.processed_dir, file_name)
			
			# Load data
			hetero_data = torch.load(data_file)
			
			# Apply transformation (if any)
			if self.transform is not None:
				hetero_data = self.transform(hetero_data)
			
			return hetero_data
		
		except Exception as e:
			print(f"Error loading {file_name}: {e}")
			# Return an empty heterogeneous graph data
			empty_data = HeteroData()
			empty_data.sample_id = f"error_{idx}"
			empty_data.label = torch.tensor([self.label_value or 0], dtype=torch.long)
			empty_data['api'].x = torch.FloatTensor([]).reshape(0, self.codebert_feature_dim)
			empty_data[('api', 'call_graph', 'api')].edge_index = torch.LongTensor([[], []])
			empty_data[('api', 'taint', 'api')].edge_index = torch.LongTensor([[], []])
			return empty_data
	

def process_asg_sample_wrapper(call_graph_file, taint_path_file, sample_id, processed_dir, label_value, embedding_loader, white_set, permission_dir, timeout_seconds=30):
	"""
	Wrapper function for processing ASG samples
	
	Args:
		call_graph_file (str): Call graph file path
		taint_path_file (str): Taint path file path
		sample_id (str): Sample ID
		processed_dir (str): Processed data output directory
		label_value (int): Sample label
		embedding_loader (CodeBERTEmbeddingLoader): CodeBERT embedding loader instance
		white_set (set): Whitelist set
		permission_dir (str): Permission files directory
		timeout_seconds (int): Processing timeout time
	
	Returns:
		bool: Returns True if processing is successful, False otherwise
	"""
	try:
		# Process a single sample

		if not os.path.exists(f'{permission_dir}/{sample_id}_Permission.txt'):
			logging.warning(f"Permission file does not exist, ASG sample processing failed: {sample_id}")
			traceback.print_exc()
			return False

		result = process_single_sample(
			call_graph_file, 
			taint_path_file, 
			sample_id, 
			white_set, 
			timeout_seconds
		)
		
		if result is None:
			logging.warning(f"ASG sample processing failed: {sample_id}")
			traceback.print_exc()
			return False

		# Convert to heterogeneous graph format (using CodeBERT embeddings)
		hetero_data = convert_networkx_to_hetero_data_asg_codebert(
			result['graph'], 
			result['abstract_call_graph'], 
			sample_id, 
			label_value,
			embedding_loader,
			permission_dir
		)
		
		# Save processed data
		output_file = os.path.join(processed_dir, f'{sample_id}.pt')
		os.makedirs(processed_dir, exist_ok=True)  # Ensure directory exists
		torch.save(hetero_data, output_file)
		
		logging.info(f"ASG sample processing successful: {sample_id}")
		return True
		
	except Exception as e:
		logging.error(f"ASG sample processing wrapper error {sample_id}: {str(e)}")
		logging.error(f"Detailed error information: {traceback.format_exc()}")
		return False

def convert_networkx_to_hetero_data_asg_codebert(nx_graph, abstract_call_graph, sample_id, label, embedding_loader, permission_dir):
	"""
	Convert NetworkX graph to HeteroData format, use CodeBERT embeddings to encode node features (multi-process version)
	
	Args:
		nx_graph (nx.DiGraph): NetworkX directed graph, with edge_types attribute
		sample_id (str): Sample ID
		label (int): Sample label
		embedding_loader (CodeBERTEmbeddingLoader): CodeBERT embedding loader instance
		permission_dir (str): Permission files directory
	
	Returns:
		HeteroData: Converted heterogeneous graph data
	"""
	# Create heterogeneous graph data object
	hetero_data = HeteroData()
	
	# Add sample metadata
	hetero_data.sample_id = sample_id
	hetero_data.y = torch.tensor([label], dtype=torch.long)
	hetero_data.permission = process_permission_feature(f'{permission_dir}/{sample_id}_Permission.txt', _permissions)

	nodes = list(abstract_call_graph.nodes())
	num_nodes = len(nodes)
	
	
	if num_nodes == 0:
		# Handle empty graph case, this is not needed as at least one dummyMainMethod() node is present
		hetero_data['api'].x = torch.FloatTensor([]).reshape(0, embedding_loader.codebert_feature_dim)
		hetero_data[('api', 'call_graph', 'api')].edge_index = torch.LongTensor([[], []])
		hetero_data[('api', 'taint', 'api')].edge_index = torch.LongTensor([[], []])
		return hetero_data
	
	# Create node to index mapping
	node_to_idx = {node: idx for idx, node in enumerate(nodes)}

	
	
	# Create node features using CodeBERT embeddings
	node_features = embedding_loader.batch_get_api_embeddings(abstract_call_graph, nodes)
	hetero_data['api'].x = node_features
	
	# Separate different types of edges
	call_graph_edges = []
	taint_edges = []
	
	for u, v, edge_data in nx_graph.edges(data=True):
		edge_types = edge_data.get('edge_types', set())
		u_idx = node_to_idx[u]
		v_idx = node_to_idx[v]
		
		if 'call_graph' in edge_types:
			call_graph_edges.append([u_idx, v_idx])
		
		if 'taint' in edge_types:
			taint_edges.append([u_idx, v_idx])
	
	# Create edge index tensors
	if call_graph_edges:
		call_edge_index = torch.LongTensor(call_graph_edges).t().contiguous()
	else:
		call_edge_index = torch.LongTensor([[], []]).long()
	
	if taint_edges:
		taint_edge_index = torch.LongTensor(taint_edges).t().contiguous()
	else:
		taint_edge_index = torch.LongTensor([[], []]).long()
	
	# Add edges to heterogeneous graph
	hetero_data[('api', 'call_graph', 'api')].edge_index = call_edge_index
	hetero_data[('api', 'taint', 'api')].edge_index = taint_edge_index
	
	# Add additional metadata
	# hetero_data.node_names = nodes  # Save node names for debugging
	hetero_data.node_to_idx = node_to_idx  # Save node mapping
	
	return hetero_data

def main():
	parser = argparse.ArgumentParser(description='ASG Heterogeneous Graph Dataset Generator')
	parser.add_argument('--mode', type=str, choices=['asg', 'process', 'single', 'merge', 'demo'], 
						default='asg', help='Run mode')
	parser.add_argument('--call_graph_dir', type=str, default='../../Dataset/2016_no_remote/malware/call_graph', required=False, 
						help='Call graph file directory path')
	parser.add_argument('--taint_path_dir', type=str, default='../../Dataset/2016_no_remote/malware/taint_path', required=False, 
						help='Taint path file directory path')
	parser.add_argument('--output_dir', type=str, default='../../TensorDataset/2016_no_remote/malware', 
						help='Output directory path')
	parser.add_argument('--label', type=int, choices=[0, 1], default=1,  
						help='Sample label (0: benign, 1: malware)')
	parser.add_argument('--codebert_embeddings_path', type=str, default='../nodeRepresentation/codebert_api_embeddings.npz',
						help='CodeBERT embedding file path')
	parser.add_argument('--timeout', type=int, default=30, 
						help='Processing timeout time (seconds)')
	parser.add_argument('--log_file', type=str, default='../../Log/asg_dataset.log',
						help='Log file path')
	parser.add_argument('--permission_dir', type=str, default='../../Raw/1980/permission',
						help='Permission files directory path')
	
	parser.add_argument('--dataset_dirs', type=str, nargs='+', help='List of dataset directories to merge (merge mode)')
	
	args = parser.parse_args()
	
	# Initialize logging system
	setup_logging(args.log_file)
	
	logging.info(f"Program startup parameters: {vars(args)}")
	
	print("=" * 80)
	print("ASG Heterogeneous Graph Dataset Generator")
	print("Using pre-calculated CodeBERT embeddings to process API features")
	print("=" * 80)
	
	if args.mode == 'asg':
		logging.info("Starting ASG dataset creation mode")
		print("Run mode: ASG dataset creation")
		
		if not args.call_graph_dir or not args.taint_path_dir or args.label is None:
			error_msg = "Error: ASG mode requires --call_graph_dir, --taint_path_dir, and --label parameters"
			logging.error(error_msg)
			print(error_msg)
			parser.print_help()
			sys.exit(1)
		
		print(f"Creating ASG dataset:")
		print(f"  - Call graph directory: {args.call_graph_dir}")
		print(f"  - Taint path directory: {args.taint_path_dir}")
		print(f"  - Label: {args.label} ({'malware' if args.label == 1 else 'benign'})")
		print(f"  - Output directory: {args.output_dir}")
		print(f"  - CodeBERT embedding: {args.codebert_embeddings_path}")
		print(f"  - Permission directory: {args.permission_dir}")
		print(f"  - Log file: {args.log_file}")
		
		# Use ASGDataset to process samples
		result = batch_process_samples_async(
			call_graph_dir=args.call_graph_dir,
			taint_path_dir=args.taint_path_dir,
			label_value=args.label,
			output_dir=args.output_dir,
			timeout_seconds=args.timeout,
			codebert_embeddings_path=args.codebert_embeddings_path,
			permission_dir=args.permission_dir
		)
		
	# delete pre_filter.pt and pre_transform.pt
	if os.path.exists(f'{args.output_dir}/processed/pre_filter.pt'):
		os.remove(f'{args.output_dir}/processed/pre_filter.pt')
	if os.path.exists(f'{args.output_dir}/processed/pre_transform.pt'):
		os.remove(f'{args.output_dir}/processed/pre_transform.pt')
	if os.path.exists(f'{args.output_dir}/raw'):
		shutil.rmtree(f'{args.output_dir}/raw')
	logging.info("Program execution completed")
	print("Program execution completed!")


# Main program
if __name__ == "__main__":
	main()

	# running example:
	# python generate_absGraph.py --call_graph_dir ../../Raw/2022/call_graph/malware/graphs --taint_path_dir ../../Raw/2022/taint_path/malware/taint_path_txt/ --permission_dir ../../Raw/2022/permission/malware/permissions/ --output_dir ../../Dataset/2022/malware --label 1 --codebert_embeddings_path ../nodeRepresentation/codebert_api_embeddings.npz --timeout 60 
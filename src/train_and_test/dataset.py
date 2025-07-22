import os
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from typing import List, Tuple
import numpy as np

class MalwareHeteroDataset(Dataset):
    """
    Heterogeneous graph malware detection dataset
    """
    def __init__(self, root_dir: str, from_year, to_year, split: str = 'train', random_state: int = 42, use_permission: bool = True):
        """
        Args:
            root_dir: Root directory path for data
            split: 'train' or 'test'
            random_state: Random seed
            use_permission: Whether to use permission features
        """
        self.root_dir = root_dir
        self.split = split
        self.random_state = random_state
        self.use_permission = use_permission
        self.from_year = from_year
        self.to_year = to_year
        
        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Load file paths and labels
        self.file_paths, self.labels = self._load_file_paths(from_year, to_year)
        
        
        self.data_paths = self.file_paths
        self.data_labels = self.labels

            
        print(f"{split.upper()} set: {len(self.data_paths)} samples")
        print(f"  - Malware: {sum(self.data_labels)}")
        print(f"  - Benign: {len(self.data_labels) - sum(self.data_labels)}")

        # Get permission feature dimension (if used)
        if self.use_permission:
            sample_data = torch.load(self.data_paths[0])
            if hasattr(sample_data, 'permission'):
                self.permission_dim = sample_data.permission.shape[1]
                print(f"Permission feature dimension: {self.permission_dim}")
            else:
                print("Warning: permission field not found in data, disabling permission features")
                self.use_permission = False
                self.permission_dim = 0
        else:
            self.permission_dim = 0
    
    def _load_file_paths(self, from_year, to_year) -> Tuple[List[str], List[int]]:
        """Load all file paths and corresponding labels"""
        file_paths = []
        labels = []
        print(f'Loading {from_year} to {to_year} data ...')
        #malware_dir = os.path.join(self.root_dir, 'features', 'malware', 'no_remote', 'processed')
        #benign_dir = os.path.join(self.root_dir, 'features', 'benign', 'no_remote', 'processed')

        if from_year == "MYST" or to_year == "MYST":
            for year in ["MYST"]:

                print(f'Now, loading {year} data ...')
                malware_dir = f'{self.root_dir}/{year}/malware/{self.split}'
                benign_dir = f'{self.root_dir}/{year}/benign/{self.split}'
                
                # load malware (label=1)
                if os.path.exists(malware_dir):
                    malware_files = [f for f in os.listdir(malware_dir) if f.endswith('.pt')]
                    for file in malware_files:
                        file_paths.append(os.path.join(malware_dir, file))
                        labels.append(1)
                # load benign (label=0)
                if os.path.exists(benign_dir):
                    benign_files = [f for f in os.listdir(benign_dir) if f.endswith('.pt')]
                    for file in benign_files:
                        file_paths.append(os.path.join(benign_dir, file))
                        labels.append(0)
                
            # shuffle data
            combined = list(zip(file_paths, labels))
            random.shuffle(combined)
            file_paths, labels = zip(*combined)
        elif type(from_year) == int:

            for year in range(from_year, to_year + 1):

                print(f'Now, loading {year} data ...')
                malware_dir = f'{self.root_dir}/{year}/malware/{self.split}'
                benign_dir = f'{self.root_dir}/{year}/benign/{self.split}'
                
                # Load malware files (label=1)
                if os.path.exists(malware_dir):
                    malware_files = [f for f in os.listdir(malware_dir) if f.endswith('.pt')]
                    for file in malware_files:
                        file_paths.append(os.path.join(malware_dir, file))
                        labels.append(1)
                # Load benign files (label=0)
                if os.path.exists(benign_dir):
                    benign_files = [f for f in os.listdir(benign_dir) if f.endswith('.pt')]
                    for file in benign_files:
                        file_paths.append(os.path.join(benign_dir, file))
                        labels.append(0)
                
            # Randomly shuffle data
            combined = list(zip(file_paths, labels))
            random.shuffle(combined)
            file_paths, labels = zip(*combined)
    
        return list(file_paths), list(labels)
    
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> HeteroData:
        """Get a single sample"""
        file_path = self.data_paths[idx]
        data = torch.load(file_path)
        
        # Create a clean HeteroData object, keeping only necessary information
        clean_data = HeteroData()
        #clean_data.sample_id = data.sample_id
        # Copy node features
        clean_data['sample_id'] = data.sample_id
        clean_data['api'].x = data['api'].x

        clean_data['nodes'] = list(data['node_to_idx'].keys())
        
        # Copy edge information
        clean_data[('api', 'call_graph', 'api')].edge_index = data[('api', 'call_graph', 'api')].edge_index
        clean_data[('api', 'taint', 'api')].edge_index = data[('api', 'taint', 'api')].edge_index
        
        # Add permission features (if used)
        if self.use_permission and hasattr(data, 'permission'):
            clean_data.permission = data.permission  # [1, 310]
        
        clean_data.y = data.y
        
        return clean_data
    
    def get_permission_dim(self) -> int:
        """Get permission feature dimension"""
        return self.permission_dim if self.use_permission else 0
        
def create_data_loaders(root_dir: str, from_year: int, to_year: int, batch_size: int = 32, random_state: int = 42, num_workers: int = 32, use_permission: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create training and testing data loaders
    
    Args:
        root_dir: Data root directory
        batch_size: Batch size
        test_size: Test set ratio
        random_state: Random seed
        num_workers: Number of data loading processes
    
    Returns:
        train_loader, test_loader
    """
    # Create training and testing datasets
    train_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='train', random_state=random_state, use_permission=use_permission)
    validate_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='validate', random_state=random_state, use_permission=use_permission)
    test_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='test', random_state=random_state, use_permission=use_permission)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )

    validate_loader = DataLoader(
        validate_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    permission_dim = train_dataset.get_permission_dim()
    
    return train_loader, validate_loader, test_loader, permission_dim

def create_concept_drift_test_loaders(root_dir: str, from_year: int, to_year: int, batch_size: int = 32, random_state: int = 42, num_workers: int = 32, use_permission: bool = True) -> Tuple[DataLoader, int]:
    """
    Create test data loaders, combining data from train, validate, and test splits
    
    Args:
        root_dir: Data root directory
        batch_size: Batch size
        random_state: Random seed
        num_workers: Number of data loading processes
        use_permission: Whether to use permission features
    
    Returns:
        test_loader, permission_dim
    """

    # Create datasets for three different splits
    train_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='train', random_state=random_state, use_permission=use_permission)
    validate_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='validate', random_state=random_state, use_permission=use_permission)
    test_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='test', random_state=random_state, use_permission=use_permission)
    
    # Combine file paths and labels from three datasets
    combined_paths = train_dataset.data_paths + validate_dataset.data_paths + test_dataset.data_paths
    combined_labels = train_dataset.data_labels + validate_dataset.data_labels + test_dataset.data_labels
    
    # Create a new combined dataset
    combined_dataset = MalwareHeteroDataset(root_dir, from_year, to_year, split='test', random_state=random_state, use_permission=use_permission)
    # Replace the combined dataset's paths and labels
    combined_dataset.data_paths = combined_paths
    combined_dataset.data_labels = combined_labels
    
    print(f"Combined dataset: {len(combined_dataset.data_paths)} samples")
    print(f"  - Malware: {sum(combined_dataset.data_labels)}")
    print(f"  - Benign: {len(combined_dataset.data_labels) - sum(combined_dataset.data_labels)}")
    
    test_loader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    permission_dim = combined_dataset.get_permission_dim()
    
    return test_loader, permission_dim

if __name__ == "__main__":
    # Test data loader
    
    root_dir = "../../Dataset/RQ3"
    train_loader, validate_loader, test_loader, permission_dim = create_data_loaders(root_dir, from_year="MYST", to_year="MYST", batch_size=8, num_workers=16, use_permission=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validate batches: {len(validate_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Permission dimension: {permission_dim}")
    
    # Check first batch
    for batch in train_loader:
        print(f"Batch node types: {batch.node_types}")
        print(f"Batch edge types: {batch.edge_types}")
        print(f"API nodes shape: {batch['api'].x.shape}")
        print(f"Sample ID: {batch['sample_id']}")
        print(f"Labels shape: {batch.y.shape}")
        print(f"Labels: {batch.y}")
        print(f"Nodes: {batch['nodes']}")
        print(f"Call graph edges shape: {batch[('api', 'call_graph', 'api')].edge_index.shape}")
        print(f"Taint edges shape: {batch[('api', 'taint', 'api')].edge_index.shape}")
        
        if hasattr(batch, 'permission'):
            print(f"Permission features shape: {batch.permission.shape}")
            print(f"Permission features sample (first row, first 10 values): {batch.permission[0][:10]}...")
            print(batch.permission[1])
        break 
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import json
import pandas as pd
import argparse
from torch.nn.functional import softmax
from dataset import create_data_loaders
from model import create_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_info():
    """Check GPU information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return gpu_count
    else:
        logger.info("No GPU available, will use CPU")
        return 0

class MalwareTrainer:
    """Malware detection trainer"""
    
    def __init__(self, config, output_dir):
        self.config = config
        
        # GPU device selection
        gpu_id = config.get('gpu_id', 0)  # Default to GPU 0
        if torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                logger.warning(f"GPU {gpu_id} not available. Available GPUs: {torch.cuda.device_count()}. Using GPU 0.")
                gpu_id = 0
            self.device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)  # Set current GPU
            logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            logger.info("CUDA not available. Using CPU.")
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        #self.output_dir = config.get('output_dir', 'outputs')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed
        self.set_seed(config.get('seed', 42))
        
        # Create data loaders
        self.train_loader, self.validate_loader, self.test_loader, self.permission_dim = create_data_loaders(
            root_dir=config['data_dir'],
            from_year=config['from_year'],
            to_year=config['to_year'],
            batch_size=config['batch_size'],
            random_state=config.get('seed'),
            num_workers=config.get('num_workers'),
            use_permission=config.get('use_permission', True)
        )
        
        # Get input dimension
        sample_batch = next(iter(self.train_loader))
        self.input_dim = sample_batch['api'].x.shape[1]
        logger.info(f"API node feature dimension: {self.input_dim}")
        
        # Create model
        self.model = create_model(
            input_dim=self.input_dim,
            permission_dim=self.permission_dim,
            model_type=config.get('model_type'),
            hidden_dim=config.get('hidden_dim'),
            num_layers=config.get('num_layers'),
            num_classes=config.get('num_classes'),
            pooling=config.get('pooling'),
            use_permission=config.get('use_permission'),
            use_attention=config.get('use_attention'),
            use_residual=config.get('use_residual')
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Check if attention mechanism is used
        if hasattr(self.model, 'use_attention'):
            logger.info(f"Attention mechanism: {'Enabled' if self.model.use_attention else 'Disabled'}")
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate'),
            weight_decay=config.get('weight_decay')
        )
        
        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'plateau')
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.get('epochs', 100)
            )
        else:
            self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'validate_loss': [], 'validate_acc': [],
            'validate_precision': [], 'validate_recall': [], 'validate_f1': [], 'validate_auc': [],
            'test_loss': [], 'test_acc': [],
            'test_precision': [], 'test_recall': [], 'test_f1': [], 'test_auc': []
        }
        
        self.best_validate_acc = 0.0
        self.best_validate_f1 = 0.0
        self.best_test_acc = 0.0
        self.best_test_f1 = 0.0
        
        # Save initial model to ensure file exists
        self.save_model('best_validate_acc_model.pth')
        self.save_model('best_validate_f1_model.pth')
        self.save_model('best_test_acc_model.pth')
        self.save_model('best_test_f1_model.pth')
    
    def set_seed(self, seed):
        """Set random seed"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Collect predictions and labels
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc='Evaluating')
            for batch in pbar:
                batch = batch.to(self.device)
                
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                
                total_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Malware probability
        
        avg_loss = total_loss / len(loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, accuracy, precision, recall, f1, auc, all_labels, all_preds
    
    def extract_single_sample(self, batch, index):
        """Extract single sample from batch"""
        from torch_geometric.data import HeteroData
        
        single_data = HeteroData()
        
        # Get device information
        device = batch['api'].x.device
        
        # Get node indices for this sample - use correct batch attribute
        if hasattr(batch['api'], 'batch'):
            api_batch = batch['api'].batch
        else:
            # If no batch information, create simple batch indices
            batch_size = batch.y.size(0)
            num_nodes = batch['api'].x.size(0)
            # Assume nodes are evenly distributed across samples
            nodes_per_sample = num_nodes // batch_size
            api_batch = torch.repeat_interleave(torch.arange(batch_size, device=device), nodes_per_sample)
            if len(api_batch) < num_nodes:
                # Handle remaining nodes
                remaining = num_nodes - len(api_batch)
                api_batch = torch.cat([api_batch, torch.full((remaining,), batch_size-1, device=device)])
        
        sample_mask = (api_batch == index)
        
        # Ensure nodes are selected
        if not sample_mask.any():
            # If no corresponding nodes found, create minimal graph
            logger.warning(f"No nodes found for sample {index}, creating minimal graph")
            single_data['api'].x = batch['api'].x[:1]  # Take first node
            single_data[('api', 'call_graph', 'api')].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            single_data[('api', 'taint', 'api')].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            sample_mask = torch.zeros_like(api_batch, dtype=torch.bool)
            sample_mask[0] = True
        else:
            # Extract node features
            single_data['api'].x = batch['api'].x[sample_mask]
        
        # Extract edge information - need to remap node indices
        old_to_new = {}
        new_idx = 0
        for old_idx in torch.where(sample_mask)[0]:
            old_to_new[old_idx.item()] = new_idx
            new_idx += 1
        
        # Process call_graph edges
        call_edge_index = batch[('api', 'call_graph', 'api')].edge_index
        if call_edge_index.size(1) > 0:
            call_mask = sample_mask[call_edge_index[0]] & sample_mask[call_edge_index[1]]
            if call_mask.any():
                call_edges = call_edge_index[:, call_mask]
                # Remap node indices
                new_call_edges = torch.zeros_like(call_edges)
                for i in range(call_edges.size(1)):
                    src_idx = call_edges[0, i].item()
                    dst_idx = call_edges[1, i].item()
                    if src_idx in old_to_new and dst_idx in old_to_new:
                        new_call_edges[0, i] = old_to_new[src_idx]
                        new_call_edges[1, i] = old_to_new[dst_idx]
                single_data[('api', 'call_graph', 'api')].edge_index = new_call_edges
            else:
                single_data[('api', 'call_graph', 'api')].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            single_data[('api', 'call_graph', 'api')].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Process taint edges
        taint_edge_index = batch[('api', 'taint', 'api')].edge_index
        if taint_edge_index.size(1) > 0:
            taint_mask = sample_mask[taint_edge_index[0]] & sample_mask[taint_edge_index[1]]
            if taint_mask.any():
                taint_edges = taint_edge_index[:, taint_mask]
                # Remap node indices
                new_taint_edges = torch.zeros_like(taint_edges)
                for i in range(taint_edges.size(1)):
                    src_idx = taint_edges[0, i].item()
                    dst_idx = taint_edges[1, i].item()
                    if src_idx in old_to_new and dst_idx in old_to_new:
                        new_taint_edges[0, i] = old_to_new[src_idx]
                        new_taint_edges[1, i] = old_to_new[dst_idx]
                single_data[('api', 'taint', 'api')].edge_index = new_taint_edges
            else:
                single_data[('api', 'taint', 'api')].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            single_data[('api', 'taint', 'api')].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Extract other information
        if isinstance(batch['sample_id'], list):
            single_data['sample_id'] = batch['sample_id'][index]
        else:
            single_data['sample_id'] = batch['sample_id'][index].item() if batch['sample_id'].dim() > 0 else batch['sample_id'].item()
            
        if isinstance(batch['nodes'], list):
            single_data['nodes'] = batch['nodes'][index]
        else:
            single_data['nodes'] = batch['nodes'][index]
            
        single_data.y = batch.y[index:index+1]
        
        if hasattr(batch, 'permission') and batch.permission is not None:
            single_data.permission = batch.permission[index:index+1]
        
        return single_data
    
    def predict_with_attention(self, output_file='malware_predictions_with_attention.csv', top_k=5):
        """
        Predict using attention mechanism and output top k contributing nodes
        
        Args:
            output_file: Output CSV file path
            top_k: Output top k contributing nodes
        """
        if not hasattr(self.model, 'use_attention') or not self.model.use_attention:
            logger.warning("Model does not have attention mechanism enabled")
            return
        
        self.model.eval()
        results = []
        
        logger.info(f"Starting attention-based prediction...")
        logger.info(f"Attention mechanism enabled: {self.model.use_attention}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Predicting with attention")):
                batch = batch.to(self.device)
                
                # Get batch size
                batch_size = batch.y.size(0)
                
                # Process each sample in batch separately
                for i in range(batch_size):
                    # Extract single sample
                    sample_data = self.extract_single_sample(batch, i)
                    # Ensure single sample is also on correct device
                    sample_data = sample_data.to(self.device)
                    
                    # Get prediction results and attention weights
                    if self.model.use_attention:
                        logits, attention_weights = self.model(sample_data, return_attention=True)
                    else:
                        logits = self.model(sample_data)
                        attention_weights = None
                    
                    # Calculate prediction probability
                    probs = softmax(logits, dim=1)
                    pred_class = torch.argmax(logits, dim=1).item()
                    malware_prob = probs[0, 1].item()  # Malware probability
                    
                    # Get sample information
                    sample_id = sample_data['sample_id']
                    nodes = sample_data['nodes']
                    true_label = sample_data.y.item()
                    
                    # If predicted as malware and attention mechanism is used
                    if pred_class == 1 and attention_weights is not None:
                        # Get top k nodes
                        top_scores, top_indices = torch.topk(attention_weights, k=min(top_k, len(attention_weights)))
                        
                        # Get corresponding node names
                        top_node_names = [nodes[idx] for idx in top_indices.cpu().numpy()]
                        top_node_scores = top_scores.cpu().numpy()
                        
                        # Build result record
                        result = {
                            'sample_id': sample_id,
                            'true_label': true_label,
                            'predicted_label': pred_class,
                            'malware_probability': malware_prob
                        }
                        
                        # Add top k node information
                        for j in range(len(top_node_names)):
                            result[f'top_{j+1}_node'] = top_node_names[j]
                            result[f'top_{j+1}_score'] = top_node_scores[j]
                        
                        results.append(result)
        # Save results to CSV
        output_path = os.path.join(self.output_dir, output_file)
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            logger.info(f"\nPrediction results saved to: {output_path}")
            logger.info(f"Samples predicted as malware: {len(results)}")
            
            # Display statistics
            if self.model.use_attention:
                attention_samples = len([r for r in results if 'top_1_node' in r])
                logger.info(f"Samples with attention weights: {attention_samples}")
                
                # Display top nodes for first few samples
                logger.info("\nTop nodes for first 5 malware predictions:")
                for i, result in enumerate(results[:5]):
                    logger.info(f"\nSample {result['sample_id']}:")
                    logger.info(f"  True label: {result['true_label']}, Predicted: {result['predicted_label']}")
                    logger.info(f"  Malware probability: {result['malware_probability']:.4f}")
                    if 'top_1_node' in result:
                        for j in range(min(3, top_k)):  # Only show first 3
                            if f'top_{j+1}_node' in result:
                                logger.info(f"  Top {j+1}: {result[f'top_{j+1}_node']} (score: {result[f'top_{j+1}_score']:.4f})")
        else:
            logger.info("No samples predicted as malware")
    
    def train(self):
        """Complete training process"""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.get('epochs')):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Evaluation
            validate_loss, validate_acc, validate_precision, validate_recall, validate_f1, validate_auc, _, _ = self.evaluate(self.validate_loader)

            test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, _, _ = self.evaluate(self.test_loader)

            self.scheduler.step(validate_f1)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['validate_loss'].append(validate_loss)
            self.history['validate_acc'].append(validate_acc)
            self.history['validate_precision'].append(validate_precision)
            self.history['validate_recall'].append(validate_recall)
            self.history['validate_f1'].append(validate_f1)
            self.history['validate_auc'].append(validate_auc)

            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['test_precision'].append(test_precision)
            self.history['test_recall'].append(test_recall)
            self.history['test_f1'].append(test_f1)
            self.history['test_auc'].append(test_auc)   
            
            # Save best models
            if validate_acc > self.best_validate_acc:
                self.best_validate_acc = validate_acc
                self.save_model('best_validate_acc_model.pth')
            
            if validate_f1 > self.best_validate_f1:
                self.best_validate_f1 = validate_f1
                self.save_model('best_validate_f1_model.pth')
            
            if test_f1 > self.best_test_f1:
                self.best_test_f1 = test_f1
                self.save_model('best_test_f1_model.pth')
            
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.save_model('best_test_acc_model.pth')

            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.get('epochs', 100)} "
                f"({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Validate Loss: {validate_loss:.4f}, Validate Acc: {validate_acc:.4f}, "
                f"Validate Precision: {validate_precision:.4f}, Validate Recall: {validate_recall:.4f}, Validate F1: {validate_f1:.4f}, Validate AUC: {validate_auc:.4f} - "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}"
            )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Final evaluation
        self.final_evaluation()
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
    
    def final_evaluation(self):
        """Final evaluation"""
        logger.info("Final evaluation...")

        best_f1_path = os.path.join(self.output_dir, 'best_validate_f1_model.pth')
        if os.path.exists(best_f1_path):
            self.load_model('best_validate_f1_model.pth')
            logger.info("Loaded best F1 model for final evaluation")
        else:
            logger.warning("Best F1 model not found, using current model")
        
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, all_labels, all_preds = self.evaluate(self.test_loader)
        
        logger.info(f"Final Test Results:")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  F1-Score: {test_f1:.4f}")
        logger.info(f"  AUC: {test_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        
        # Save results
        results = {
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1),
            'auc': float(test_auc),
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    def save_model(self, filename):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'input_dim': self.input_dim
        }, os.path.join(self.output_dir, filename))
    
    def load_model(self, filename):
        """Load model"""
        checkpoint = torch.load(os.path.join(self.output_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_history(self):
        """Save training history"""
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['validate_loss'], label='Validate Loss')
        axes[0, 0].plot(self.history['test_loss'], label='Test Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['validate_acc'], label='Validate Acc')
        axes[0, 1].plot(self.history['test_acc'], label='Test Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        axes[1, 0].plot(self.history['validate_f1'], label='Validate F1')
        axes[1, 0].plot(self.history['validate_precision'], label='Validate Precision')
        axes[1, 0].plot(self.history['validate_recall'], label='Validate Recall')
        axes[1, 0].plot(self.history['test_f1'], label='Test F1')
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC curves
        axes[1, 1].plot(self.history['validate_auc'], label='Validate AUC')
        axes[1, 1].plot(self.history['test_auc'], label='Test AUC')
        axes[1, 1].set_title('AUC Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malware'],
                   yticklabels=['Benign', 'Malware'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

def RQ1_real_world_distribution():
    """Main function"""
    
    # Check GPU information
    gpu_count = check_gpu_info()
    
    for use_permission in [True]:
        for num_layers in [5]:
            for batch_size in [256]:
                for hidden_dim in [256]:
                    # Configuration parameters
                    logging.info(f"use_permission: {use_permission}, num_layers: {num_layers}, batch_size: {batch_size}, hidden_dim: {hidden_dim}")
                    config = {
                        'data_dir': '../../Dataset/RQ1',
                        'from_year': 2010,
                        'to_year': 2016,
                        'gpu_id': 0,  # Specify GPU ID to use (0 or 1)
                        'use_residual': True,
                        'use_permission': use_permission,
                        'use_attention': True,  # Enable attention mechanism
                        'batch_size': batch_size,
                        'epochs': 100,
                        'learning_rate': 0.001,
                        'weight_decay': 1e-5,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'num_classes': 2,  # Add missing classification parameter
                        'model_type': 'hgnn',
                        'pooling': 'concat',  # 'mean', 'max', 'concat'
                        'scheduler': 'plateau',  # 'plateau', 'cosine', None
                        'seed': 42,
                        'num_workers': 12,
                    }
                    
                    output_dir = f'../../Output/RQ1_Outputs/RQ1_ExA_permission:{config["use_permission"]}_{config["from_year"]}-{config["to_year"]}_layers:{config["num_layers"]}_epochs:{config["epochs"]}_batch:{config["batch_size"]}_hiddenDim:{config["hidden_dim"]}_residual:{config["use_residual"]}_lr_{config["learning_rate"]}'
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Output directory: {output_dir}")
                    # Create trainer and start training
                    trainer = MalwareTrainer(config, output_dir)
                    trainer.train()


def RQ3_malcious_component_localization():
    # Check GPU information
    gpu_count = check_gpu_info()
  
    use_permission = True
    num_layers = 5
    batch_size = 256
    hidden_dim = 256

    # Configuration parameters
    logging.info(f"use_permission: {use_permission}, num_layers: {num_layers}, batch_size: {batch_size}, hidden_dim: {hidden_dim}")
    config = {
        'data_dir': '../../Dataset/RQ3',
        'from_year': "MYST",
        'to_year': "MYST",
        'gpu_id': 0,  # Specify GPU ID to use (0 or 1)
        'use_residual': True,
        'use_permission': use_permission,
        'use_attention': True,  # Enable attention mechanism
        'batch_size': batch_size,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': 2,  # Add missing classification parameter
        'model_type': 'hgnn',
        'pooling': 'concat',  # 'mean', 'max', 'concat'
        'scheduler': 'plateau',  # 'plateau', 'cosine', None
        'seed': 42,
        'num_workers': 24,
    }
    
    # Create output directory
    #output_dir = f'output/new_feature_test_outputs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir = f'../../Output/RQ3_Outputs/RQ3_ExA_permission:{config["use_permission"]}_layers:{config["num_layers"]}_epochs:{config["epochs"]}_batch:{config["batch_size"]}_hiddenDim:{config["hidden_dim"]}_residual:{config["use_residual"]}_lr_{config["learning_rate"]}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    # Create trainer and start training
    trainer = MalwareTrainer(config, output_dir)
    trainer.train()
    
    
    # After training, use attention mechanism for prediction
    if config['use_attention']:
        logger.info("Starting attention-based prediction after training...")
        trainer.predict_with_attention(
            output_file='malware_predictions_with_attention.csv',
            top_k=50
        )
        logger.info("Attention-based prediction completed!")

if __name__ == "__main__":
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='ASGDroid Training Script')
    parser.add_argument('--rq', type=str, choices=['RQ1', 'RQ3'], required=True,
                       help='Choose which research question to execute: RQ1 or RQ3')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute based on argument
    if args.rq == 'RQ1':
        print("Executing RQ1: Real-world distribution analysis")
        RQ1_real_world_distribution()
    elif args.rq == 'RQ3':
        print("Executing RQ3: Malicious component localization")
        RQ3_malcious_component_localization()
    else:
        print(f"Invalid RQ choice: {args.rq}. Please choose 'RQ1' or 'RQ3'")
import torch
from dataset import create_concept_drift_test_loaders
import os
import json
import torch.nn as nn
import torch.optim as optim
from train import create_model
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestModel:
    def __init__(self, config, output_dir):
        self.config = config
        gpu_id = config.get('gpu_id', 0)
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
            logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device('cpu')
            
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.test_loader, self.permission_dim = create_concept_drift_test_loaders(
            root_dir=config['data_dir'],
            from_year=config['from_year'],
            to_year=config['to_year'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            use_permission=config['use_permission']
            )
        
        sample_batch = next(iter(self.test_loader))
        self.input_dim = sample_batch['api'].x.shape[1]
        logger.info(f"API node feature dimension: {self.input_dim}")

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
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate'),
            weight_decay=config.get('weight_decay')
        )

    def load_model(self,model_path):
        checkpoint = torch.load(os.path.join(model_path), map_location=self.device)
        print(f'loading model from {model_path}')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(self, loader):
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
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, accuracy, precision, recall, f1, auc, all_labels, all_preds

if __name__ == "__main__":

    test_dict = {}

    for test_year in [2017, 2018, 2019, 2020, 2021, 2022]:

        use_permission = True
        num_layers = 5
        batch_size = 256
        hidden_dim = 256


        model_path = f'../../Output/RQ1_Outputs/RQ1_ExA_permission:{use_permission}_2010-2016_layers:{num_layers}_epochs:100_batch:{batch_size}_hiddenDim:{hidden_dim}_residual:True_lr_0.001/best_validate_f1_model.pth'

        logging.info(f"use_permission: {use_permission}, num_layers: {num_layers}, batch_size: {batch_size}, hidden_dim: {hidden_dim}")
        config = {
            'data_dir': '../../Dataset/RQ1',
            'from_year': test_year,
            'to_year': test_year,
            'gpu_id': 1,
            'use_residual': True,
            'use_permission': use_permission,
            'use_attention': True,
            'batch_size': batch_size,
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': 2,
            'model_type': 'hgnn',
            'pooling': 'concat',
            'scheduler': 'plateau',
            'seed': 42,
            'num_workers': 24,
        }
        
        
        output_dir = f'../../Output/RQ1_ConceptDrift_Outputs/RQ1_ExA_permission:{config["use_permission"]}_layers:{config["num_layers"]}_epochs:{config["epochs"]}_batch:{config["batch_size"]}_hiddenDim:{config["hidden_dim"]}_residual:{config["use_residual"]}_lr_{config["learning_rate"]}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        tester = TestModel(config, output_dir)
        tester.load_model(model_path)
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, _, _ = tester.evaluate(tester.test_loader)
        print(f"{test_year} loss: {test_loss}, accuracy: {test_acc}, precision: {test_precision}, recall: {test_recall}, f1: {test_f1}, auc: {test_auc}")

        test_dict[test_year] = {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
        }
    print(test_dict)
    with open(f'{output_dir}/test_dict.json', 'w') as f:
        json.dump(test_dict, f)
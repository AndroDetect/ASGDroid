#!/usr/bin/env python3
"""
Use CodeBERT to embed Android APIs
"""

import torch
import numpy as np
import json
from typing import List
from transformers import RobertaTokenizer, RobertaModel
import warnings
warnings.filterwarnings("ignore")

class CodeBERTAPIEmbedder:
    """
    Class for embedding APIs using CodeBERT
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize CodeBERT API encoder
        
        Args:
            device: Device type ("auto", "cpu", "cuda")
        """
        self.model_name = "microsoft/codebert-base"
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
        print(f"Loading CodeBERT model...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"CodeBERT loaded successfully!")
        print(f"Embedding dimension: {self.model.config.hidden_size}")
    
    def preprocess_api(self, api: str) -> str:
        """
        Preprocess API string
        
        Args:
            api: Original API string
            
        Returns:
            Preprocessed API string
        """
        # android.telephony.TelephonyManager: java.lang.String getDeviceId()
        api = api.strip()

        return api
    
    def encode_api(self, api: str, max_length: int = 128) -> np.ndarray:
        """
        Encode a single API
        
        Args:
            api: API string
            max_length: Maximum sequence length
            
        Returns:
            768-dimensional vector representation of the API
        """
        # Preprocess API
        processed_api = self.preprocess_api(api)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_api,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]
            
            # Average pooling (ignore padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            summed = masked_hidden.sum(dim=1)
            lengths = attention_mask.sum(dim=1)
            pooled = summed / lengths  # [1, 768]
        
        return pooled.cpu().numpy().flatten()
    
    def encode_batch(self, api_list: List[str], max_length: int = 128, 
                    batch_size: int = 32) -> np.ndarray:
        """
        Batch encode API list
        
        Args:
            api_list: List of APIs
            max_length: Maximum sequence length
            batch_size: Batch size
            
        Returns:
            API vector matrix with shape (num_apis, 768)
        """
        all_embeddings = []
        
        print(f"Starting batch encoding of {len(api_list)} APIs...")
        
        for i in range(0, len(api_list), batch_size):
            batch_apis = api_list[i:i + batch_size]
            
            # Preprocess batch APIs
            processed_apis = [self.preprocess_api(api) for api in batch_apis]
            # Tokenize batch
            inputs = self.tokenizer(
                processed_apis,
                return_tensors="pt",
                max_length=max_length,
                padding=True,
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Batch encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Average pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                summed = masked_hidden.sum(dim=1)
                lengths = attention_mask.sum(dim=1)
                pooled = summed / lengths
            
            all_embeddings.append(pooled.cpu().numpy())
            
            # Progress indicator
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(api_list):
                print(f"Processed: {min(i + batch_size, len(api_list))}/{len(api_list)}")
        
        return np.vstack(all_embeddings)
    
    def save_embeddings(self, embeddings: np.ndarray, api_list: List[str], 
                       output_file: str) -> None:
        """
        Save embedding results
        
        Args:
            embeddings: Embedding matrix with shape (n_apis, embedding_dim)
            api_list: List of APIs corresponding to rows in embeddings
            output_file: Output file path
        """
        # Create API to index mapping for fast lookup
        api_to_index = {api: i for i, api in enumerate(api_list)}
        
        np.savez_compressed(
            output_file,
            embeddings=embeddings,
            api_list=api_list,
            api_to_index=api_to_index,  # Save API to index mapping instead of API to vector mapping
            model_name=self.model_name
        )
        print(f"Embeddings saved to: {output_file}")
        print(f"Saved embeddings for {len(api_list)} APIs")
    
    @staticmethod
    def load_embeddings(file_path: str) -> tuple:
        """
        Load saved embeddings (compatible with old and new versions)
        
        Args:
            file_path: File path
            
        Returns:
            (embeddings, api_list, model_name, api_to_index)
        """
        data = np.load(file_path, allow_pickle=True)
        embeddings = data['embeddings']
        api_list = data['api_list'].tolist()  # Ensure it's a Python list
        model_name = str(data['model_name'])

        api_to_index = data['api_to_index'].item()  # Use .item() to handle 0-dimensional arrays
            

        
        return embeddings, api_list, model_name, api_to_index


def main():
    """Main function: demonstrate CodeBERT API embedding"""
    
    print("=== CodeBERT API Embedder ===\n")
    
    # 1. Initialize encoder
    embedder = CodeBERTAPIEmbedder(device="auto")
    
    # 2. Load API data
    try:
        with open("../input/sensitive_apis/sensitive_apis.json", 'r', encoding='utf-8') as f:
            sensitive_api_list = json.load(f)
        print(f"\nLoaded {len(sensitive_api_list)} APIs")

        with open("../input/sensitive_apis_func/sensitive_apis_in_func.json", 'r', encoding='utf-8') as f:
            sensitive_func_api_list = json.load(f)
        print(f"\nLoaded {len(sensitive_func_api_list)} APIs")

        with open("../input/sources_sinks_apis/sources_sinks_apis_selected.json", 'r', encoding='utf-8') as f:
            taint_api_list = json.load(f)
        print(f"\nLoaded {len(taint_api_list)} APIs")

        api_list = sensitive_api_list + sensitive_func_api_list + taint_api_list
        print(f"\nLoaded {len(api_list)} APIs")
    except FileNotFoundError:
        print("\nAPI file not found")
        api_list = None
    
    # 3. Single API encoding example
    print(f"\n=== Single API Encoding Example ===")
    test_api = api_list[0]
    print(f"Test API: {test_api}")
    
    embedding = embedder.encode_api(test_api)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    
    # 4. Batch encoding
    print(f"\n=== Batch Encoding ===")
    embeddings = embedder.encode_batch(
        api_list, 
        max_length=128,
        batch_size=256
    )
    
    print(f"\nEncoding completed!")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of APIs: {embeddings.shape[0]}")
    
    # 5. Save embeddings
    output_file = "codebert_api_embeddings.npz"
    embedder.save_embeddings(embeddings, api_list, output_file)

    return embeddings, api_list

def load_embeddings(file_path: str) -> tuple:
        """
        Load saved embeddings (global function version, compatible with old code)
        
        Args:
            file_path: File path
            
        Returns:
            (embeddings, api_list, model_name, api_to_index)
        """
        return CodeBERTAPIEmbedder.load_embeddings(file_path)



if __name__ == "__main__":
    
    embeddings, api_list = main()
    
    print(f"\nUsage:")
    print(f"# Load saved embeddings")
    print(f"embeddings, api_list, model_name, api_to_index = CodeBERTAPIEmbedder.load_embeddings('codebert_api_embeddings.npz')")
    print(f"")
    print(f"# Encode new API")
    print(f"embedder = CodeBERTAPIEmbedder()")
    print(f"new_embedding = embedder.encode_api('android.app.Activity.finish()')") 
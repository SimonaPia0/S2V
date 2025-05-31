import argparse
import torch
from torch_geometric.loader import DataLoader
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import random
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import random_split
from pathlib import Path

# Import your custom modules
from source.loadData import GraphDataset 
from source.model import GraphClassifier, Graph_Net 

def introduce_label_noise(dataset, noise_ratio, num_classes):
    if noise_ratio == 0:
        print("Noise ratio is 0, no label noise introduced.")
        return dataset

    noisy_dataset = []
    num_noisy_samples = int(len(dataset) * noise_ratio)
    indices_to_corrupt = random.sample(range(len(dataset)), num_noisy_samples)

    print(f"Introducing {noise_ratio*100:.2f}% symmetric noise to {len(dataset)} samples. Corrupting {num_noisy_samples} labels...")

    for i, data in enumerate(dataset):
        if i in indices_to_corrupt:
            if hasattr(data, 'y') and data.y is not None and data.y.numel() == 1:
                original_label = data.y.item()
                # Scegli una nuova etichetta casuale diversa dall'originale
                new_label = random.randint(0, num_classes - 1)
                while new_label == original_label:
                    new_label = random.randint(0, num_classes - 1)
                data.y = torch.tensor([new_label], dtype=torch.long)
            else:
                print(f"Warning: Attempted to corrupt label for graph {i} but 'y' attribute is missing or invalid.")
        noisy_dataset.append(data)
    
    print("Label noise introduction complete.")
    return noisy_dataset


def main():
    parser = argparse.ArgumentParser(description='Graph Classification with PyTorch Geometric')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset (.json.gz)')
    parser.add_argument('--test_path', type=str, default=None, help='Path to the test dataset (.json.gz), optional for submission generation')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='Ratio of labels to corrupt symmetrically (0.0 to 1.0)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in GNN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help='Ratio of training data to use for validation')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of target classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Epsilon for label smoothing (0.0 for no smoothing)')
    parser.add_argument('--loss_type', type=str, default='cross_entropy', choices=['cross_entropy', 'label_smoothing', 'gce'], help='Type of loss function to use')
    parser.add_argument('--gce_q', type=float, default=0.7, help='Parameter q for Generalized Cross Entropy Loss (0 to 1)')
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Creazione delle cartelle richieste ---
    checkpoints_dir = Path('checkpoints')
    submission_dir = Path('submission')
    logs_dir = Path('logs')

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Estrai il foldername (A, B, C, D) dal percorso del dataset di training
    try:
        train_path_obj = Path(args.train_path)
        folder_name = train_path_obj.parent.name
        
        if len(folder_name) != 1 or not folder_name.isalpha():
            folder_name = "default" 
            print(f"Warning: Could not extract single letter folder name from {args.train_path}. Using 'default'.")
    except Exception as e: 
        folder_name = "default"
        print(f"Warning: Could not extract folder name from {args.train_path} due to an error: {e}. Using 'default'.")

    # --- Data Loading ---
    print("Loading training data...")
    train_dataset = GraphDataset(is_test_set=False)
    train_data_list = train_dataset.loadGraphs(args.train_path)
    
    if not train_data_list:
        print("Error: No training graphs loaded. Exiting.")
        return

    # Determine num_node_features dynamically from the first loaded graph
    if train_data_list and hasattr(train_data_list[0], 'x') and train_data_list[0].x is not None:
        num_node_features = train_data_list[0].x.shape[1]
        print(f"Detected num_node_features: {num_node_features}")
    else:
        print("Warning: Could not determine num_node_features. Defaulting to 10 (expected for PPA features).")
        num_node_features = 10 

    # Determine edge_channels dynamically from the first loaded graph
    edge_channels = 0
    if train_data_list and hasattr(train_data_list[0], 'edge_attr') and train_data_list[0].edge_attr is not None:
        edge_channels = train_data_list[0].edge_attr.shape[1]
        print(f"Detected edge_attr_dim: {edge_channels}")
    else:
        print("No edge_attr found or they were set to None. Model will not use edge features.")

    # Introduce noise to labels (only for training data)
    train_data_list_noisy = introduce_label_noise(train_data_list, args.noise_ratio, args.num_classes)
    
    # Split training data into training and validation
    num_train_samples = len(train_data_list_noisy)
    num_val_samples = int(num_train_samples * args.val_split_ratio)
    num_actual_train_samples = num_train_samples - num_val_samples

    if num_actual_train_samples <= 0:
        print(f"Error: Not enough samples for training after validation split. Total: {num_train_samples}, Val: {num_val_samples}. Exiting.")
        return

    train_subset, val_subset = random_split(
        train_data_list_noisy, 
        [num_actual_train_samples, num_val_samples],
        generator=torch.Generator().manual_seed(args.seed) 
    )
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")
    print(f"Batch size: {args.batch_size}")

    # --- Model and Trainer Setup ---
    model = GraphClassifier(
        num_node_features=num_node_features,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=0.5, 
        learning_rate=args.learning_rate,
        edge_channels=edge_channels,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type, 
        gce_q=args.gce_q 
    )

    # Logger: Usa la sottocartella specifica per il dataset
    logger = CSVLogger(save_dir=logs_dir, name=folder_name)

    # Checkpoint Callback: Salva nella sottocartella 'checkpoints/<foldername>'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir / folder_name, 
        filename=f'model_{folder_name}epoch{{epoch:02d}}',
        monitor='val_f1_epoch', 
        mode='max',            
        save_top_k=5,          
        every_n_epochs=1,      
        verbose=True
    )

    # Early Stopping Callback
    early_stopping_callback = EarlyStopping(
        monitor='val_f1_epoch',
        patience=20,            
        mode='max',
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator='auto', 
        devices='auto'
    )

    # --- Training ---
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining finished. Best model saved at: {checkpoint_callback.best_model_path}")

    # --- Generate Predictions for Test Set (if provided) ---
    if args.test_path:
        print("\nLoading test data for predictions...")
        test_dataset = GraphDataset(is_test_set=True)
        test_data_list = test_dataset.loadGraphs(args.test_path)

        if test_data_list:
            test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            # Carica il miglior modello basato sul validation F1 score
            best_model = GraphClassifier.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                num_node_features=num_node_features,
                num_classes=args.num_classes,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout_rate=0.5,
                learning_rate=args.learning_rate,
                edge_channels=edge_channels,
                label_smoothing=args.label_smoothing,
                loss_type=args.loss_type, 
                gce_q=args.gce_q 
            )
            # Mettere il modello in modalità eval
            best_model.eval()
            best_model.freeze()

            print("Generating predictions on the test set...")
            predicted_labels = []
            numeric_graph_ids = [] 
            
            # Estrai le predizioni dal modello e convertile in label singole
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Collecting predictions"):
                    edge_attr = getattr(batch, 'edge_attr', None)
                    out = best_model(batch.x, batch.edge_index, batch.batch, edge_attr) # Chiamata al metodo forward
                    
                    predicted_class = torch.argmax(F.softmax(out, dim=1), dim=1) 
                    predicted_labels.extend(predicted_class.cpu().numpy())
                    
                    if hasattr(batch, 'graph_id'):
                        for gid in batch.graph_id:
                            try:
                                numeric_id = int(gid)
                            except ValueError:
                                if isinstance(gid, str) and gid.startswith("graph_"):
                                    try:
                                        numeric_id = int(gid.split('_')[-1])
                                    except ValueError:
                                        print(f"Warning: Could not parse numeric ID from '{gid}'. Using 0.")
                                        numeric_id = 0 
                                else:
                                    print(f"Warning: Unexpected graph_id format '{gid}'. Using 0.")
                                    numeric_id = 0 
                            numeric_graph_ids.append(numeric_id)
                    else: 
                        current_batch_ids = list(range(len(numeric_graph_ids), len(numeric_graph_ids) + batch.num_graphs))
                        numeric_graph_ids.extend(current_batch_ids)

            # Crea il DataFrame per la sottomissione con 'id' e 'pred'
            submission_df = pd.DataFrame({
                'id': numeric_graph_ids, 
                'pred': predicted_labels
            })
            
            # Salva il file di sottomissione con il nome corretto
            submission_path = submission_dir / f'testset_{folder_name}.csv'
            submission_df.to_csv(submission_path, index=False)
            print(f"Submission file created at: {submission_path}")
        else:
            print("No test graphs loaded. Skipping prediction generation.")
    else:
        print("No test path provided. Skipping prediction generation.")


    # --- Save Training Summary ---
    summary_path = Path(logger.log_dir) / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Training epochs: {args.epochs}\n")
        f.write(f"Training batch size: {args.batch_size}\n")
        f.write(f"Training samples: {len(train_subset)}\n")
        f.write(f"Validation samples: {len(val_subset)}\n")
        f.write(f"Noise ratio applied: {args.noise_ratio}\n")
        f.write(f"Label smoothing epsilon: {args.label_smoothing}\n")
        f.write(f"Loss type: {args.loss_type}\n")
        f.write(f"Best model path: {checkpoint_callback.best_model_path}\n")

if __name__ == '__main__':
    main()
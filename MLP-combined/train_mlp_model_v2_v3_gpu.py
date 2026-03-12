"""
Multi-Layer Perceptron (MLP) Model for Ethereum Gas Usage Prediction
Using Combined V2 and V3 Transaction Datasets

Author: Shakil Ahmed - ML researcher
Email: ahmedmshakil1@gmail.com

This script trains an MLP neural network model to predict Ethereum gas usage
with proper data leakage handling and 5-Fold Cross Validation for research publication.

*** GPU ACCELERATED VERSION using PyTorch ***

Objective: Predict GasUsed based on transaction features
- Compare model performance WITH and WITHOUT data leakage (CumulativeGasUsed)
- Use K-Fold Cross Validation for robust performance estimation
- Generate comprehensive visualizations and analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports for GPU support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# GPU/DEVICE CONFIGURATION
# ============================================================================
def get_device():
    """Detect and return the best available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🚀 GPU DETECTED: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        return device
    else:
        print("\n⚠️  No GPU detected, using CPU")
        print("   For GPU support, install PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return torch.device('cpu')

DEVICE = get_device()

# Create output directory
os.makedirs('MLP-combined', exist_ok=True)

def print_separator():
    """Print a visual separator line"""
    print("\n" + "="*80 + "\n")

# ============================================================================
# PYTORCH MLP MODEL DEFINITION
# ============================================================================
class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for Regression using PyTorch
    Supports GPU acceleration
    """
    def __init__(self, input_size, hidden_layers=(256, 128, 64, 32), dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization for faster training
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
            self.counter = 0

def train_pytorch_mlp(model, train_loader, val_loader, criterion, optimizer, 
                      scheduler, epochs, device, early_stopping=None):
    """
    Train PyTorch MLP model with GPU support
    
    Returns:
        dict: Training history with loss curves
    """
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if early_stopping:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return history

# Start timing
start_time = time.time()

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
print("="*80)
print("MULTI-LAYER PERCEPTRON (MLP) MODEL TRAINING - GPU ACCELERATED")
print("Ethereum Gas Usage Prediction - Combined V2+V3 Dataset")
print("="*80)

print("\nAuthor: Shakil Ahmed - ML researcher")
print("Email: ahmedmshakil1@gmail.com")
print("\nLoading V2 and V3 datasets...")

# Load V2 dataset
df_v2 = pd.read_csv('../Ethereum_V2_Transactions.csv')
df_v2['Dataset'] = 'V2'

# Load V3 dataset  
df_v3 = pd.read_csv('../Ethereum_V3_Transactions.csv')
df_v3['Dataset'] = 'V3'

# Combine datasets
df = pd.concat([df_v2, df_v3], ignore_index=True)

# Display basic information about the combined dataset
print(f"\nCombined Dataset Info:")
print(f"V2 Transactions: {len(df_v2):,}")
print(f"V3 Transactions: {len(df_v3):,}")
print(f"Total Transactions: {len(df):,}")
print("\nDataset Info:")
print(df.info())
print("\nSample of data:")
print(df.head())

# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

print("\nPreprocessing data...")

# Convert timestamps to datetime and extract temporal features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year

print("\nTemporal features extracted:")
print("  - Hour: Hour of the day (0-23)")
print("  - Day: Day of the month (1-31)")
print("  - Month: Month of the year (1-12)")
print("  - Year: Year of the transaction")

# ============================================================================
# SECTION 3: FEATURE DEFINITION - DATA LEAKAGE HANDLING
# ============================================================================
print("\n" + "="*80)
print("DATA LEAKAGE ANALYSIS")
print("="*80)

# Define features WITH potential data leakage
features_with_leakage = [
    'BlockNumber', 'Nonce', 'TransactionIndex', 'Value', 'gas',
    'GasPrice', 'CumulativeGasUsed',  # This causes data leakage!
    'Confirmations', 'isError', 'Hour', 'Day', 'Month', 'Year'
]

# Define features WITHOUT data leakage (for legitimate model)
features_without_leakage = [
    'BlockNumber', 'Nonce', 'TransactionIndex', 'Value', 'gas',
    'GasPrice',  # Removed CumulativeGasUsed to prevent leakage
    'Confirmations', 'isError', 'Hour', 'Day', 'Month', 'Year'
]

# Target variable
target = 'GasUsed'

print("\nFeatures WITH Leakage (13 features):")
for i, feat in enumerate(features_with_leakage, 1):
    leakage_marker = " [LEAKAGE!]" if feat == 'CumulativeGasUsed' else ""
    print(f"  {i}. {feat}{leakage_marker}")

print("\nFeatures WITHOUT Leakage (12 features):")
for i, feat in enumerate(features_without_leakage, 1):
    print(f"  {i}. {feat}")

print(f"\nTarget Variable: {target}")

print("\n[!] Data Leakage Explanation:")
print("    CumulativeGasUsed is the running total of gas used in a block.")
print("    This value is only known AFTER the transaction is executed.")
print("    Using it for prediction would be 'cheating' as this information")
print("    wouldn't be available at the time of making predictions.")

# ============================================================================
# SECTION 4: MLP MODEL TRAINING FUNCTION (GPU-ACCELERATED)
# ============================================================================

def train_mlp_model_with_cv(X, y, features_name, save_prefix, n_folds=5):
    """
    Train Multi-Layer Perceptron model with K-Fold Cross Validation using GPU
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    features_name : str
        Description of the feature set being used
    save_prefix : str
        Prefix for saving output files
    n_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict : Dictionary containing model, metrics, and results
    """
    print(f"\n{'='*80}")
    print(f"Training MLP Neural Network with {features_name}")
    print(f"Features: {len(X.columns)} features")
    print(f"Cross Validation: {n_folds} folds")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}")
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Storage for CV results
    cv_results = []
    fold_predictions = []
    fold_training_history = []
    
    # Prepare data for final model (full dataset split)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nRunning 5-Fold Cross Validation...")
    print("-" * 60)
    
    # MLP configuration for CV
    cv_config = {
        'hidden_layers': (128, 64, 32),
        'dropout_rate': 0.2,
        'batch_size': 512,
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 10
    }
    
    print("\nMLP Architecture for Cross-Validation (GPU):")
    print(f"  - Hidden Layers: {cv_config['hidden_layers']}")
    print(f"  - Dropout Rate: {cv_config['dropout_rate']}")
    print(f"  - Batch Size: {cv_config['batch_size']}")
    print(f"  - Learning Rate: {cv_config['learning_rate']}")
    print(f"  - Max Epochs: {cv_config['epochs']}")
    print(f"  - Early Stopping Patience: {cv_config['patience']}")
    
    # Perform K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
        fold_start = time.time()
        print(f"\nFold {fold}/{n_folds}:")
        print("-" * 20)
        
        # Split data for this fold
        X_train_fold = X_train_full.iloc[train_idx].values
        X_val_fold = X_train_full.iloc[val_idx].values
        y_train_fold = y_train_full.iloc[train_idx].values
        y_val_fold = y_train_full.iloc[val_idx].values
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Convert to PyTorch tensors and move to GPU
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train_fold).to(DEVICE)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(DEVICE)
        y_val_tensor = torch.FloatTensor(y_val_fold).to(DEVICE)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=cv_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cv_config['batch_size'], shuffle=False)
        
        # Create model
        model = MLPRegressor(
            input_size=X_train_scaled.shape[1],
            hidden_layers=cv_config['hidden_layers'],
            dropout_rate=cv_config['dropout_rate']
        ).to(DEVICE)
        
        # Loss, optimizer, scheduler
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cv_config['learning_rate'], weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=cv_config['patience'])
        
        # Train model
        history = train_pytorch_mlp(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, cv_config['epochs'], DEVICE, early_stopping
        )
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            y_pred_fold = model(X_val_tensor).cpu().numpy()
        
        # Calculate metrics
        mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
        rmse_fold = np.sqrt(mse_fold)
        r2_fold = r2_score(y_val_fold, y_pred_fold)
        mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
        
        fold_time = time.time() - fold_start
        actual_epochs = len(history['train_loss'])
        
        # Store results
        fold_result = {
            'Fold': fold,
            'R2_Score': r2_fold,
            'RMSE': rmse_fold,
            'MSE': mse_fold,
            'MAE': mae_fold,
            'Train_Size': len(X_train_fold),
            'Val_Size': len(X_val_fold),
            'Iterations': actual_epochs,
            'Training_Time_Seconds': fold_time
        }
        cv_results.append(fold_result)
        
        # Store predictions
        fold_pred = pd.DataFrame({
            'Fold': fold,
            'Actual': y_val_fold,
            'Predicted': y_pred_fold,
            'Error': y_val_fold - y_pred_fold,
            'Abs_Error': np.abs(y_val_fold - y_pred_fold)
        })
        fold_predictions.append(fold_pred)
        
        # Store training history
        fold_training_history.append({
            'Fold': fold,
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        })
        
        # Print fold results
        print(f"  R2 Score: {r2_fold:.4f}")
        print(f"  RMSE: {rmse_fold:.2f}")
        print(f"  MAE: {mae_fold:.2f}")
        print(f"  Epochs: {actual_epochs}")
        print(f"  Train samples: {len(X_train_fold):,}")
        print(f"  Val samples: {len(X_val_fold):,}")
        print(f"  Time: {fold_time:.2f}s")
        
        # Clear GPU memory
        del model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create CV results DataFrame
    cv_df = pd.DataFrame(cv_results)
    
    # Calculate cross-validation statistics
    cv_stats = {
        'Mean_R2': cv_df['R2_Score'].mean(),
        'Std_R2': cv_df['R2_Score'].std(),
        'Mean_RMSE': cv_df['RMSE'].mean(),
        'Std_RMSE': cv_df['RMSE'].std(),
        'Mean_MAE': cv_df['MAE'].mean(),
        'Std_MAE': cv_df['MAE'].std(),
        'Min_R2': cv_df['R2_Score'].min(),
        'Max_R2': cv_df['R2_Score'].max(),
        'Min_RMSE': cv_df['RMSE'].min(),
        'Max_RMSE': cv_df['RMSE'].max(),
        'Mean_Iterations': cv_df['Iterations'].mean(),
        'Total_CV_Time': cv_df['Training_Time_Seconds'].sum()
    }
    
    print(f"\n{'='*60}")
    print("CROSS VALIDATION SUMMARY:")
    print(f"{'='*60}")
    print(f"Mean R2 Score: {cv_stats['Mean_R2']:.4f} (+/- {cv_stats['Std_R2']:.4f})")
    print(f"Mean RMSE: {cv_stats['Mean_RMSE']:.2f} (+/- {cv_stats['Std_RMSE']:.2f})")
    print(f"Mean MAE: {cv_stats['Mean_MAE']:.2f} (+/- {cv_stats['Std_MAE']:.2f})")
    print(f"R2 Range: [{cv_stats['Min_R2']:.4f}, {cv_stats['Max_R2']:.4f}]")
    print(f"RMSE Range: [{cv_stats['Min_RMSE']:.2f}, {cv_stats['Max_RMSE']:.2f}]")
    print(f"Mean Epochs: {cv_stats['Mean_Iterations']:.0f}")
    print(f"Total CV Time: {cv_stats['Total_CV_Time']:.2f}s")
    
    # Save CV results
    cv_df.to_csv(f'MLP-combined/{save_prefix}_cv_results.csv', index=False)
    
    # Combine and save all fold predictions
    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    all_predictions.to_csv(f'MLP-combined/{save_prefix}_cv_predictions.csv', index=False)
    
    # Plot CV loss curves
    plt.figure(figsize=(12, 6))
    for history in fold_training_history:
        plt.plot(history['val_loss'], label=f"Fold {history['Fold']}", alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Training Loss Curves - Cross Validation ({features_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_cv_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # TRAIN FINAL PRODUCTION MODEL
    # ========================================================================
    print(f"\n{'='*60}")
    print("TRAINING FINAL PRODUCTION MODEL (GPU)")
    print(f"{'='*60}")
    print(f"  - Using ALL {len(X_train_full):,} training samples")
    print(f"  - Enhanced parameters for maximum performance")
    print(f"  - Device: {DEVICE}")
    
    # Scale data for final model
    scaler_final = StandardScaler()
    X_train_scaled_final = scaler_final.fit_transform(X_train_full.values)
    X_test_scaled_final = scaler_final.transform(X_test_full.values)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled_final).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_full.values).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test_scaled_final).to(DEVICE)
    y_test_np = y_test_full.values
    
    # Production model configuration
    prod_config = {
        'hidden_layers': (256, 128, 64, 32),
        'dropout_rate': 0.15,
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 200,
        'patience': 20
    }
    
    best_params = prod_config.copy()
    
    print(f"\nProduction MLP Architecture (GPU):")
    print(f"  - Hidden Layers: {prod_config['hidden_layers']}")
    print(f"  - Total Hidden Neurons: {sum(prod_config['hidden_layers'])}")
    print(f"  - Dropout Rate: {prod_config['dropout_rate']}")
    print(f"  - Batch Size: {prod_config['batch_size']}")
    print(f"  - Learning Rate: {prod_config['learning_rate']}")
    print(f"  - Max Epochs: {prod_config['epochs']}")
    
    # Create DataLoaders
    # Use 10% of training data for validation during training
    val_size = int(0.1 * len(X_train_tensor))
    train_size = len(X_train_tensor) - val_size
    
    train_dataset = TensorDataset(X_train_tensor[:train_size], y_train_tensor[:train_size])
    val_dataset = TensorDataset(X_train_tensor[train_size:], y_train_tensor[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=prod_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=prod_config['batch_size'], shuffle=False)
    
    # Create final model
    model_final = MLPRegressor(
        input_size=X_train_scaled_final.shape[1],
        hidden_layers=prod_config['hidden_layers'],
        dropout_rate=prod_config['dropout_rate']
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model_final.parameters())
    trainable_params = sum(p.numel() for p in model_final.parameters() if p.requires_grad)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_final.parameters(), lr=prod_config['learning_rate'], weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=prod_config['patience'])
    
    # Train final model
    print(f"\nTraining final production MLP model on GPU...")
    final_train_start = time.time()
    
    final_history = train_pytorch_mlp(
        model_final, train_loader, val_loader, criterion, optimizer,
        scheduler, prod_config['epochs'], DEVICE, early_stopping
    )
    
    final_train_time = time.time() - final_train_start
    
    # Final predictions
    model_final.eval()
    with torch.no_grad():
        y_pred_final = model_final(X_test_tensor).cpu().numpy()
    
    # Final metrics
    mse_final = mean_squared_error(y_test_np, y_pred_final)
    rmse_final = np.sqrt(mse_final)
    r2_final = r2_score(y_test_np, y_pred_final)
    mae_final = mean_absolute_error(y_test_np, y_pred_final)
    
    print(f"\n{'='*60}")
    print("FINAL MODEL PERFORMANCE ON TEST SET:")
    print(f"{'='*60}")
    print(f"  R2 Score: {r2_final:.4f}")
    print(f"  RMSE: {rmse_final:.2f}")
    print(f"  MAE: {mae_final:.2f}")
    print(f"  Epochs: {len(final_history['train_loss'])}")
    print(f"  Training Time: {final_train_time:.2f}s")
    print(f"  Test Samples: {len(y_test_np):,}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    # 1. Final model loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(final_history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    plt.plot(final_history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Final Model Training Loss Curve ({features_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_final_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Actual vs Predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_np, y_pred_final, alpha=0.5, s=20)
    plt.plot([y_test_np.min(), y_test_np.max()], 
             [y_test_np.min(), y_test_np.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Gas Used')
    plt.ylabel('Predicted Gas Used')
    plt.title(f'Final Model: Actual vs Predicted Gas Used\n({features_name})\nR² = {r2_final:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_final_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Plot
    residuals = y_test_np - y_pred_final
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_final, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({features_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_final_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Residual Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution ({features_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_final_residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature weights visualization (from first layer)
    plt.figure(figsize=(12, 8))
    first_layer_weights = model_final.network[0].weight.data.cpu().numpy()
    input_weights = np.abs(first_layer_weights).mean(axis=0)
    
    weight_importance = pd.DataFrame({
        'feature': X.columns,
        'weight_importance': input_weights
    }).sort_values('weight_importance', ascending=False)
    
    plt.barh(range(len(weight_importance)), weight_importance['weight_importance'])
    plt.yticks(range(len(weight_importance)), weight_importance['feature'])
    plt.xlabel('Mean Absolute Weight (First Hidden Layer)')
    plt.title(f'Feature Weight Importance ({features_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_final_weight_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Vertical version
    plt.figure(figsize=(12, 8))
    top_features = weight_importance.head(10)
    plt.bar(top_features['feature'], top_features['weight_importance'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absolute Weight')
    plt.title(f'Top 10 Feature Weight Importance ({features_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'MLP-combined/{save_prefix}_final_weight_importance_vertical.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the final model and scaler
    torch.save({
        'model_state_dict': model_final.state_dict(),
        'config': prod_config,
        'input_size': X_train_scaled_final.shape[1]
    }, f'MLP-combined/{save_prefix}_final_model.pt')
    joblib.dump(scaler_final, f'MLP-combined/{save_prefix}_final_scaler.joblib')
    
    # Save final predictions
    predictions_final_df = pd.DataFrame({
        'Actual': y_test_np,
        'Predicted': y_pred_final,
        'Residuals': residuals,
        'Abs_Error': np.abs(residuals)
    })
    predictions_final_df.to_csv(f'MLP-combined/{save_prefix}_final_predictions.csv', index=False)
    
    # Save model architecture summary
    architecture_summary = {
        'Input_Features': len(X.columns),
        'Hidden_Layers': str(prod_config['hidden_layers']),
        'Total_Hidden_Neurons': sum(prod_config['hidden_layers']),
        'Total_Parameters': total_params,
        'Trainable_Parameters': trainable_params,
        'Dropout_Rate': prod_config['dropout_rate'],
        'Learning_Rate_Init': prod_config['learning_rate'],
        'Batch_Size': prod_config['batch_size'],
        'Max_Epochs': prod_config['epochs'],
        'Actual_Epochs': len(final_history['train_loss']),
        'Training_Time_Seconds': final_train_time,
        'Device': str(DEVICE)
    }
    
    pd.DataFrame([architecture_summary]).to_csv(
        f'MLP-combined/{save_prefix}_architecture_summary.csv', index=False
    )
    
    # Clear GPU memory
    del model_final, X_train_tensor, y_train_tensor, X_test_tensor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'r2': r2_final,
        'rmse': rmse_final,
        'mae': mae_final,
        'y_test': y_test_full,
        'y_pred': y_pred_final,
        'weight_importance': weight_importance,
        'cv_results': cv_df,
        'cv_stats': cv_stats,
        'all_predictions': all_predictions,
        'best_params': best_params,
        'final_loss_curve': final_history['val_loss']
    }

# ============================================================================
# SECTION 5: TRAIN MODELS WITH AND WITHOUT DATA LEAKAGE
# ============================================================================

# Prepare target variable
y = df[target]

print_separator()

# Train model WITH leakage (including CumulativeGasUsed)
print("[WARNING] Training MLP model WITH potential data leakage...")
print("[WARNING] This model uses CumulativeGasUsed which is NOT available at prediction time!")
X_with_leakage = df[features_with_leakage]
results_with_leakage = train_mlp_model_with_cv(
    X_with_leakage, y, 
    "Features WITH Leakage (includes CumulativeGasUsed)", 
    "mlp_with_leakage"
)

print_separator()

# Train model WITHOUT leakage (excluding CumulativeGasUsed)
print("[CLEAN] Training MLP model WITHOUT data leakage...")
print("[CLEAN] This model represents legitimate, publication-ready performance!")
X_without_leakage = df[features_without_leakage]
results_without_leakage = train_mlp_model_with_cv(
    X_without_leakage, y, 
    "Features WITHOUT Leakage (excludes CumulativeGasUsed)", 
    "mlp_without_leakage"
)

print_separator()

# ============================================================================
# SECTION 6: COMPARATIVE ANALYSIS
# ============================================================================
print("COMPARATIVE ANALYSIS: WITH vs WITHOUT Data Leakage")
print("="*80)

# Performance comparison - Final Models
print("\nFINAL MODEL PERFORMANCE:")
print("-" * 60)
print(f"[WARNING] WITH Leakage    - R2: {results_with_leakage['r2']:.4f}, RMSE: {results_with_leakage['rmse']:.2f}, MAE: {results_with_leakage['mae']:.2f}")
print(f"[CLEAN] WITHOUT Leakage - R2: {results_without_leakage['r2']:.4f}, RMSE: {results_without_leakage['rmse']:.2f}, MAE: {results_without_leakage['mae']:.2f}")
print(f"[DIFF] Difference      - R2: {results_with_leakage['r2'] - results_without_leakage['r2']:.4f}, "
      f"RMSE: {results_with_leakage['rmse'] - results_without_leakage['rmse']:.2f}")

# Cross-Validation comparison
print("\nCROSS VALIDATION PERFORMANCE:")
print("-" * 60)
print(f"[WARNING] WITH Leakage CV    - Mean R2: {results_with_leakage['cv_stats']['Mean_R2']:.4f} (+/- {results_with_leakage['cv_stats']['Std_R2']:.4f})")
print(f"[CLEAN] WITHOUT Leakage CV - Mean R2: {results_without_leakage['cv_stats']['Mean_R2']:.4f} (+/- {results_without_leakage['cv_stats']['Std_R2']:.4f})")
print(f"[WARNING] WITH Leakage CV    - Mean RMSE: {results_with_leakage['cv_stats']['Mean_RMSE']:.2f} (+/- {results_with_leakage['cv_stats']['Std_RMSE']:.2f})")
print(f"[CLEAN] WITHOUT Leakage CV - Mean RMSE: {results_without_leakage['cv_stats']['Mean_RMSE']:.2f} (+/- {results_without_leakage['cv_stats']['Std_RMSE']:.2f})")
print(f"[WARNING] WITH Leakage CV    - Mean MAE: {results_with_leakage['cv_stats']['Mean_MAE']:.2f} (+/- {results_with_leakage['cv_stats']['Std_MAE']:.2f})")
print(f"[CLEAN] WITHOUT Leakage CV - Mean MAE: {results_without_leakage['cv_stats']['Mean_MAE']:.2f} (+/- {results_without_leakage['cv_stats']['Std_MAE']:.2f})")

# Neural Network Architecture
print("\nMLP ARCHITECTURE:")
print("-" * 60)
print(f"Production Hidden Layers: {results_with_leakage['best_params']['hidden_layers']}")
print(f"Total Hidden Neurons: {sum(results_with_leakage['best_params']['hidden_layers'])}")
print(f"Dropout Rate: {results_with_leakage['best_params']['dropout_rate']}")
print(f"Device Used: {DEVICE}")

# ============================================================================
# SECTION 7: COMPREHENSIVE COMPARISON VISUALIZATIONS
# ============================================================================

# Create comprehensive comparison visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Plot 1: R2 Score by Fold
ax1 = axes[0, 0]
folds = results_with_leakage['cv_results']['Fold']
ax1.plot(folds, results_with_leakage['cv_results']['R2_Score'], 'ro-', 
         label='WITH Leakage', linewidth=2, markersize=8)
ax1.plot(folds, results_without_leakage['cv_results']['R2_Score'], 'go-', 
         label='WITHOUT Leakage', linewidth=2, markersize=8)
ax1.set_xlabel('Fold')
ax1.set_ylabel('R2 Score')
ax1.set_title('R2 Score by Fold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: RMSE by Fold
ax2 = axes[0, 1]
ax2.plot(folds, results_with_leakage['cv_results']['RMSE'], 'ro-', 
         label='WITH Leakage', linewidth=2, markersize=8)
ax2.plot(folds, results_without_leakage['cv_results']['RMSE'], 'go-', 
         label='WITHOUT Leakage', linewidth=2, markersize=8)
ax2.set_xlabel('Fold')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE by Fold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: R2 Score Distribution (Box Plot)
ax3 = axes[1, 0]
ax3.boxplot([results_with_leakage['cv_results']['R2_Score'], 
             results_without_leakage['cv_results']['R2_Score']], 
            labels=['WITH Leakage', 'WITHOUT Leakage'])
ax3.set_ylabel('R2 Score')
ax3.set_title('R2 Score Distribution')
ax3.grid(True, alpha=0.3)

# Plot 4: RMSE Distribution (Box Plot)
ax4 = axes[1, 1]
ax4.boxplot([results_with_leakage['cv_results']['RMSE'], 
             results_without_leakage['cv_results']['RMSE']], 
            labels=['WITH Leakage', 'WITHOUT Leakage'])
ax4.set_ylabel('RMSE')
ax4.set_title('RMSE Distribution')
ax4.grid(True, alpha=0.3)

# Plot 5: Distribution of Gas Used by Dataset
ax5 = axes[2, 0]
v2_mask = df['Dataset'] == 'V2'
v3_mask = df['Dataset'] == 'V3'
ax5.hist([df[v2_mask][target], df[v3_mask][target]], 
         bins=50, alpha=0.7, label=['V2', 'V3'], color=['blue', 'orange'])
ax5.set_xlabel('Gas Used')
ax5.set_ylabel('Frequency')
ax5.set_title('Distribution of Gas Used by Dataset')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Dataset Composition
ax6 = axes[2, 1]
ax6.pie([len(df_v2), len(df_v3)], 
        labels=[f'V2\n({len(df_v2):,})', f'V3\n({len(df_v3):,})'], 
        autopct='%1.1f%%', colors=['blue', 'orange'])
ax6.set_title('Dataset Composition')

plt.tight_layout()
plt.savefig('MLP-combined/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create leakage comparison analysis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Actual vs Predicted WITH leakage
ax1.scatter(results_with_leakage['y_test'], results_with_leakage['y_pred'], 
           alpha=0.6, color='red', s=20)
ax1.plot([results_with_leakage['y_test'].min(), results_with_leakage['y_test'].max()], 
         [results_with_leakage['y_test'].min(), results_with_leakage['y_test'].max()], 
         'k--', lw=2)
ax1.set_xlabel('Actual Gas Used')
ax1.set_ylabel('Predicted Gas Used')
ax1.set_title(f'WITH Leakage (MLP-GPU)\nR² = {results_with_leakage["r2"]:.4f}')
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted WITHOUT leakage
ax2.scatter(results_without_leakage['y_test'], results_without_leakage['y_pred'], 
           alpha=0.6, color='green', s=20)
ax2.plot([results_without_leakage['y_test'].min(), results_without_leakage['y_test'].max()], 
         [results_without_leakage['y_test'].min(), results_without_leakage['y_test'].max()], 
         'k--', lw=2)
ax2.set_xlabel('Actual Gas Used')
ax2.set_ylabel('Predicted Gas Used')
ax2.set_title(f'WITHOUT Leakage (MLP-GPU)\nR² = {results_without_leakage["r2"]:.4f}')
ax2.grid(True, alpha=0.3)

# Plot 3: Weight importance comparison (WITH leakage)
top_weights_with = results_with_leakage['weight_importance'].head(8)
ax3.barh(range(len(top_weights_with)), top_weights_with['weight_importance'], color='red', alpha=0.7)
ax3.set_yticks(range(len(top_weights_with)))
ax3.set_yticklabels(top_weights_with['feature'])
ax3.set_xlabel('Weight Importance')
ax3.set_title('Feature Weight Importance (WITH Leakage)')
ax3.grid(True, alpha=0.3)

# Plot 4: Weight importance comparison (WITHOUT leakage)
top_weights_without = results_without_leakage['weight_importance'].head(8)
ax4.barh(range(len(top_weights_without)), top_weights_without['weight_importance'], color='green', alpha=0.7)
ax4.set_yticks(range(len(top_weights_without)))
ax4.set_yticklabels(top_weights_without['feature'])
ax4.set_xlabel('Weight Importance')
ax4.set_title('Feature Weight Importance (WITHOUT Leakage)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('MLP-combined/leakage_comparison_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create MLP-specific visualization: Training Loss Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve WITH leakage
ax1 = axes[0]
ax1.plot(results_with_leakage['final_loss_curve'], 'r-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss - WITH Leakage')
ax1.grid(True, alpha=0.3)

# Loss curve WITHOUT leakage
ax2 = axes[1]
ax2.plot(results_without_leakage['final_loss_curve'], 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss - WITHOUT Leakage')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('MLP-combined/training_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# SECTION 8: SAVE COMPREHENSIVE RESULTS SUMMARY
# ============================================================================

# Save comprehensive results summary
comparison_df = pd.DataFrame({
    'Metric': ['R2_Score', 'RMSE', 'MAE', 'Features_Count', 
               'CV_Mean_R2', 'CV_Std_R2', 'CV_Mean_RMSE', 'CV_Std_RMSE',
               'CV_Mean_MAE', 'CV_Std_MAE'],
    'WITH_Leakage': [
        results_with_leakage['r2'],
        results_with_leakage['rmse'],
        results_with_leakage['mae'],
        len(features_with_leakage),
        results_with_leakage['cv_stats']['Mean_R2'],
        results_with_leakage['cv_stats']['Std_R2'],
        results_with_leakage['cv_stats']['Mean_RMSE'],
        results_with_leakage['cv_stats']['Std_RMSE'],
        results_with_leakage['cv_stats']['Mean_MAE'],
        results_with_leakage['cv_stats']['Std_MAE']
    ],
    'WITHOUT_Leakage': [
        results_without_leakage['r2'],
        results_without_leakage['rmse'],
        results_without_leakage['mae'],
        len(features_without_leakage),
        results_without_leakage['cv_stats']['Mean_R2'],
        results_without_leakage['cv_stats']['Std_R2'],
        results_without_leakage['cv_stats']['Mean_RMSE'],
        results_without_leakage['cv_stats']['Std_RMSE'],
        results_without_leakage['cv_stats']['Mean_MAE'],
        results_without_leakage['cv_stats']['Std_MAE']
    ],
    'Difference': [
        results_with_leakage['r2'] - results_without_leakage['r2'],
        results_with_leakage['rmse'] - results_without_leakage['rmse'],
        results_with_leakage['mae'] - results_without_leakage['mae'],
        len(features_with_leakage) - len(features_without_leakage),
        results_with_leakage['cv_stats']['Mean_R2'] - results_without_leakage['cv_stats']['Mean_R2'],
        results_with_leakage['cv_stats']['Std_R2'] - results_without_leakage['cv_stats']['Std_R2'],
        results_with_leakage['cv_stats']['Mean_RMSE'] - results_without_leakage['cv_stats']['Mean_RMSE'],
        results_with_leakage['cv_stats']['Std_RMSE'] - results_without_leakage['cv_stats']['Std_RMSE'],
        results_with_leakage['cv_stats']['Mean_MAE'] - results_without_leakage['cv_stats']['Mean_MAE'],
        results_with_leakage['cv_stats']['Std_MAE'] - results_without_leakage['cv_stats']['Std_MAE']
    ]
})
comparison_df.to_csv('MLP-combined/model_comparison_summary.csv', index=False)

# Save weight importance comparison
weight_comparison = pd.DataFrame({
    'Feature': features_without_leakage,
    'WITH_Leakage': results_with_leakage['weight_importance'].set_index('feature').loc[features_without_leakage, 'weight_importance'].values,
    'WITHOUT_Leakage': results_without_leakage['weight_importance']['weight_importance'].values
})
weight_comparison.to_csv('MLP-combined/weight_importance_comparison.csv', index=False)

# Calculate execution time
execution_time = time.time() - start_time

# ============================================================================
# SECTION 9: FINAL SUMMARY
# ============================================================================
print_separator()

print("OUTPUT FILES SAVED:")
print("-" * 60)
print("MLP-combined/")
print("   Visualizations:")
print("   - comprehensive_analysis.png - Full comparative analysis")
print("   - leakage_comparison_analysis.png - Leakage impact visualization")
print("   - training_loss_comparison.png - Training loss curves")
print("   - mlp_*_cv_loss_curves.png - Cross-validation loss curves")
print("   - mlp_*_final_loss_curve.png - Final model loss curves")
print("   - mlp_*_final_actual_vs_predicted.png - Prediction scatter plots")
print("   - mlp_*_final_residuals.png - Residual plots")
print("   - mlp_*_final_residual_distribution.png - Residual histograms")
print("   - mlp_*_final_weight_importance.png - Feature weight importance")
print("")
print("   Results:")
print("   - model_comparison_summary.csv - Performance metrics comparison")
print("   - weight_importance_comparison.csv - Feature weight comparison")
print("   - mlp_*_cv_results.csv - Cross-validation results")
print("   - mlp_*_cv_predictions.csv - CV predictions")
print("   - mlp_*_final_predictions.csv - Final model predictions")
print("   - mlp_*_architecture_summary.csv - MLP architecture details")
print("")
print("   Models:")
print("   - mlp_*_final_model.pt - Trained PyTorch MLP models")
print("   - mlp_*_final_scaler.joblib - Fitted StandardScalers")

print_separator()

print("CONCLUSION:")
print("="*80)
print(f"   - Combined V2+V3 dataset has {len(df):,} total transactions")
print(f"   - MLP WITHOUT leakage achieves {results_without_leakage['r2']:.1%} R² accuracy")
print(f"   - This represents legitimate, publication-ready performance!")
print(f"   - Data leakage impact: {abs(results_with_leakage['r2'] - results_without_leakage['r2']):.4f} R² difference")
print(f"   - CV stability: {results_without_leakage['cv_stats']['Std_R2']:.4f} R² standard deviation")
print(f"   - Neural Network Architecture: {results_without_leakage['best_params']['hidden_layers']}")
print(f"   - Device Used: {DEVICE}")
print(f"   - Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

print("\n" + "="*80)
print("MULTI-LAYER PERCEPTRON (MLP) GPU-ACCELERATED ANALYSIS COMPLETE!")
print("="*80)

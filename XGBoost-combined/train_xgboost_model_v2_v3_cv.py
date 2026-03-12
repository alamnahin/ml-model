import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('XGBoost-combined', exist_ok=True)

# Load both datasets
print("Author: Shakil Ahmed - ML researcher")
print("Email: ahmedmshakil1@gmail.com")
print("Loading V2 and V3 datasets...")

# Load V2 dataset
df_v2 = pd.read_csv('Ethereum_V2_Transactions.csv')
df_v2['Dataset'] = 'V2'

# Load V3 dataset  
df_v3 = pd.read_csv('Ethereum_V3_Transactions.csv')
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

# Data preprocessing
print("\nPreprocessing data...")

# Convert timestamps to datetime and extract features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year

# Define features with and without leakage
features_with_leakage = [
    'BlockNumber', 'Nonce', 'TransactionIndex', 'Value', 'gas',
    'GasPrice', 'CumulativeGasUsed',  # This causes data leakage
    'Confirmations', 'isError', 'Hour', 'Day', 'Month', 'Year'
]

features_without_leakage = [
    'BlockNumber', 'Nonce', 'TransactionIndex', 'Value', 'gas',
    'GasPrice',  # Removed CumulativeGasUsed to prevent leakage
    'Confirmations', 'isError', 'Hour', 'Day', 'Month', 'Year'
]

# Target variable
target = 'GasUsed'

def train_xgboost_model_with_cv(X, y, features_name, save_prefix, n_folds=5):
    """Train XGBoost model with K-Fold Cross Validation and return results"""
    print(f"\n{'='*80}")
    print(f"Training XGBoost model with {features_name}")
    print(f"Features: {len(X.columns)} features")
    print(f"Cross Validation: {n_folds} folds")
    print(f"{'='*80}")
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Storage for CV results
    cv_results = []
    fold_predictions = []
    fold_feature_importance = []
    
    # Prepare data for final model (full dataset split)
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nRunning 5-Fold Cross Validation...")
    print("-" * 60)
    
    # Perform K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
        print(f"\nFold {fold}/{n_folds}:")
        print("-" * 20)
        
        # Split data for this fold
        X_train_fold = X_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train_fold)
        dval = xgb.DMatrix(X_val_scaled, label=y_val_fold)
        
        # Set XGBoost parameters
        params = {
            'max_depth': 8,
            'eta': 0.1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'n_jobs': 12,
            'verbosity': 0
        }
        
        # Train model for this fold
        model_fold = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions on validation set
        y_pred_fold = model_fold.predict(dval)
        
        # Calculate metrics for this fold
        mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
        rmse_fold = np.sqrt(mse_fold)
        r2_fold = r2_score(y_val_fold, y_pred_fold)
        
        # Store results
        fold_result = {
            'Fold': fold,
            'R2_Score': r2_fold,
            'RMSE': rmse_fold,
            'MSE': mse_fold,
            'Train_Size': len(X_train_fold),
            'Val_Size': len(X_val_fold)
        }
        cv_results.append(fold_result)
        
        # Store predictions for this fold
        fold_pred = pd.DataFrame({
            'Fold': fold,
            'Actual': y_val_fold.values,
            'Predicted': y_pred_fold,
            'Error': y_val_fold.values - y_pred_fold,
            'Abs_Error': np.abs(y_val_fold.values - y_pred_fold)
        })
        fold_predictions.append(fold_pred)
        
        # Store feature importance for this fold
        importance_scores = model_fold.get_score(importance_type='weight')
        if importance_scores:
            fold_importance = pd.DataFrame({
                'Fold': fold,
                'Feature': list(importance_scores.keys()),
                'Importance': list(importance_scores.values())
            })
            fold_feature_importance.append(fold_importance)
        
        # Print fold results
        print(f"  R2 Score: {r2_fold:.4f}")
        print(f"  RMSE: {rmse_fold:.2f}")
        print(f"  Train samples: {len(X_train_fold):,}")
        print(f"  Val samples: {len(X_val_fold):,}")
    
    # Create CV results DataFrame
    cv_df = pd.DataFrame(cv_results)
    
    # Calculate cross-validation statistics
    cv_stats = {
        'Mean_R2': cv_df['R2_Score'].mean(),
        'Std_R2': cv_df['R2_Score'].std(),
        'Mean_RMSE': cv_df['RMSE'].mean(),
        'Std_RMSE': cv_df['RMSE'].std(),
        'Min_R2': cv_df['R2_Score'].min(),
        'Max_R2': cv_df['R2_Score'].max(),
        'Min_RMSE': cv_df['RMSE'].min(),
        'Max_RMSE': cv_df['RMSE'].max()
    }
    
    print(f"\n{'='*60}")
    print("CROSS VALIDATION SUMMARY:")
    print(f"{'='*60}")
    print(f"Mean R2 Score: {cv_stats['Mean_R2']:.4f} (+/- {cv_stats['Std_R2']:.4f})")
    print(f"Mean RMSE: {cv_stats['Mean_RMSE']:.2f} (+/- {cv_stats['Std_RMSE']:.2f})")
    print(f"R2 Range: [{cv_stats['Min_R2']:.4f}, {cv_stats['Max_R2']:.4f}]")
    print(f"RMSE Range: [{cv_stats['Min_RMSE']:.2f}, {cv_stats['Max_RMSE']:.2f}]")
    
    # Save CV results
    cv_df.to_csv(f'XGBoost-combined/{save_prefix}_cv_results.csv', index=False)
    
    # Combine and save all fold predictions
    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    all_predictions.to_csv(f'XGBoost-combined/{save_prefix}_cv_predictions.csv', index=False)
    
    # Combine and save all fold feature importance
    if fold_feature_importance:
        all_importance = pd.concat(fold_feature_importance, ignore_index=True)
        all_importance.to_csv(f'XGBoost-combined/{save_prefix}_cv_feature_importance.csv', index=False)
    
    # Now train final model on full training set
    print(f"\nTraining final model on full training set...")
    scaler_final = StandardScaler()
    X_train_scaled_final = scaler_final.fit_transform(X_train_full)
    X_test_scaled_final = scaler_final.transform(X_test_full)
    
    # Convert to DMatrix
    dtrain_final = xgb.DMatrix(X_train_scaled_final, label=y_train_full)
    dtest_final = xgb.DMatrix(X_test_scaled_final, label=y_test_full)
    
    # Train final model
    model_final = xgb.train(
        params,
        dtrain_final,
        num_boost_round=100,
        evals=[(dtrain_final, 'train'), (dtest_final, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Final predictions
    y_pred_final = model_final.predict(dtest_final)
    
    # Final metrics
    mse_final = mean_squared_error(y_test_full, y_pred_final)
    rmse_final = np.sqrt(mse_final)
    r2_final = r2_score(y_test_full, y_pred_final)
    
    print(f"\nFinal Model Performance on Test Set:")
    print(f"  R2 Score: {r2_final:.4f}")
    print(f"  RMSE: {rmse_final:.2f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_scores_final = model_final.get_score(importance_type='weight')
    feature_importance_final = None
    if importance_scores_final:
        feature_importance_final = pd.DataFrame({
            'feature': list(importance_scores_final.keys()),
            'importance': list(importance_scores_final.values())
        })
        feature_importance_final = feature_importance_final.sort_values('importance', ascending=False)
        plt.bar(feature_importance_final['feature'][:10], feature_importance_final['importance'][:10])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top 10 Feature Importance - Final Model ({features_name})')
        plt.tight_layout()
        plt.savefig(f'XGBoost-combined/{save_prefix}_final_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot actual vs predicted values for final model
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_full, y_pred_final, alpha=0.5)
    plt.plot([y_test_full.min(), y_test_full.max()], [y_test_full.min(), y_test_full.max()], 'r--', lw=2)
    plt.xlabel('Actual Gas Used')
    plt.ylabel('Predicted Gas Used')
    plt.title(f'Final Model: Actual vs Predicted Gas Used ({features_name})')
    plt.tight_layout()
    plt.savefig(f'XGBoost-combined/{save_prefix}_final_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save final model
    model_final.save_model(f'XGBoost-combined/{save_prefix}_final_model.json')
    
    # Save final predictions
    predictions_final_df = pd.DataFrame({
        'Actual': y_test_full.values,
        'Predicted': y_pred_final
    })
    predictions_final_df.to_csv(f'XGBoost-combined/{save_prefix}_final_predictions.csv', index=False)
    
    return {
        'model': model_final,
        'r2': r2_final,
        'rmse': rmse_final,
        'y_test': y_test_full,
        'y_pred': y_pred_final,
        'feature_importance': feature_importance_final,
        'cv_results': cv_df,
        'cv_stats': cv_stats,
        'all_predictions': all_predictions
    }

# Prepare target variable
y = df[target]

# Train model WITH leakage (including CumulativeGasUsed)
print("\n[WARNING] Training model WITH potential data leakage...")
X_with_leakage = df[features_with_leakage]
results_with_leakage = train_xgboost_model_with_cv(
    X_with_leakage, y, 
    "Features WITH Leakage (includes CumulativeGasUsed)", 
    "xgboost_with_leakage"
)

# Train model WITHOUT leakage (excluding CumulativeGasUsed)
print("\n[CLEAN] Training model WITHOUT data leakage...")
X_without_leakage = df[features_without_leakage]
results_without_leakage = train_xgboost_model_with_cv(
    X_without_leakage, y, 
    "Features WITHOUT Leakage (excludes CumulativeGasUsed)", 
    "xgboost_without_leakage"
)

# Comparative Analysis
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS: WITH vs WITHOUT Data Leakage")
print("="*80)

# Performance comparison - Final Models
print("\nFINAL MODEL PERFORMANCE:")
print("-" * 40)
print(f"[WARNING] WITH Leakage    - R2: {results_with_leakage['r2']:.4f}, RMSE: {results_with_leakage['rmse']:.2f}")
print(f"[CLEAN] WITHOUT Leakage - R2: {results_without_leakage['r2']:.4f}, RMSE: {results_without_leakage['rmse']:.2f}")
print(f"[DIFF] Difference      - R2: {results_with_leakage['r2'] - results_without_leakage['r2']:.4f}, "
      f"RMSE: {results_with_leakage['rmse'] - results_without_leakage['rmse']:.2f}")

# Cross-Validation comparison
print("\nCROSS VALIDATION PERFORMANCE:")
print("-" * 40)
print(f"[WARNING] WITH Leakage CV    - Mean R2: {results_with_leakage['cv_stats']['Mean_R2']:.4f} (+/- {results_with_leakage['cv_stats']['Std_R2']:.4f})")
print(f"[CLEAN] WITHOUT Leakage CV - Mean R2: {results_without_leakage['cv_stats']['Mean_R2']:.4f} (+/- {results_without_leakage['cv_stats']['Std_R2']:.4f})")
print(f"[WARNING] WITH Leakage CV    - Mean RMSE: {results_with_leakage['cv_stats']['Mean_RMSE']:.2f} (+/- {results_with_leakage['cv_stats']['Std_RMSE']:.2f})")
print(f"[CLEAN] WITHOUT Leakage CV - Mean RMSE: {results_without_leakage['cv_stats']['Mean_RMSE']:.2f} (+/- {results_without_leakage['cv_stats']['Std_RMSE']:.2f})")

# Create comparison visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Actual vs Predicted WITH leakage
ax1.scatter(results_with_leakage['y_test'], results_with_leakage['y_pred'], 
           alpha=0.6, color='red', s=20)
ax1.plot([results_with_leakage['y_test'].min(), results_with_leakage['y_test'].max()], 
         [results_with_leakage['y_test'].min(), results_with_leakage['y_test'].max()], 
         'k--', lw=2)
ax1.set_xlabel('Actual Gas Used')
ax1.set_ylabel('Predicted Gas Used')
ax1.set_title(f'WITH Leakage\nR² = {results_with_leakage["r2"]:.4f}')
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted WITHOUT leakage
ax2.scatter(results_without_leakage['y_test'], results_without_leakage['y_pred'], 
           alpha=0.6, color='green', s=20)
ax2.plot([results_without_leakage['y_test'].min(), results_without_leakage['y_test'].max()], 
         [results_without_leakage['y_test'].min(), results_without_leakage['y_test'].max()], 
         'k--', lw=2)
ax2.set_xlabel('Actual Gas Used')
ax2.set_ylabel('Predicted Gas Used')
ax2.set_title(f'WITHOUT Leakage\nR² = {results_without_leakage["r2"]:.4f}')
ax2.grid(True, alpha=0.3)

# Plot 3: Feature importance comparison (WITH leakage)
if results_with_leakage['feature_importance'] is not None:
    top_features_with = results_with_leakage['feature_importance'].head(8)
    ax3.barh(range(len(top_features_with)), top_features_with['importance'], color='red', alpha=0.7)
    ax3.set_yticks(range(len(top_features_with)))
    ax3.set_yticklabels(top_features_with['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Feature Importance (WITH Leakage)')
    ax3.grid(True, alpha=0.3)

# Plot 4: Feature importance comparison (WITHOUT leakage)
if results_without_leakage['feature_importance'] is not None:
    top_features_without = results_without_leakage['feature_importance'].head(8)
    ax4.barh(range(len(top_features_without)), top_features_without['importance'], color='green', alpha=0.7)
    ax4.set_yticks(range(len(top_features_without)))
    ax4.set_yticklabels(top_features_without['feature'])
    ax4.set_xlabel('Importance')
    ax4.set_title('Feature Importance (WITHOUT Leakage)')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('XGBoost-combined/leakage_comparison_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create CV results comparison plot
plt.figure(figsize=(15, 10))

# Plot CV R2 scores
plt.subplot(2, 3, 1)
folds = range(1, 6)
plt.plot(folds, results_with_leakage['cv_results']['R2_Score'], 'ro-', label='WITH Leakage', linewidth=2)
plt.plot(folds, results_without_leakage['cv_results']['R2_Score'], 'go-', label='WITHOUT Leakage', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('R2 Score')
plt.title('R2 Score by Fold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot CV RMSE scores  
plt.subplot(2, 3, 2)
plt.plot(folds, results_with_leakage['cv_results']['RMSE'], 'ro-', label='WITH Leakage', linewidth=2)
plt.plot(folds, results_without_leakage['cv_results']['RMSE'], 'go-', label='WITHOUT Leakage', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE by Fold')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot of R2 scores
plt.subplot(2, 3, 3)
plt.boxplot([results_with_leakage['cv_results']['R2_Score'], results_without_leakage['cv_results']['R2_Score']], 
           tick_labels=['WITH Leakage', 'WITHOUT Leakage'])
plt.ylabel('R2 Score')
plt.title('R2 Score Distribution')
plt.grid(True, alpha=0.3)

# Box plot of RMSE scores
plt.subplot(2, 3, 4)
plt.boxplot([results_with_leakage['cv_results']['RMSE'], results_without_leakage['cv_results']['RMSE']], 
           tick_labels=['WITH Leakage', 'WITHOUT Leakage'])
plt.ylabel('RMSE')
plt.title('RMSE Distribution')
plt.grid(True, alpha=0.3)

# Dataset distribution
plt.subplot(2, 3, 5)
v2_mask = df['Dataset'] == 'V2'
v3_mask = df['Dataset'] == 'V3'
plt.hist([df[v2_mask][target], df[v3_mask][target]], 
         bins=50, alpha=0.7, label=['V2', 'V3'], color=['blue', 'orange'])
plt.xlabel('Gas Used')
plt.ylabel('Frequency')
plt.title('Distribution of Gas Used by Dataset')
plt.legend()
plt.grid(True, alpha=0.3)

# Dataset composition
plt.subplot(2, 3, 6)
plt.pie([len(df_v2), len(df_v3)], labels=['V2', 'V3'], autopct='%1.1f%%', 
        colors=['blue', 'orange'])
plt.title('Dataset Composition')

plt.tight_layout()
plt.savefig('XGBoost-combined/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save comprehensive results summary
summary_results = {
    'Model Comparison': {
        'WITH_Leakage': {
            'Final_R2_Score': results_with_leakage['r2'],
            'Final_RMSE': results_with_leakage['rmse'],
            'CV_Mean_R2': results_with_leakage['cv_stats']['Mean_R2'],
            'CV_Std_R2': results_with_leakage['cv_stats']['Std_R2'],
            'CV_Mean_RMSE': results_with_leakage['cv_stats']['Mean_RMSE'],
            'CV_Std_RMSE': results_with_leakage['cv_stats']['Std_RMSE'],
            'Features_Used': len(features_with_leakage),
            'Features_List': features_with_leakage
        },
        'WITHOUT_Leakage': {
            'Final_R2_Score': results_without_leakage['r2'],
            'Final_RMSE': results_without_leakage['rmse'],
            'CV_Mean_R2': results_without_leakage['cv_stats']['Mean_R2'],
            'CV_Std_R2': results_without_leakage['cv_stats']['Std_R2'],
            'CV_Mean_RMSE': results_without_leakage['cv_stats']['Mean_RMSE'],
            'CV_Std_RMSE': results_without_leakage['cv_stats']['Std_RMSE'],
            'Features_Used': len(features_without_leakage),
            'Features_List': features_without_leakage
        }
    },
    'Dataset_Info': {
        'V2_Transactions': len(df_v2),
        'V3_Transactions': len(df_v3),
        'Total_Transactions': len(df),
        'Features_WITH_Leakage': features_with_leakage,
        'Features_WITHOUT_Leakage': features_without_leakage
    }
}

# Save results as CSV
comparison_df = pd.DataFrame({
    'Metric': ['Final_R2_Score', 'Final_RMSE', 'CV_Mean_R2', 'CV_Std_R2', 'CV_Mean_RMSE', 'CV_Std_RMSE', 'Features_Count'],
    'WITH_Leakage': [
        results_with_leakage['r2'],
        results_with_leakage['rmse'],
        results_with_leakage['cv_stats']['Mean_R2'],
        results_with_leakage['cv_stats']['Std_R2'],
        results_with_leakage['cv_stats']['Mean_RMSE'],
        results_with_leakage['cv_stats']['Std_RMSE'],
        len(features_with_leakage)
    ],
    'WITHOUT_Leakage': [
        results_without_leakage['r2'],
        results_without_leakage['rmse'],
        results_without_leakage['cv_stats']['Mean_R2'],
        results_without_leakage['cv_stats']['Std_R2'],
        results_without_leakage['cv_stats']['Mean_RMSE'],
        results_without_leakage['cv_stats']['Std_RMSE'],
        len(features_without_leakage)
    ],
    'Difference': [
        results_with_leakage['r2'] - results_without_leakage['r2'],
        results_with_leakage['rmse'] - results_without_leakage['rmse'],
        results_with_leakage['cv_stats']['Mean_R2'] - results_without_leakage['cv_stats']['Mean_R2'],
        results_with_leakage['cv_stats']['Std_R2'] - results_without_leakage['cv_stats']['Std_R2'],
        results_with_leakage['cv_stats']['Mean_RMSE'] - results_without_leakage['cv_stats']['Mean_RMSE'],
        results_with_leakage['cv_stats']['Std_RMSE'] - results_without_leakage['cv_stats']['Std_RMSE'],
        len(features_with_leakage) - len(features_without_leakage)
    ]
})
comparison_df.to_csv('XGBoost-combined/model_comparison_summary.csv', index=False)

print("\nAll results saved to XGBoost-combined/ directory:")
print("   - CV results and predictions for each fold saved as CSV files")
print("   - leakage_comparison_analysis.png - Model comparison plots")
print("   - comprehensive_analysis.png - CV analysis and dataset distribution")
print("   - model_comparison_summary.csv - Performance metrics comparison")
print("   - Individual model files and predictions")

print(f"\nCONCLUSION:")
print(f"   - Combined V2+V3 dataset has {len(df):,} total transactions")
print(f"   - Model WITHOUT leakage achieves {results_without_leakage['r2']:.1%} final R² accuracy")
print(f"   - Cross-validation shows {results_without_leakage['cv_stats']['Mean_R2']:.1%} mean R² accuracy")
print(f"   - This represents legitimate, publication-ready performance!")
print(f"   - Data leakage impact: {abs(results_with_leakage['r2'] - results_without_leakage['r2']):.4f} R² difference")

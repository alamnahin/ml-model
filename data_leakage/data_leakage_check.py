import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your actual data
df = pd.read_csv('../Ethereum_V3_Transactions.csv')

# Feature correlation matrix
numerical_features = [
    'BlockNumber', 'Nonce', 'TransactionIndex', 'Value', 'gas',
    'GasPrice', 'CumulativeGasUsed', 'GasUsed', 'Confirmations',
    'isError'
]

# Calculate correlation with target
correlations = df[numerical_features + ['GasUsed']].corr()['GasUsed'].abs()
correlations = correlations.sort_index(ascending=False)

print("=== FEATURE CORRELATION WITH TARGET ===")
for feature, corr in correlations.items():
    if feature != 'GasUsed':
        print(f"{feature}: {corr:.4f}")
        if corr > 0.9:
            print(f"⚠️  POTENTIAL LEAKAGE: {feature} has {corr:.2%} correlation!")

# Check for perfect correlations
print("\n=== HIGH CORRELATION FEATURES (>90%) ===")
high_corr = correlations[correlations > 0.9]
if len(high_corr) > 1:  # Excluding target itself
    print("Suspicious features found:")
    for feature, corr in high_corr.items():
        if feature != 'GasUsed':
            print(f"- {feature}: {corr:.4f}")
else:
    print("✅ No high correlation leakage detected")

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df[numerical_features + ['GasUsed']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('v3/correlation_matrix.png')
plt.close()

print(f"\nCorrelation matrix saved: v3/correlation_matrix.png")

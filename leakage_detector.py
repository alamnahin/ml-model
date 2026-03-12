# Simple Leakage Detection Script
import pandas as pd
import numpy as np

def detect_leakage(df, target_col, threshold=0.95):
    """Detect potential data leakage"""
    
    print(f"🔍 Checking for leakage in target: {target_col}")
    print("="*50)
    
    # Calculate correlations
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    print("📊 Feature Correlations with Target:")
    for feature, corr in correlations.items():
        if feature != target_col:
            status = "🚨 LEAKAGE!" if corr > threshold else "✅ OK"
            print(f"  {feature:20}: {corr:.4f} {status}")
    
    # Find suspicious features
    leaky_features = []
    for feature, corr in correlations.items():
        if feature != target_col and corr > threshold:
            leaky_features.append(feature)
    
    if leaky_features:
        print(f"\n⚠️  Found {len(leaky_features)} potentially leaky features:")
        for feature in leaky_features:
            print(f"   - {feature}")
        print("\n💡 Recommendation: Remove these features and retrain model")
    else:
        print("\n✅ No obvious leakage detected!")
    
    return leaky_features

# Example usage:
# df = pd.read_csv('your_data.csv')
# leaky_features = detect_leakage(df, 'GasUsed', threshold=0.9)

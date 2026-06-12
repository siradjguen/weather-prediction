import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
FILES = {
    "Forward Fill":    "meteorological_data_cleaned.csv",
    "Interpolation":   "meteorological_data_cleaned_INTERPOLATION.csv",
    "KNN":             "meteorological_data_cleaned_KNN.csv"
}

NUMERIC_COLS = ["Temperature", "Humidity", "Wind Speed",
                "Pressure", "Solar Radiation", "Rainfall"]

# Contamination: expected proportion of anomalies (e.g., 0.05 = 5% outliers)
CONTAMINATION = 0.05  # Assume 5% of data points are anomalies

# Number of top anomalies to display in detail
TOP_ANOMALIES = 20

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def detect_anomalies(data, contamination=0.05):
    """Train Isolation Forest and detect anomalies"""
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        bootstrap=False,
        n_jobs=-1
    )
    
    # Fit and predict (-1 for anomalies, 1 for normal)
    predictions = iso_forest.fit_predict(data)
    
    # Get anomaly scores (more negative = more anomalous)
    scores = iso_forest.decision_function(data)
    
    return predictions, scores, iso_forest

def analyze_anomalies(df, predictions, scores):
    """Analyze detected anomalies in detail"""
    df_analysis = df.copy()
    df_analysis['Anomaly'] = predictions
    df_analysis['Anomaly_Score'] = scores
    
    # Separate normal and anomalous data
    normal_data = df_analysis[df_analysis['Anomaly'] == 1]
    anomalies = df_analysis[df_analysis['Anomaly'] == -1]
    
    print("\n" + "="*70)
    print("  ANOMALY DETECTION SUMMARY")
    print("="*70)
    print(f"\nTotal samples:        {len(df_analysis)}")
    print(f"Normal samples:       {len(normal_data)} ({len(normal_data)/len(df_analysis)*100:.2f}%)")
    print(f"Anomalous samples:    {len(anomalies)} ({len(anomalies)/len(df_analysis)*100:.2f}%)")
    
    return normal_data, anomalies

def display_top_anomalies(anomalies, top_n=20):
    """Display the most extreme anomalies"""
    # Sort by anomaly score (most negative = most anomalous)
    top_anomalies = anomalies.nsmallest(top_n, 'Anomaly_Score')
    
    print("\n" + "="*70)
    print(f"  TOP {top_n} MOST EXTREME ANOMALIES")
    print("="*70)
    
    for idx, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        print(f"\n🚨 ANOMALY #{idx}")
        print(f"   Date:          {row['Date']}")
        print(f"   Anomaly Score: {row['Anomaly_Score']:.4f} (more negative = more unusual)")
        print(f"   ─────────────────────────────────────")
        
        for col in NUMERIC_COLS:
            print(f"   {col:<20}: {row[col]:>8.2f}")
    
    return top_anomalies

def identify_anomaly_reasons(anomalies, normal_data):
    """Identify which features are most anomalous"""
    print("\n" + "="*70)
    print("  ANOMALY CHARACTERISTICS")
    print("="*70)
    
    comparison = []
    
    for col in NUMERIC_COLS:
        normal_mean = normal_data[col].mean()
        normal_std = normal_data[col].std()
        anomaly_mean = anomalies[col].mean()
        anomaly_std = anomalies[col].std()
        
        # Calculate z-score difference
        z_score = abs(anomaly_mean - normal_mean) / normal_std if normal_std > 0 else 0
        
        print(f"\n{col}:")
        print(f"   Normal data:    {normal_mean:>8.2f} ± {normal_std:.2f}")
        print(f"   Anomalies:      {anomaly_mean:>8.2f} ± {anomaly_std:.2f}")
        print(f"   Z-score diff:   {z_score:>8.2f} {'⚠️ SIGNIFICANT' if z_score > 2 else ''}")
        
        comparison.append({
            'Feature': col,
            'Normal_Mean': normal_mean,
            'Anomaly_Mean': anomaly_mean,
            'Difference': anomaly_mean - normal_mean,
            'Z_Score': z_score
        })
    
    return pd.DataFrame(comparison)

def categorize_anomalies(anomalies, normal_data):
    """Categorize anomalies into types"""
    print("\n" + "="*70)
    print("  ANOMALY CATEGORIES")
    print("="*70)
    
    categories = {
        'Extreme Heat': 0,
        'Extreme Cold': 0,
        'Heavy Rainfall': 0,
        'High Wind': 0,
        'Low Pressure': 0,
        'High Pressure': 0,
        'Multiple Extremes': 0
    }
    
    # Define thresholds (mean ± 2.5 * std)
    temp_high = normal_data['Temperature'].mean() + 2.5 * normal_data['Temperature'].std()
    temp_low = normal_data['Temperature'].mean() - 2.5 * normal_data['Temperature'].std()
    rain_high = normal_data['Rainfall'].mean() + 2.5 * normal_data['Rainfall'].std()
    wind_high = normal_data['Wind Speed'].mean() + 2.5 * normal_data['Wind Speed'].std()
    pressure_low = normal_data['Pressure'].mean() - 2.5 * normal_data['Pressure'].std()
    pressure_high = normal_data['Pressure'].mean() + 2.5 * normal_data['Pressure'].std()
    
    for _, row in anomalies.iterrows():
        extreme_count = 0
        
        if row['Temperature'] > temp_high:
            categories['Extreme Heat'] += 1
            extreme_count += 1
        if row['Temperature'] < temp_low:
            categories['Extreme Cold'] += 1
            extreme_count += 1
        if row['Rainfall'] > rain_high:
            categories['Heavy Rainfall'] += 1
            extreme_count += 1
        if row['Wind Speed'] > wind_high:
            categories['High Wind'] += 1
            extreme_count += 1
        if row['Pressure'] < pressure_low:
            categories['Low Pressure'] += 1
            extreme_count += 1
        if row['Pressure'] > pressure_high:
            categories['High Pressure'] += 1
            extreme_count += 1
        
        if extreme_count >= 2:
            categories['Multiple Extremes'] += 1
    
    print("\nAnomaly Types:")
    for category, count in categories.items():
        if count > 0:
            percentage = (count / len(anomalies)) * 100
            print(f"   {category:<25}: {count:>4} ({percentage:>5.1f}%)")
    
    return categories

# ─────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────
results = {}

for technique, filepath in FILES.items():
    print(f"\n{'='*70}")
    print(f"  ISOLATION FOREST ANOMALY DETECTION: {technique}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    print(f"\nDataset: {len(df)} records")
    print(f"Features: {', '.join(NUMERIC_COLS)}")
    print(f"Expected anomaly rate: {CONTAMINATION*100:.1f}%")
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[NUMERIC_COLS])
    
    # Detect anomalies
    predictions, scores, model = detect_anomalies(scaled_data, CONTAMINATION)
    
    # Analyze results
    normal_data, anomalies = analyze_anomalies(df, predictions, scores)
    
    # Display top anomalies
    top_anomalies = display_top_anomalies(anomalies, TOP_ANOMALIES)
    
    # Compare normal vs anomalous characteristics
    feature_comparison = identify_anomaly_reasons(anomalies, normal_data)
    
    # Categorize anomalies
    anomaly_categories = categorize_anomalies(anomalies, normal_data)
    
    # Store results
    results[technique] = {
        'total_samples': len(df),
        'anomaly_count': len(anomalies),
        'anomaly_percentage': len(anomalies) / len(df) * 100,
        'predictions': predictions,
        'scores': scores,
        'anomalies': anomalies,
        'normal_data': normal_data,
        'top_anomalies': top_anomalies,
        'feature_comparison': feature_comparison,
        'categories': anomaly_categories
    }
    
    # ─────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────
    
    # 1. Anomaly Score Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(normal_data['Anomaly_Score'], bins=50, color='green', alpha=0.7, label='Normal', edgecolor='black')
    ax1.hist(anomalies['Anomaly_Score'], bins=50, color='red', alpha=0.7, label='Anomalies', edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    ax1.set_xlabel('Anomaly Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Anomaly Score Distribution - {technique}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart
    sizes = [len(normal_data), len(anomalies)]
    labels = ['Normal', 'Anomalies']
    colors = ['#90EE90', '#FF6B6B']
    explode = (0, 0.1)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
    ax2.set_title(f'Data Distribution - {technique}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'isolation_forest_distribution_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"\n✓ Saved: isolation_forest_distribution_{technique.replace(' ', '_')}.png")
    
    # 2. Time Series with Anomalies Highlighted
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(NUMERIC_COLS):
        ax = axes[idx]
        
        # Plot normal data
        ax.scatter(normal_data['Date'], normal_data[col], c='blue', alpha=0.5, s=10, label='Normal')
        
        # Highlight anomalies
        ax.scatter(anomalies['Date'], anomalies[col], c='red', alpha=0.8, s=50, 
                  marker='X', edgecolors='black', linewidth=0.5, label='Anomaly', zorder=5)
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(f'{col} Over Time', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Anomaly Detection in Time Series - {technique}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'isolation_forest_timeseries_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Saved: isolation_forest_timeseries_{technique.replace(' ', '_')}.png")
    
    # 3. Feature Comparison Heatmap
    plt.figure(figsize=(10, 6))
    
    comparison_matrix = feature_comparison.set_index('Feature')[['Normal_Mean', 'Anomaly_Mean']]
    comparison_matrix_normalized = (comparison_matrix - comparison_matrix.min()) / (comparison_matrix.max() - comparison_matrix.min())
    
    sns.heatmap(comparison_matrix_normalized.T, annot=comparison_matrix.T.values, fmt='.2f',
                cmap='RdYlGn_r', cbar_kws={'label': 'Normalized Value'},
                linewidths=0.5, linecolor='gray', yticklabels=['Normal Data', 'Anomalies'])
    plt.title(f'Feature Comparison: Normal vs Anomalies - {technique}', fontsize=14, fontweight='bold')
    plt.xlabel('Weather Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'isolation_forest_comparison_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Saved: isolation_forest_comparison_{technique.replace(' ', '_')}.png")
    
    # 4. Anomaly Categories Bar Chart
    if sum(anomaly_categories.values()) > 0:
        plt.figure(figsize=(10, 6))
        
        categories_df = pd.DataFrame(list(anomaly_categories.items()), columns=['Category', 'Count'])
        categories_df = categories_df[categories_df['Count'] > 0].sort_values('Count', ascending=False)
        
        colors_bar = plt.cm.Reds(np.linspace(0.4, 0.8, len(categories_df)))
        bars = plt.barh(categories_df['Category'], categories_df['Count'], color=colors_bar, edgecolor='black', linewidth=1.5)
        
        plt.xlabel('Number of Anomalies', fontsize=12)
        plt.ylabel('Anomaly Category', fontsize=12)
        plt.title(f'Anomaly Categories - {technique}', fontsize=14, fontweight='bold')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'isolation_forest_categories_{technique.replace(" ", "_")}.png', dpi=300)
        print(f"✓ Saved: isolation_forest_categories_{technique.replace(' ', '_')}.png")
    
    # 5. Box Plots for Each Feature
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(NUMERIC_COLS):
        ax = axes[idx]
        
        data_to_plot = [normal_data[col], anomalies[col]]
        bp = ax.boxplot(data_to_plot, labels=['Normal', 'Anomalies'], patch_artist=True,
                       widths=0.6, showfliers=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('#90EE90')
        bp['boxes'][1].set_facecolor('#FF6B6B')
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        
        ax.set_ylabel(col, fontsize=11)
        ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Feature Distribution: Normal vs Anomalies - {technique}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'isolation_forest_boxplots_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Saved: isolation_forest_boxplots_{technique.replace(' ', '_')}.png")
    
    plt.close('all')

# ─────────────────────────────────────────
# FINAL COMPARISON
# ─────────────────────────────────────────
print(f"\n{'='*70}")
print("  FINAL COMPARISON ACROSS IMPUTATION TECHNIQUES")
print(f"{'='*70}")
print(f"{'Technique':<20} {'Total Samples':>15} {'Anomalies':>12} {'Percentage':>12}")
print("-" * 70)

for technique, res in results.items():
    print(f"{technique:<20} {res['total_samples']:>15} {res['anomaly_count']:>12} {res['anomaly_percentage']:>11.2f}%")

print("\n" + "="*70)
print("  ISOLATION FOREST ANOMALY DETECTION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("   • Anomaly score distributions")
print("   • Time series with anomalies highlighted")
print("   • Feature comparison heatmaps")
print("   • Anomaly category charts")
print("   • Box plots comparing normal vs anomalous data")
print("\nKey Insights:")
print("   → Check the TOP 20 anomalies for specific extreme weather events")
print("   → Review anomaly categories to understand types of unusual weather")
print("   → Compare feature means to see which variables deviate most")

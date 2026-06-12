import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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

# Test different numbers of clusters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
OPTIMAL_CLUSTERS = 4  # Will be determined by elbow method

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def find_optimal_clusters(data, min_k, max_k):
    """Use Elbow Method and Silhouette Score to find optimal K"""
    inertias = []
    silhouette_scores = []
    K_range = range(min_k, max_k + 1)
    
    print("\nFinding optimal number of clusters...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score needs at least 2 clusters
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
        
        print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")
    
    return K_range, inertias, silhouette_scores

def analyze_clusters(df, labels, scaled_data):
    """Analyze characteristics of each cluster"""
    df_analysis = df.copy()
    df_analysis['Cluster'] = labels
    
    print("\n" + "="*70)
    print("  CLUSTER CHARACTERISTICS")
    print("="*70)
    
    cluster_stats = []
    for cluster_id in sorted(df_analysis['Cluster'].unique()):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
        print(f"\n📊 CLUSTER {cluster_id} ({len(cluster_data)} samples, {len(cluster_data)/len(df)*100:.1f}%)")
        print("-" * 70)
        
        stats = {}
        stats['Cluster'] = cluster_id
        stats['Count'] = len(cluster_data)
        stats['Percentage'] = f"{len(cluster_data)/len(df)*100:.1f}%"
        
        for col in NUMERIC_COLS:
            mean_val = cluster_data[col].mean()
            std_val = cluster_data[col].std()
            print(f"   {col:<20}: {mean_val:>8.2f} ± {std_val:.2f}")
            stats[f"{col}_mean"] = mean_val
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)

def interpret_clusters(cluster_stats):
    """Give meaningful names to clusters based on their characteristics"""
    print("\n" + "="*70)
    print("  CLUSTER INTERPRETATION")
    print("="*70)
    
    interpretations = []
    
    for idx, row in cluster_stats.iterrows():
        cluster_id = int(row['Cluster'])
        temp = row['Temperature_mean']
        humidity = row['Humidity_mean']
        rainfall = row['Rainfall_mean']
        solar = row['Solar Radiation_mean']
        wind = row['Wind Speed_mean']
        
        # Simple rule-based interpretation
        name = ""
        description = []
        
        # Temperature classification
        if temp > 25:
            name = "Hot"
            description.append("high temperature")
        elif temp < 15:
            name = "Cold"
            description.append("low temperature")
        else:
            name = "Moderate"
            description.append("moderate temperature")
        
        # Rainfall
        if rainfall > 5:
            name += " & Rainy"
            description.append("high rainfall")
        elif rainfall > 1:
            description.append("some rainfall")
        
        # Humidity
        if humidity > 70:
            description.append("humid")
        elif humidity < 40:
            description.append("dry")
        
        # Solar radiation
        if solar > 500:
            description.append("sunny")
        elif solar < 200:
            description.append("cloudy/dark")
        
        # Wind
        if wind > 15:
            description.append("windy")
        
        interpretation = f"Cluster {cluster_id}: {name} Weather"
        details = ", ".join(description)
        
        print(f"\n{interpretation}")
        print(f"   → Characteristics: {details}")
        print(f"   → {row['Percentage']} of all observations")
        
        interpretations.append({
            'Cluster': cluster_id,
            'Name': name,
            'Details': details,
            'Percentage': row['Percentage']
        })
    
    return pd.DataFrame(interpretations)

# ─────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────
results = {}

for technique, filepath in FILES.items():
    print(f"\n{'='*70}")
    print(f"  K-MEANS CLUSTERING: {technique}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    print(f"\nDataset: {len(df)} records")
    print(f"Features: {', '.join(NUMERIC_COLS)}")
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[NUMERIC_COLS])
    
    # Find optimal number of clusters
    K_range, inertias, silhouette_scores = find_optimal_clusters(
        scaled_data, MIN_CLUSTERS, MAX_CLUSTERS
    )
    
    # Determine best K (highest silhouette score)
    best_k_idx = np.argmax(silhouette_scores[1:]) + 1  # Skip K=1
    optimal_k = K_range[best_k_idx]
    
    print(f"\n✓ Optimal number of clusters: {optimal_k}")
    print(f"  (Silhouette Score: {silhouette_scores[best_k_idx]:.4f})")
    
    # Perform final clustering with optimal K
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(scaled_data)
    
    # Calculate metrics
    silhouette = silhouette_score(scaled_data, labels)
    davies_bouldin = davies_bouldin_score(scaled_data, labels)
    
    print(f"\nClustering Quality Metrics:")
    print(f"   Silhouette Score:     {silhouette:.4f} (higher is better, range: -1 to 1)")
    print(f"   Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    # Analyze clusters
    cluster_stats = analyze_clusters(df, labels, scaled_data)
    interpretations = interpret_clusters(cluster_stats)
    
    # Store results
    results[technique] = {
        'optimal_k': optimal_k,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'labels': labels,
        'cluster_stats': cluster_stats,
        'interpretations': interpretations,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }
    
    # ─────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────
    
    # 1. Elbow Method Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax1.set_title(f'Elbow Method - {technique}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(K_range[1:], silhouette_scores[1:], 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title(f'Silhouette Score - {technique}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'kmeans_elbow_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"\n✓ Saved: kmeans_elbow_{technique.replace(' ', '_')}.png")
    
    # 2. Cluster Distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    
    bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Cluster Distribution - {technique}', fontsize=14, fontweight='bold')
    plt.xticks(cluster_counts.index)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'kmeans_distribution_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Saved: kmeans_distribution_{technique.replace(' ', '_')}.png")
    
    # 3. Cluster Heatmap (mean values)
    plt.figure(figsize=(12, 6))
    heatmap_data = cluster_stats.set_index('Cluster')[[f"{col}_mean" for col in NUMERIC_COLS]]
    heatmap_data.columns = NUMERIC_COLS  # Clean column names
    
    # Normalize each feature for better visualization
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    sns.heatmap(heatmap_normalized, annot=heatmap_data.values, fmt='.2f', 
                cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'},
                linewidths=0.5, linecolor='gray')
    plt.title(f'Cluster Characteristics Heatmap - {technique}', fontsize=14, fontweight='bold')
    plt.xlabel('Weather Features', fontsize=12)
    plt.ylabel('Cluster ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'kmeans_heatmap_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Saved: kmeans_heatmap_{technique.replace(' ', '_')}.png")
    
    # 4. Time Series with Cluster Colors
    plt.figure(figsize=(16, 6))
    colors_map = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    
    for cluster_id in range(optimal_k):
        mask = labels == cluster_id
        plt.scatter(df[mask]['Date'], df[mask]['Temperature'], 
                   c=[colors_map[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=20)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title(f'Temperature Over Time (Colored by Cluster) - {technique}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'kmeans_timeseries_{technique.replace(" ", "_")}.png', dpi=300)
    print(f"✓ Saved: kmeans_timeseries_{technique.replace(' ', '_')}.png")
    
    plt.close('all')

# ─────────────────────────────────────────
# FINAL COMPARISON
# ─────────────────────────────────────────
print(f"\n{'='*70}")
print("  FINAL COMPARISON ACROSS IMPUTATION TECHNIQUES")
print(f"{'='*70}")
print(f"{'Technique':<20} {'Optimal K':>12} {'Silhouette':>12} {'Davies-Bouldin':>16}")
print("-" * 70)

for technique, res in results.items():
    print(f"{technique:<20} {res['optimal_k']:>12} {res['silhouette']:>12.4f} {res['davies_bouldin']:>16.4f}")

best_technique = max(results, key=lambda x: results[x]['silhouette'])
print(f"\n✓ Best clustering quality: {best_technique}")
print(f"  (Highest Silhouette Score: {results[best_technique]['silhouette']:.4f})")

print("\n" + "="*70)
print("  K-MEANS CLUSTERING ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("   • Elbow method plots")
print("   • Cluster distribution charts")
print("   • Cluster characteristics heatmaps")
print("   • Time series with cluster colors")

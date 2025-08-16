import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def visualize_knn_with_decision_boundary(csv_file='KNN.csv'):
    """
    Visualize KNN results with decision boundary and prediction labels
    """
    # Read the data
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} data points")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get feature columns
    x_col, y_col = df.columns[0], df.columns[1]
    
    # Separate data types
    dataset_points = df[df['type'] == 'dataset']
    query_points = df[df['type'] == 'query']
    neighbor_points = df[df['type'] == 'neighbor']
    
    # Color scheme
    colors = {'query': 'red', 'neighbor': 'orange', 'dataset': 'black'}
    sizes = {'query': 200, 'neighbor': 100, 'dataset': 50}
    markers = {'query': '*', 'neighbor': 'o', 'dataset': '.'}
    
    # Plot 1: Standard KNN visualization
    for point_type in ['dataset', 'neighbor', 'query']:
        subset = df[df['type'] == point_type]
        if len(subset) > 0:
            ax1.scatter(subset[x_col], subset[y_col], 
                       c=colors[point_type], 
                       s=sizes[point_type],
                       marker=markers[point_type],
                       label=f'{point_type.title()} ({len(subset)} points)',
                       alpha=0.7 if point_type == 'dataset' else 1.0,
                       edgecolors='black' if point_type != 'dataset' else 'none')
    
    # Draw connections and circles for each query
    for query_id in query_points['query_id'].unique():
        query_point = query_points[query_points['query_id'] == query_id]
        query_neighbors = neighbor_points[neighbor_points['query_id'] == query_id]
        
        if len(query_point) > 0 and len(query_neighbors) > 0:
            qx, qy = query_point.iloc[0][x_col], query_point.iloc[0][y_col]
            
            # Draw lines from query to each neighbor
            for _, neighbor in query_neighbors.iterrows():
                nx, ny = neighbor[x_col], neighbor[y_col]
                ax1.plot([qx, nx], [qy, ny], 'r--', alpha=0.5, linewidth=1)
            
            # Draw encompassing circle
            max_distance = query_neighbors['distance'].max()
            circle = plt.Circle((qx, qy), max_distance, 
                               fill=False, color='red', 
                               linewidth=2, alpha=0.6, linestyle='--')
            ax1.add_patch(circle)
            
            # Calculate prediction for this query
            neighbor_labels = query_neighbors['label'].tolist()
            vote_counts = Counter(neighbor_labels)
            predicted_label = vote_counts.most_common(1)[0][0]
            
            # Annotate query with prediction
            ax1.annotate(f'Query {int(query_id)}\nPredicted: {predicted_label}', 
                        (qx, qy), xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax1.set_xlabel(f'{x_col} (Feature 1)')
    ax1.set_ylabel(f'{y_col} (Feature 2)')
    ax1.set_title('KNN Visualization with Query Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decision boundary visualization
    if len(dataset_points) > 0:
        # Create a mesh of points to show decision boundary
        x_min, x_max = df[x_col].min() - 5, df[x_col].max() + 5
        y_min, y_max = df[y_col].min() - 5, df[y_col].max() + 5
        
        # Create a grid of points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 2),
                            np.arange(y_min, y_max, 2))
        
        # For each grid point, find the K nearest neighbors and predict
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = []
        
        K = len(neighbor_points) // len(query_points)  # Infer K from data
        
        for point in grid_points:
            # Calculate distances to all dataset points
            distances = []
            for _, dataset_point in dataset_points.iterrows():
                dist = np.sqrt((point[0] - dataset_point[x_col])**2 + 
                              (point[1] - dataset_point[y_col])**2)
                distances.append((dist, dataset_point['label']))
            
            # Sort by distance and take K nearest
            distances.sort()
            k_nearest = distances[:K]
            
            # Vote
            votes = [label for _, label in k_nearest]
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        
        # Reshape predictions to match the grid
        predictions = np.array(predictions).reshape(xx.shape)
        
        # Plot decision boundary using contour
        unique_labels = sorted(dataset_points['label'].unique())
        colors_boundary = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        contour = ax2.contourf(xx, yy, predictions, levels=len(unique_labels)-1, 
                              colors=colors_boundary, alpha=0.4)
        
        # Add contour lines
        ax2.contour(xx, yy, predictions, levels=len(unique_labels)-1, 
                   colors='black', linewidths=0.5, alpha=0.5)
        
        # Plot dataset points colored by their labels
        for i, label in enumerate(unique_labels):
            label_points = dataset_points[dataset_points['label'] == label]
            ax2.scatter(label_points[x_col], label_points[y_col], 
                       c=[colors_boundary[i]], s=50, 
                       label=f'Class {label}', edgecolors='black', linewidth=0.5)
        
        # Plot queries with their predictions
        for query_id in query_points['query_id'].unique():
            query_point = query_points[query_points['query_id'] == query_id]
            query_neighbors = neighbor_points[neighbor_points['query_id'] == query_id]
            
            if len(query_point) > 0 and len(query_neighbors) > 0:
                qx, qy = query_point.iloc[0][x_col], query_point.iloc[0][y_col]
                
                # Calculate prediction
                neighbor_labels = query_neighbors['label'].tolist()
                vote_counts = Counter(neighbor_labels)
                predicted_label = vote_counts.most_common(1)[0][0]
                
                ax2.scatter(qx, qy, c='red', s=200, marker='*', 
                           edgecolors='black', linewidth=2, zorder=10)
                
                # Add prediction label in the region
                ax2.annotate(f'Q{int(query_id)}: Class {predicted_label}', 
                            (qx, qy), xytext=(15, 15), textcoords='offset points',
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                                    edgecolor="red", linewidth=2, alpha=0.9))
        
        ax2.set_xlabel(f'{x_col} (Feature 1)')
        ax2.set_ylabel(f'{y_col} (Feature 2)')
        ax2.set_title(f'KNN Decision Boundary (K={K})\nBackground colors show predicted regions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar to show class regions
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label('Predicted Class')
    
    plt.tight_layout()
    
    # Print detailed analysis
    print("\n=== DETAILED KNN ANALYSIS ===")
    for query_id in query_points['query_id'].unique():
        query_neighbors = neighbor_points[neighbor_points['query_id'] == query_id]
        if len(query_neighbors) > 0:
            print(f"\nQuery {int(query_id)}:")
            print(f"  K = {len(query_neighbors)} nearest neighbors")
            print(f"  Average distance: {query_neighbors['distance'].mean():.3f}")
            print(f"  Neighbor labels: {list(query_neighbors['label'])}")
            
            # Vote counting
            label_counts = query_neighbors['label'].value_counts()
            print(f"  Vote counts: {dict(label_counts)}")
            predicted_label = label_counts.index[0]
            print(f"  PREDICTED CLASS: {predicted_label}")
            
            # Show confidence
            total_votes = len(query_neighbors)
            confidence = label_counts.iloc[0] / total_votes * 100
            print(f"  Confidence: {confidence:.1f}%")
    
    plt.show()

if __name__ == "__main__":
    visualize_knn_with_decision_boundary('KNN.csv')
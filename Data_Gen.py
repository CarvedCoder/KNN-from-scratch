import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_circles
import argparse

def generate_clustered_data(num_points=200, dimension=2, num_classes=2, spread=100, cluster_std=15):
    """Generate data with natural clusters for each class"""
    # Create cluster centers for each class
    centers = np.random.rand(num_classes, dimension) * spread
    
    # Generate points around each center
    points_per_class = num_points // num_classes
    X = []
    y = []
    
    for class_id in range(num_classes):
        # Generate points around this class center
        class_points = np.random.normal(
            loc=centers[class_id], 
            scale=cluster_std, 
            size=(points_per_class, dimension)
        )
        X.append(class_points)
        y.extend([class_id] * points_per_class)
    
    # Handle remaining points if num_points doesn't divide evenly
    remaining = num_points - len(y)
    if remaining > 0:
        extra_class = np.random.randint(0, num_classes)
        extra_points = np.random.normal(
            loc=centers[extra_class], 
            scale=cluster_std, 
            size=(remaining, dimension)
        )
        X.append(extra_points)
        y.extend([extra_class] * remaining)
    
    X = np.vstack(X)
    y = np.array(y)
    
    return X, y

def generate_sklearn_classification(num_points=200, dimension=2, num_classes=2, 
                                   class_sep=1.0, noise=0.1):
    """Use sklearn's make_classification for realistic data"""
    X, y = make_classification(
        n_samples=num_points,
        n_features=dimension,
        n_informative=dimension,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=num_classes,
        class_sep=class_sep,
        random_state=42,
        flip_y=noise  # Add some label noise
    )
    
    # Scale to desired range
    X = (X - X.min()) / (X.max() - X.min()) * 100
    
    return X, y

def generate_blobs(num_points=200, dimension=2, num_classes=2, spread=100, cluster_std=15):
    """Generate blob-like clusters"""
    X, y = make_blobs(
        n_samples=num_points,
        centers=num_classes,
        n_features=dimension,
        cluster_std=cluster_std,
        center_box=(0.0, spread),
        random_state=42
    )
    return X, y

def generate_circles(num_points=200, noise=0.1):
    """Generate circular decision boundary (2D, 2 classes only)"""
    X, y = make_circles(
        n_samples=num_points,
        noise=noise,
        factor=0.6,
        random_state=42
    )
    
    # Scale to 0-100 range
    X = (X - X.min()) / (X.max() - X.min()) * 100
    
    return X, y

def generate_custom_pattern():
    """Generate a custom interesting pattern"""
    np.random.seed(42)
    
    # Class 0: Two separate clusters
    cluster1_0 = np.random.normal([20, 20], [8, 8], (50, 2))
    cluster2_0 = np.random.normal([80, 80], [8, 8], (50, 2))
    class_0 = np.vstack([cluster1_0, cluster2_0])
    
    # Class 1: One cluster between the class 0 clusters
    class_1 = np.random.normal([50, 50], [12, 12], (100, 2))
    
    # Combine
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(100), np.ones(100)])
    
    return X, y

def visualize_and_save(X, y, filename="dataset.csv", title="Generated Dataset"):
    """Visualize the data and save to CSV"""
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Scatter plot
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for class_id in np.unique(y):
        mask = y == class_id
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[class_id % len(colors)], 
                   label=f'Class {class_id}', 
                   alpha=0.7, s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title} - Scatter Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Class distribution
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts, color=[colors[i % len(colors)] for i in unique])
    plt.xlabel('Class')
    plt.ylabel('Number of Points')
    plt.title('Class Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save to CSV
    if X.shape[1] == 2:
        df = pd.DataFrame({
            'feature0': X[:, 0],
            'feature1': X[:, 1],
            'label': y.astype(int)
        })
    else:
        # For higher dimensions
        columns = [f'feature{i}' for i in range(X.shape[1])] + ['label']
        data = np.hstack((X, y.reshape(-1, 1)))
        df = pd.DataFrame(data, columns=columns)
    
    df.to_csv(filename, index=False)
    
    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total points: {len(X)}")
    print(f"Dimensions: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Feature ranges:")
    for i in range(X.shape[1]):
        print(f"  Feature {i}: {X[:, i].min():.2f} to {X[:, i].max():.2f}")
    print(f"Dataset saved to '{filename}'")

def main():
    print("Dataset Generation Options:")
    print("1. Clustered data (Gaussian clusters)")
    print("2. Sklearn classification dataset")
    print("3. Blob clusters")
    print("4. Circular decision boundary")
    print("5. Custom interesting pattern")
    print("6. Your original random method")
    
    choice = input("Choose method (1-6): ").strip()
    
    if choice == '1':
        X, y = generate_clustered_data(
            num_points=200, 
            dimension=2, 
            num_classes=3, 
            spread=100, 
            cluster_std=12
        )
        title = "Clustered Data"
        
    elif choice == '2':
        X, y = generate_sklearn_classification(
            num_points=200, 
            dimension=2, 
            num_classes=3, 
            class_sep=1.5, 
            noise=0.05
        )
        title = "Sklearn Classification"
        
    elif choice == '3':
        X, y = generate_blobs(
            num_points=200, 
            dimension=2, 
            num_classes=3, 
            spread=100, 
            cluster_std=15
        )
        title = "Blob Clusters"
        
    elif choice == '4':
        X, y = generate_circles(num_points=200, noise=0.1)
        title = "Circular Boundary"
        
    elif choice == '5':
        X, y = generate_custom_pattern()
        title = "Custom Pattern"
        
    else:  # choice == '6' or default
        # Your original method
        X = np.random.rand(200, 2) * 100
        y = np.random.randint(0, 2, size=200)
        title = "Random Labels (Original)"
    
    visualize_and_save(X, y, title=title)

if __name__ == "__main__":
    main()

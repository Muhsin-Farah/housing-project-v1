import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_train_test_split(train_size: int, test_size: int) -> None:
    """Visualize the train/test split distribution."""
    sizes = [train_size, test_size]
    labels = ['Training', 'Testing']
    colors = ['#66b3ff', '#ff9999']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Train/Test Split Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_actual, y_predicted, title: str = "Predictions vs Actual") -> None:
    """Plot actual values vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.5, edgecolors='k')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_actual, y_predicted, title: str = "Residuals Plot") -> None:
    """Plot residuals (errors) of predictions."""
    residuals = y_actual - y_predicted
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_predicted, residuals, alpha=0.5, edgecolors='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importance_scores, title: str = "Feature Importance") -> None:
    """Plot feature importance from model."""
    indices = np.argsort(importance_scores)[::-1][:10]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importance_scores[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_model_performance_metrics(metrics: dict) -> None:
    """Plot model evaluation metrics (MSE, RMSE, R², MAE)."""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#66b3ff', '#99ff99', '#ffcc99', '#ff9999'])
    plt.ylabel('Score / Error')
    plt.title('Model Performance Metrics')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_data_distribution(data, column: str, bins: int = 30) -> None:
    """Plot histogram of data distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, title: str = "Feature Correlation Heatmap") -> None:
    """Plot correlation heatmap of features."""
    import matplotlib.patches as mpatches
    
    correlation_matrix = df.corr(numeric_only=True)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title(title)
    plt.tight_layout()
    plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from perceptron import Perceptron

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_pred)
    return accuracy

def test_different_learning_rates():
    """Teste différents learning rates"""
    X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    print("=" * 50)
    print("Test de différents learning rates:")
    print("=" * 50)
    
    for lr in learning_rates:
        p = Perceptron(l_rate=lr, n_iters=1000)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)
        acc = accuracy(y_test, predictions)
        print(f"Learning rate: {lr:5.3f} | Accuracy: {acc:.4f}")

def visualize_perceptron(learning_rate=0.01):
    """Visualise le perceptron avec un learning rate donné"""
    X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    p = Perceptron(l_rate=learning_rate, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)
    
    print(f"\nPerceptron (lr={learning_rate}) classification accuracy: {accuracy(y_test, predictions):.4f}")
    
    # Visualisation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Points d'entraînement
    scatter = plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train, alpha=0.8, s=50)
    
    # Points de test
    plt.scatter(X_test[:,0], X_test[:,1], marker='s', c=y_test, alpha=0.6, s=80, edgecolors='black')
    
    # Ligne de séparation
    if p.weights[1] != 0:  # Éviter la division par zéro
        x0_1 = np.amin(X_train[:,0]) - 1
        x0_2 = np.amax(X_train[:,0]) + 1
        
        x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
        x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
        
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k-', linewidth=2, label='Frontière de décision')
    
    plt.title(f'Perceptron (Learning Rate = {learning_rate})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Frontière', 'Classe 0 (train)', 'Classe 1 (train)', 'Test'])
    
    ymin = np.amin(X_train[:,1]) - 2
    ymax = np.amax(X_train[:,1]) + 2
    ax.set_ylim([ymin, ymax])
    
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Test des différents learning rates
    test_different_learning_rates()
    
    # Visualisation avec un learning rate spécifique
    visualize_perceptron(learning_rate=0.01)

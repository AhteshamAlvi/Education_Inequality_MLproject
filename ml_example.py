
# Linear Regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_s_curve
from sklearn.metrics import mean_squared_error, r2_score

X_linear, y_linear = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# (2) "convex" like dataset
X_convex = np.linspace(-5, 5, 100).reshape(-1, 1)
y_convex = X_convex**2 + np.random.normal(0, 5, size=100).reshape(-1, 1)
y_convex = y_convex.flatten()

# (3) "S-shaped curve" dataset (alternative to trigonometric)
X_s_curve, y_s_curve = make_s_curve(n_samples=100, noise=0.1, random_state=42)
X_s_curve = X_s_curve[:, 0].reshape(-1, 1) # Use only the first feature for simplicity

datasets = {
    "linear_real": (X_linear, y_linear),
    "convex": (X_convex, y_convex),
    "Trigonometric": (X_s_curve, y_s_curve)
}

# Create the subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot each dataset
for ax, dataset in zip(axes, datasets.items()):
    title, (X, y) = dataset
    ax.scatter(X, y, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')

plt.tight_layout()
plt.show()

X_linear, Y_linear = datasets["linear_real"]
X_convex, Y_convex = datasets["convex"]
X_tri, Y_tri = datasets["Trigonometric"]

random_state = 42
np.random.seed(random_state)
test_size = 0.2

def split_data(X, Y, test_size=test_size, random_state=random_state):
    # YOUR CODE HERE (1-2 lines)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

# DO NOT MODIFY
# Apply the function for each dataset
X_linear_train, X_linear_test, Y_linear_train, Y_linear_test = split_data(X_linear, Y_linear)
X_convex_train, X_convex_test, Y_convex_train, Y_convex_test = split_data(X_convex, Y_convex)
X_tri_train, X_tri_test, Y_tri_train, Y_tri_test = split_data(X_tri, Y_tri)

# Set the variable to select the model to plot
selected_dataset = 'Linear'  # Choose between 'Linear', 'Convex', 'Trigonometric'

# DO NOT MODIFY
def draw_scatter(X_train, Y_train, X_test, Y_test, title, xlabel='Feature', ylabel='Target'):
    plt.scatter(X_train, Y_train, color='blue', label='Train', alpha=0.5)
    plt.scatter(X_test, Y_test, color='red', label='Test', alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Dictionary to store datasets
datasets = {
    'Linear': (X_linear_train, Y_linear_train, X_linear_test, Y_linear_test, 'Linear Relationship'),
    'Convex': (X_convex_train, Y_convex_train, X_convex_test, Y_convex_test, 'Convex Relationship'),
    'Trigonometric': (X_tri_train, Y_tri_train, X_tri_test, Y_tri_test, 'Trigonometric Relationship')
}

# Extract selected dataset
X_train, Y_train, X_test, Y_test, title = datasets[selected_dataset]

# Plot the selected dataset
draw_scatter(X_train, Y_train, X_test, Y_test, title)

# Function to fit the model
def fit_model(X_train, Y_train):
    # YOUR CODE HERE (1-2 lines)
    model = LinearRegression().fit(X_train, Y_train)
    return model

# Function to predict using the fitted model
def predict_data(model, X_train, X_test):
    # YOUR CODE HERE (1-2 lines)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    return Y_train_pred, Y_test_pred


# DO NOT MODIFY
# Fit the model for each dataset and predict
linear_model = fit_model(X_linear_train, Y_linear_train)
Y_linear_train_pred, Y_linear_test_pred = predict_data(linear_model, X_linear_train, X_linear_test)
convex_model = fit_model(X_convex_train, Y_convex_train)
Y_convex_train_pred, Y_convex_test_pred = predict_data(convex_model, X_convex_train, X_convex_test)
trigonometric_model = fit_model(X_tri_train, Y_tri_train)
Y_tri_train_pred, Y_tri_test_pred = predict_data(trigonometric_model, X_tri_train, X_tri_test)

# Set the variable to select the model to plot
selected_dataset = 'Trigonometric'  # Choose between 'Linear', 'Convex', 'Trigonometric'

# DO NOT MODIFY
# Function to draw scatter plot with regression line
def draw_scatter_with_regression(X_train, Y_train, X_test, Y_test, Y_train_pred, title, xlabel='Feature', ylabel='Target'):
    plt.scatter(X_train, Y_train, color='blue', label='Train')
    plt.scatter(X_test, Y_test, color='orange', label='Test')
    plt.plot(X_train, Y_train_pred, color='green', label='Regression Line')
    plt.title(title)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Dictionary to store datasets and predictions
datasets = {
    'Linear': (X_linear_train, Y_linear_train, X_linear_test, Y_linear_test, Y_linear_train_pred, 'Linear Dataset Regression'),
    'Convex': (X_convex_train, Y_convex_train, X_convex_test, Y_convex_test, Y_convex_train_pred, 'Convex Dataset Regression'),
    'Trigonometric': (X_tri_train, Y_tri_train, X_tri_test, Y_tri_test, Y_tri_train_pred, 'Trigonometric Dataset Regression')
}

# Extract selected dataset
X_train, Y_train, X_test, Y_test, Y_train_pred, title = datasets[selected_dataset]

# Plot the selected dataset with regression line
draw_scatter_with_regression(X_train, Y_train, X_test, Y_test, Y_train_pred, title)


# Function to calculate MSE and R^2
def evaluate_model(Y_train, Y_train_pred, Y_test, Y_test_pred):
    # YOUR CODE HERE
    mse_train = mean_squared_error(Y_train, Y_train_pred)
    mse_test = mean_squared_error(Y_test, Y_test_pred)

    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)

    return mse_train, mse_test, r2_train, r2_test

# DO NOT MODIFY
# Apply the function for each dataset
mse_linear_train, mse_linear_test, r2_linear_train, r2_linear_test = evaluate_model(
    Y_linear_train, Y_linear_train_pred, Y_linear_test, Y_linear_test_pred)

mse_convex_train, mse_convex_test, r2_convex_train, r2_convex_test = evaluate_model(
    Y_convex_train, Y_convex_train_pred, Y_convex_test, Y_convex_test_pred)

mse_tri_train, mse_tri_test, r2_tri_train, r2_tri_test = evaluate_model(
    Y_tri_train, Y_tri_train_pred, Y_tri_test, Y_tri_test_pred)

# DO NOT MODIFY
# List of datasets and their metrics
datasets = ["Linear", "Convex", "Trigonometric"]
mse_train = [mse_linear_train, mse_convex_train, mse_tri_train]
mse_test = [mse_linear_test, mse_convex_test, mse_tri_test]
r2_train = [r2_linear_train, r2_convex_train, r2_tri_train]
r2_test = [r2_linear_test, r2_convex_test, r2_tri_test]

# Function to store and display results
def store_and_display_results(datasets, mse_train, mse_test, r2_train, r2_test):
    results = []
    for i, dataset in enumerate(datasets):
        result = {
            "Dataset": dataset,
            "MSE Train": mse_train[i],
            "MSE Test": mse_test[i],
            "R2 Train": r2_train[i],
            "R2 Test": r2_test[i]
        }
        results.append(result)

        # Display the results
        print(f"{dataset} Dataset:")
        print(f"  MSE Train: {mse_train[i]:.4f}")
        print(f"  MSE Test: {mse_test[i]:.4f}")
        print(f"  R2 Train: {r2_train[i]:.4f}")
        print(f"  R2 Test: {r2_test[i]:.4f}")
        print()  # Blank line for readability

    return results

# Store and display the results
results = store_and_display_results(datasets, mse_train, mse_test, r2_train, r2_test)


# Function to fit the polynomial model
def fit_model_poly(X_train, Y_train, degree):
    # YOUR CODE HERE
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, Y_train)
    
    return model

# Function to predict using the fitted polynomial model
def predict_data_poly(model, X_train, X_test):
    # YOUR CODE HERE
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    return Y_train_pred, Y_test_pred

# DO NOT MODIFY
# Define degrees for polynomial features
degree_linear = 1  # For the linear dataset, a degree of 1 is just a linear relationship
degree_convex = 3  # For the convex dataset, let's try a quadratic model
degree_tri = 8    # For the trigonometric dataset, a higher degree might capture the sine/cosine waves

# Fit and predict for the linear dataset
linear_model_poly = fit_model_poly(X_linear_train, Y_linear_train, degree_linear)
Y_linear_train_pred_poly, Y_linear_test_pred_poly = predict_data_poly(linear_model_poly, X_linear_train, X_linear_test)

# Fit and predict for the convex dataset
convex_model_poly = fit_model_poly(X_convex_train, Y_convex_train, degree_convex)
Y_convex_train_pred_poly, Y_convex_test_pred_poly = predict_data_poly(convex_model_poly, X_convex_train, X_convex_test)

# Fit and predict for the trigonometric dataset
trigonometric_model_poly = fit_model_poly(X_tri_train, Y_tri_train, degree_tri)
Y_tri_train_pred_poly, Y_tri_test_pred_poly = predict_data_poly(trigonometric_model_poly, X_tri_train, X_tri_test)

# Set the variable to select the model to plot
selected_dataset = 'Trigonometric'  # Choose between 'Linear', 'Convex', 'Trigonometric'

# DO NOT MODIFY
# Function to plot dataset with regression line
def plot_dataset_with_regression(X_train, Y_train, X_test, Y_test, Y_train_pred, title):
    sorted_idx = np.argsort(X_train.ravel())
    plt.scatter(X_train, Y_train, color='blue', label='Train')
    plt.scatter(X_test, Y_test, color='orange', label='Test')
    plt.plot(X_train[sorted_idx], Y_train_pred[sorted_idx], color='green', label='Regression Line')
    plt.title(title)
    plt.legend()
    plt.show()

# Dictionary to store datasets and predictions
datasets = {
    'Linear': (X_linear_train, Y_linear_train, X_linear_test, Y_linear_test, Y_linear_train_pred_poly, 'Linear Dataset Regression'),
    'Convex': (X_convex_train, Y_convex_train, X_convex_test, Y_convex_test, Y_convex_train_pred_poly, 'Convex Dataset Regression'),
    'Trigonometric': (X_tri_train, Y_tri_train, X_tri_test, Y_tri_test, Y_tri_train_pred_poly, 'Trigonometric Dataset Regression')
}

# Extract selected dataset
X_train, Y_train, X_test, Y_test, Y_train_pred, title = datasets[selected_dataset]

# Plot the selected dataset with regression line
plot_dataset_with_regression(X_train, Y_train, X_test, Y_test, Y_train_pred, title)

# DO NOT MODIFY

# Calculate for all datasets
mse_linear_train_poly, mse_linear_test_poly, r2_linear_train_poly, r2_linear_test_poly = evaluate_model(
    Y_linear_train, Y_linear_train_pred_poly, Y_linear_test, Y_linear_test_pred_poly)

mse_convex_train_poly, mse_convex_test_poly, r2_convex_train_poly, r2_convex_test_poly = evaluate_model(
    Y_convex_train, Y_convex_train_pred_poly, Y_convex_test, Y_convex_test_pred_poly)

mse_tri_train_poly, mse_tri_test_poly, r2_tri_train_poly, r2_tri_test_poly = evaluate_model(
    Y_tri_train, Y_tri_train_pred_poly, Y_tri_test, Y_tri_test_pred_poly)

# Print the MSE and R^2 values for both training and testing sets for each dataset with polynomial features
print("Linear Dataset with Polynomial Features:")
print(f"Train - MSE: {mse_linear_train_poly:.4f}, R^2: {r2_linear_train_poly:.4f}")
print(f"Test - MSE: {mse_linear_test_poly:.4f}, R^2: {r2_linear_test_poly:.4f}\n")

print("Convex Dataset with Polynomial Features:")
print(f"Train - MSE: {mse_convex_train_poly:.4f}, R^2: {r2_convex_train_poly:.4f}")
print(f"Test - MSE: {mse_convex_test_poly:.4f}, R^2: {r2_convex_test_poly:.4f}\n")

print("Trigonometric Dataset with Polynomial Features:")
print(f"Train - MSE: {mse_tri_train_poly:.4f}, R^2: {r2_tri_train_poly:.4f}")
print(f"Test - MSE: {mse_tri_test_poly:.4f}, R^2: {r2_tri_test_poly:.4f}\n")

class SimpleLinearRegression:
    def __init__(self, iterations=1000, learning_rate=0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        # YOUR CODE HERE
        self.m = None
        self.b = None


    def fit(self, X, Y):
        # YOUR CODE HERE
        n = len(X)
        self.m = 0.0
        self.b = 0.0

        for _ in range(self.iterations):
            y_pred = (self.m * X) + self.b

            dm = (2 / n) * np.sum((y_pred - Y) * X)
            db = (2 / n) * np.sum(y_pred - Y)


            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

    def predict(self, X):
        # YOUR CODE HERE
        return (self.m * X) + self.b

### DO NOT MODIFY
def test_linear_regression(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    metric = max(Y_test)


    relative_error = np.mean(np.abs((Y_test - predictions)/metric))
    return relative_error < 0.05

datasets = [
   (np.arange(0, 50), np.arange(0, 50)*2),
    (np.arange(10, 50), np.arange(10, 50)**2),
   (np.array([0, 0, 1, 1, 0.25, 0.25, 0.75, 0.75, 0.5]), np.array([0, 1, 1, 0, 0.25, 0.75, 0.75, 0.25, 0.5])),
    (np.arange(40, 80), np.arange(40,80)*5 + np.random.normal(0, 15, 40).clip(-10, 10))
]

datasets = [split_data(X, Y, 0.2) for X, Y in datasets]

# Visualize regression
def visualize_datasets(datasets, titles):
    plt.figure(figsize=(10, 10))
    for idx, dataset in enumerate(datasets):
        X_train, X_test, Y_train, Y_test = dataset

        model = SimpleLinearRegression(iterations=6000, learning_rate=1e-4)
        model.fit(X_train, Y_train)
        Y_train_pred = model.predict(X_train)

        sorted_idx = np.argsort(X_train.ravel())
        plt.subplot(2, 2, idx + 1)
        plt.scatter(X_train, Y_train, color='blue', label='Train')
        plt.scatter(X_test, Y_test, color='orange', label='Test')
        plt.plot(X_train[sorted_idx], Y_train_pred[sorted_idx], color='green', label='Regression Line')
        plt.title(titles[idx])
        plt.legend()

    plt.tight_layout()
    plt.show()

titles = ["Linear", "Cubic", "XOR", "Linear + Noise"]
visualize_datasets(datasets, titles)

# Test each dataset
results = [test_linear_regression(SimpleLinearRegression(iterations=10000, learning_rate=1e-4), *dataset) for dataset in datasets]
answers = [True, False, False, True]
if results == answers:
  print("Good job!")
else:
  print("Wrong Implmentation! Your linear regression model needs more testing!")
  assert(False)

### DO NOT MODIFY

### DO NOT MODIFY: Dataset Preparation
num_samples = 800
X = np.linspace(0, 24, num_samples)
x_norm = (X - 12) / 12
Y = (200 - 500 * x_norm**4 + 1800 * x_norm**2 + 500 * x_norm**3) * 50  #+ np.random.normal(0, 500.0, num_samples).clip(-2000, 2000)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

print(X.shape, Y.shape)

class LinearRegression:
    def __init__(self, iterations=1000, learning_rate=0.1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None  # will be initialized in fit
        self.bias = 0.0

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)
        n, d = X.shape

        ### YOUR CODE HERE
        self.weights = np.zeros((d, 1))

        for iter in range(self.iterations):
            ### YOUR CODE HERE
            Y_pred = np.dot(X, self.weights) + self.bias
            
            grad_w = (2/n) * np.dot(X.T, Y_pred - Y)
            grad_b = (2/n) * np.sum(Y_pred - Y)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.dot(X, self.weights) + self.bias


### YOUR CODE HERE
model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train.reshape(-1, 1), Y_train)

### DO NOT MODIFY
traffic_at_8am = model.predict(np.array([[8]]))
traffic_at_9pm = model.predict(np.array([[21]]))

print("Traffic at 8 AM: ", traffic_at_8am.item())
print("Traffic at 9 PM: ", traffic_at_9pm.item())

def plot_regression_line(model, X, Y):
    # Predict values using the fitted model
    X = X.reshape(-1, 1)
    Y_pred = model.predict(X)

    # Plot the actual data and the fitted regression line
    plt.scatter(X, Y, color='blue', label='Actual Data')
    plt.plot(X, Y_pred, color='red', label='Fitted Line')
    plt.title('Traffic Volume Prediction Using Regression')
    plt.xlabel('Time of Day (hours)')
    plt.ylabel('Traffic Volume (vehicles/hour)')
    plt.legend()
    plt.show()

plot_regression_line(model, X_train, Y_train)
print("Predicted traffic volume at 8 AM:", traffic_at_8am)
print("Predicted traffic volume at 9 PM:", traffic_at_9pm)

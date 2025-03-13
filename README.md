# Housing Price Prediction Model

## Overview
This project implements a simple linear regression model to predict housing prices based on house size. The model uses the equation `price = w * size + b` where `w` (weight) and `b` (bias) are the parameters of the model.

## Features
- Data loading and preprocessing
- Cost function implementation for error calculation
- Visualization utilities for model predictions

## File Description
The main script contains the following components:

### Data Loading
```python
data = np.loadtxt('data.txt', delimiter=',') 
x_train = data[:,0]  # House sizes (in 1000 sqft)
y_train = data[:,1]  # House prices (in 1000s of dollars)
```

### Cost Function
The `compute_cost` function calculates the mean squared error between predicted values and actual values:
```python
def compute_cost(x, y, w, b):
    total_error = 0
    m = len(x)

    for i in range(m): 
        f_wb = w * x[i] + b 
        error = (f_wb - y[i])**2
        total_error = total_error + error

    cost = (1/2*m) * total_error
    return cost
```

### Visualization Function
The `plt_intuition` function visualizes the actual data points and the model's predictions:
```python
def plt_intuition(x, y, w, b=100):
    plt.scatter(x, y, marker='x', c='r', label="Actual Values")
    
    # Generate predicted values
    x_model = np.linspace(min(x), max(x), 100)
    y_model = w * x_model + b
    plt.plot(x_model, y_model, c='b', label=f"Prediction (y = {w}x + {b})")
    
    plt.title("Housing Prices")
    plt.ylabel('Price (in 1000s of dollars)')
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()
```

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Usage
1. Ensure you have a file named `data.txt` in the same directory with comma-separated values (size,price).
2. Run the script to see model predictions with default parameters.
3. Modify the `w` and `b` values in the `plt_intuition` function call to experiment with different models.

## Example
```python
# Calculate cost with specific parameters
print("Cost:", compute_cost(x_train, y_train, 50, 100))

# Visualize model predictions with custom parameters
plt_intuition(x_train, y_train, w=200, b=100)

# Visualize with default parameters
plt_intuition(x_train, y_train)
```

## Notes
- The cost function uses a factor of 1/(2*m) for the mean squared error, which is a common convention in gradient descent implementations.
- The visualization function creates a smooth prediction line across the range of input features.

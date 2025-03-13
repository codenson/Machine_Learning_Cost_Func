import numpy as np
import matplotlib.pyplot as plt
"""
This file implements a simple linear regression model to predict housing prices based on house size.
It includes functions for computing the cost (error) of the model and visualizing the predictions.

The main components are:
- Data loading and preprocessing
- Cost function implementation
- Visualization function for model predictions
"""


# a model which can predict housing prices given the size of the house.
data = np.loadtxt('data.txt', delimiter=',') 
x_train = data[:,0]
y_train = data[:,1]

# function to calculate the cost of the model. It calculates the error between the predicted value and the actual value.
#  f_wb is the predicted value for each feature x and b is the bias term.
def compute_cost (x, y, w, b):
    total_error = 0
    m  = len(x)

    for i in range (m): 
        f_wb = w *x[i]+b 
        error  = (f_wb - y[i])**2
        total_error= total_error + error

    cost  = (1/2*m) * total_error
    return cost
print("val : " , compute_cost(x_train, y_train, 50, 100))

### Visualization function for model predictions
def plt_intuition(x, y, w, b=100):
    plt.scatter(x, y, marker='x', c='r', label="Actual Values")
    
    # Generate predicted values
    x_model = np.linspace(min(x), max(x), 100)  # Create a smooth line
    y_model = w * x_model + b
    plt.plot(x_model, y_model, c='b', label=f"Prediction (y = {w}x + {b})")
    
    plt.title("Housing Prices")
    plt.ylabel('Price (in 1000s of dollars)')
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()

# Call function with w and b values
plt_intuition(x_train, y_train, w=200, b=100)

    
plt_intuition(x_train,y_train)

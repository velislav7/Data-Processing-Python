import numpy as np

my_array = np.array([6,7,7,9,43,10])
result_addition = my_array + 10
result_multiplication = my_array * 2

my_list = [1,2,3,4,5]
result_addition1 = [x + 10 for x in my_list]
result_multiplication1 = [x * 2 for x in my_list]

# print(result_addition)
# print(result_multiplication)
# print(result_addition1)
# print(result_multiplication1)

# # how many rows and columns (dimensions)
# print("Shape:", my_array.shape)
# # total elements
# print("Size:", my_array.size)
# # the type of data stored in the array (integer, float, etc.).
# print("Data Type:", my_array.dtype)

# 2D array
matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
# Accessing elements in 2D array
element_2_2 = matrix[1,1] #row 1 column 1

# print(element_2_2)

my_array2 = np.array([1,2,3,4,44,10])

result_addition = my_array + my_array2
result_subtraction = my_array - my_array2
result_multiplication = my_array * my_array2
result_division = my_array2 / my_array

# print("Addition Result:", result_addition)
# print("Subtraction Result:", result_subtraction)
# print("Multiplication Result:", result_multiplication)
# print("Division Result:", result_division)
# print(my_array2 + 10)

# Using universal functions (ufuncs)
squared = np.square(my_array2)
square_root = np.sqrt(my_array2)
exponential = np.exp(my_array2)
# print(squared)
# print(square_root)
# print(exponential)

# Aggregation along axes
row_sum = np.sum(matrix, axis=1)
column_mean = np.mean(matrix, axis=0)
# print("Row Sum:", row_sum)
# print("Column Mean:", column_mean)

matrix_a = np.array([[1,2,3], [4,5,6]])
matrix_b = ([[2,1,-1], [5,2,4], [7,3,2]])

# The np.dot() function in NumPy is used for matrix multiplication (dot product) of two arrays. 
# The behavior of np.dot() depends on the dimensions of the input arrays:
# If both arguments are 1-D arrays, it performs the dot product (scalar product).
# If both arguments are 2-D arrays, it performs matrix multiplication.
# If either argument is an N-dimensional array (where N > 2), it treats it as a stack of matrices in a higher-dimensional space, and the behavior depends on the dimensions

result_matrix = np.dot(matrix_a, matrix_b)
# print(result_matrix)

# Identity matrix and inverse
identity_matrix = np.eye(3)  # 3x3 identity matrix
inverse_matrix = np.linalg.inv(matrix_b)
# print(identity_matrix)
# print(inverse_matrix)

# Coefficient Matrix 
A = np.array([[2,3], [4,5]])    

# Constants vector 
B = np.array([8,2])

# Check if A is invertible 
if np.linalg.det(A) != 0:
    # Calculate the inverse of A
    A_inverse = np.linalg.inv(A)
    
    # Solve for x
    x = np.dot(A_inverse, B)
    # print(x)
else:
    print("The coefficient matrix is singular. No unique solution")

# function, which is a more efficient way to solve systems of linear equations.
# Using np.linalg.solve()
x_solution = np.linalg.solve(A, B)
# print(x_solution)

#Eigenvalues and Eigenvectors with NumPy
#Eigenvalues (λλ) and eigenvectors (vv) are fundamental concepts in linear algebra. For a square matrix AA, if there exists a scalar λλ and a non-zero vector vv such that Av=λvAv=λv, then λλ is an eigenvalue of AA, and vv is the corresponding eigenvector.
matrix_c = np.array([[4, -2], [1, 1]])
eigenvalues, eigenvectors = np.linalg.eig(matrix_c)
# print(eigenvalues)
# print(eigenvectors)

# Singular Value Decomposition (SVD) with NumPy
# SVD is a factorization of a matrix AA into three other matrices: U, Σ, and V^T. For a matrix A of size m×n:
# A=UΣV^T

#     U is an m×m orthogonal matrix.
#     Σ is an m×n diagonal matrix with non-negative real numbers on the diagonal (singular values).
#     V^T is the transpose of an n×n orthogonal matrix.

# Computing SVD
U, Sigma, V_transpose = np.linalg.svd(matrix_a)

# print("Matrix U:")
# print(U)

# print("\nMatrix Sigma:")
# print(Sigma)

# print("\nMatrix V^T:")
# print(V_transpose)

# Reconstructing A from SVD components
# Computing SVD

# Ensure Sigma is a square matrix with correct dimensions
Sigma_matrix = np.diag(Sigma)

# Reshape Sigma to ensure it's a 2D array
Sigma_matrix_reshaped = np.zeros_like(matrix_a, dtype=float)
np.fill_diagonal(Sigma_matrix_reshaped, Sigma)

# Reconstructing a from SVD components
reconstructed_a = U @ Sigma_matrix_reshaped @ V_transpose

# print("Original Matrix A:")
# print(matrix_a)

# print("\nReconstructed Matrix A:")
# print(reconstructed_a)


# Principal Component Analysis (PCA) with NumPy
# PCA is a dimensionality reduction technique that aims to transform the data into a new coordinate system where the features are uncorrelated, and the majority of the variance is captured by a smaller number of components called principal components.
# 2. Steps for PCA:

#     Standardize the Data:
#     Standardize the features by subtracting the mean and scaling to unit variance.

#     Compute Covariance Matrix:
#     Compute the covariance matrix of the standardized data.

#     Compute Eigenvectors and Eigenvalues:
#     Perform eigenvalue decomposition on the covariance matrix to obtain the eigenvectors and eigenvalues.

#     Sort Eigenvectors:
#     Sort the eigenvectors by their corresponding eigenvalues in descending order.

#     Select Principal Components:
#     Choose the top kk eigenvectors to form the projection matrix (WW).

#     Transform the Data:
#     Multiply the original data by the projection matrix to obtain the transformed data.

#create a dataset
np.random.seed(42)
data = np.random.rand(3,5) #3 features, 5 samples

#standardize the data
mean = np.mean(data, axis=1, keepdims=True)
std_dev = np.std(data, axis=1, keepdims=True)
standardized_data = (data - mean) / std_dev

# Compute covariance matrix
covariance_matrix = np.cov(standardized_data)

# Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvectors by eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select top k eigenvectors (e.g., k=2)
k = 2
projection_matrix = sorted_eigenvectors[:, :k]

# Transform the data
transformed_data = np.dot(projection_matrix.T, standardized_data)

# print("Original Data:")
# print(data)

# print("\nTransformed Data:")
# print(transformed_data)


# Optimization with NumPy
# Optimization involves finding the best solution to a problem, 
# typically the maximum or minimum of a function. In many cases, optimization problems can be expressed as mathematical equations, 
# and finding the optimal solution requires minimizing or maximizing these equations.
# NumPy provides tools for optimization, and one commonly used function is numpy.optimize.minimize()

from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2

initial_guess = [1,1]

# Perform an optimization
result = minimize(objective_function, initial_guess)

# print(result.x)

# Constraints (ограничения)
# Define objective function with constraints
def constrained_objective(x):
    return x[0]**2 + x[1]**2

# Define constraint function
def constraint_function(x):
    return x[0] + x[1] - 1

# Define the constraint type (inequality)
constraint = {'type': 'ineq', 'fun': constraint_function}

# Perform constrained optimization
constrained_result = minimize(constrained_objective, initial_guess, constraints=constraint)

# print("Constrained optimal solution:")
# print(constrained_result.x)


# NumPy in Data Manipulation and Analysis
# NumPy provides a robust framework for working with multidimensional arrays, making it a powerful tool for data manipulation and analysis.# Creating a 1D array

# Creating NumPy Arrays:
array_1d = np.array([1, 2, 3])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
# Creating an array with zeros
zeros_array = np.zeros((3, 4))
# Creating an array with ones
ones_array = np.ones((2, 2))
# Creating a range of values
range_array = np.arange(0, 10, 2)

# NumPy Operations for Data Manipulation:
# Indexing and slicing
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
element = array_2d[1, 2]
sliced_row = array_2d[0, :]
# Aggregation and Statistical Operations:
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
mean_value = np.mean(array_2d)
sum_axis_0 = np.sum(array_2d, axis=0)
# Boolean Indexing:
array_1d = np.array([1, 2, 3, 4, 5])
even_numbers = array_1d[array_1d % 2 == 0]

# NumPy in Data Analysis
# Loading Data with NumPy:
# data = np.loadtxt('data.txt', delimiter=',')
# Data Reshaping:
array_1d = np.arange(1, 7)
reshaped_array = array_1d.reshape((2, 3))


# NumPy in Machine Learning
# Loading Data:
# NumPy is often used in conjunction with Pandas for loading and handling datasets. 
# Pandas provides DataFrames, and NumPy facilitates the conversion of DataFrame columns into NumPy arrays

import pandas as pd 
import matplotlib.pyplot as plt

# Loading data from a CSV file using Pandas
data = pd.read_csv('dataset.csv')

# Converting a DataFrame column to a NumPy array
target = data['target'].values

# Data Preprocessing:
# Before feeding data into machine learning models, preprocessing is essential. 
# NumPy integrates seamlessly with scikit-learn for tasks like feature scaling.

from sklearn.preprocessing import StandardScaler

# Standardizing features using scikit-learn and NumPy
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# NumPy in Model Training:
# example with synthetic data for a linear regression problem
# Generating synthetic data
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualizing the data
plt.scatter(X, y, alpha=0.7)
plt.title('Synthetic Data for Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Adding a bias term to features
X_b = np.c_[np.ones((100, 1)), X]

# Initializing weights and biases
weights = np.zeros((2, 1))

# Setting hyperparameters
learning_rate = 0.01
num_epochs = 1000

# Training the model using gradient descent
for epoch in range(num_epochs):
    predictions = np.dot(X_b, weights)
    mse_loss = np.mean((predictions - y) ** 2)

    # Backpropagation
    gradients = 2/100 * X_b.T.dot(predictions - y)
    weights -= learning_rate * gradients

# Visualizing the trained model
plt.scatter(X, y, alpha=0.7, label='Original Data')
plt.plot(X, predictions, color='red', label='Linear Regression')
plt.title('Linear Regression with Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
# Assuming you have a dataset with features and targets
# features = np.array([[feature1, feature2], [feature3, feature4], ...])
# targets = np.array([target1, target2, ...])
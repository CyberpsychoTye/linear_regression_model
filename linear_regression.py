from sklearn import datasets, model_selection
from sklearn import linear_model, metrics
import numpy as np
from typing import Annotated
import matplotlib.pyplot as plt


X, y = datasets.fetch_california_housing(return_X_y = True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 42)
X_validate, X_test, y_validate, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5, random_state = 40)

class LinearRegression:
    def __init__(self):
        self.random_number_generator = np.random.default_rng(169365102679363528484982587533432496514)
        self.W = self.random_number_generator.normal(scale = 0.5, size = (len(X[1]), 1))
        self.b = self.random_number_generator.normal(scale = 1, size = 1)
    
    def __call__(self, X:np.ndarray):
        y = np.dot(X, self.W) + self.b
        return y
    
    def update_parameters(self, new_W:Annotated[np.ndarray, "shape = self.W.shape"], new_b:Annotated[np.ndarray, "shape = self.b.shape"]):
        self.W = new_W
        self.b = new_b

    def cost(self, predicted_y:np.ndarray, actual_y:np.ndarray) -> float:
        array_of_errors = (predicted_y - actual_y) ** 2
        mean_squared_error = np.mean(array_of_errors)
        return mean_squared_error
    
    def parameters_optimiser(self, X:np.ndarray, y:np.ndarray):
        bias_matrix = np.ones((X.shape[0], 1))
        X_with_bias = np.hstack((bias_matrix, X))
        optimal_parameters = np.matmul(np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)), np.matmul(X_with_bias.T, y))
        return optimal_parameters[1:], optimal_parameters[0]
    
class SKLinearRegression:
    def __init__(self):
        self.regression_instance = linear_model.LinearRegression(n_jobs = -1)
        self.model = None

    def __call__(self, X_train:np.ndarray, y_train:np.ndarray):
        self.model = self.regression_instance.fit(X_train, y_train)
        return "Your model has been trained on the data you provided."
    
    def predict(self, X_test:Annotated[np.ndarray, "shape(n_samples, n_features)"]):
        if self.model == None:
            raise NotImplementedError("You need to train a model by calling your instance with training data first!")
        prediction = self.model.predict(X_test)
        return prediction
    
    def mse_evaluation(self, predicted_y, actual_y):
        mean_squared_error = metrics.mean_squared_error(actual_y, predicted_y)
        return mean_squared_error



model = LinearRegression()
optimised_weights, optimised_bias = model.parameters_optimiser(X_train, y_train)
model.update_parameters(optimised_weights, optimised_bias)
prediction = model(X)
cost = model.cost(prediction, y)
print(cost)
print(prediction[:8])
print("\n")
sk_model = SKLinearRegression()
sk_model(X_train, y_train)
sk_prediction = sk_model.predict(X)
sk_cost = sk_model.mse_evaluation(sk_prediction, y)
print(sk_cost)
print(sk_prediction[:8])
print("\n")
print(y[:8])

# fig, ax = plt.subplots()
# ax.scatter(np.arange(11),prediction[:11], c = "c", marker = "x", label = "predicted_y")
# ax.scatter(np.arange(11),sk_prediction[:11], c = "m", marker = "o", label = "sk_predicted_y" )
# ax.scatter(np.arange(11), y[:11], c = "g", marker = ".", label = "actual_y")
# ax.legend()
# ax.set_xlabel("i-th sample")
# ax.set_ylabel("values")
# plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[2], [4], [6], [8]])  # Study hours
y = np.array([20, 40, 60, 80])      # Marks

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 5 hours of study
print("Predicted marks:", model.predict([[5]]))

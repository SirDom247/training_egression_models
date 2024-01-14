# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Identify and remove non-numeric values
unique_values = np.unique(target)
non_numeric_values = [value for value in unique_values if not isinstance(value, (int, float))]
if non_numeric_values:
    print("Non-numeric values found in target:", non_numeric_values)
    # Remove rows with non-numeric values
    non_numeric_indices = [idx for idx, value in enumerate(target) if not isinstance(value, (int, float))]
    data = np.delete(data, non_numeric_indices, axis=0)
    target = np.delete(target, non_numeric_indices)

# Check if there are any remaining non-numeric values in target
remaining_non_numeric_values = [value for value in np.unique(target) if not isinstance(value, (int, float))]
if remaining_non_numeric_values:
    print("Remaining non-numeric values in target:", remaining_non_numeric_values)
    # Handle the remaining non-numeric values or remove them

# Check if there are enough samples for train-test split
if len(data) > 0:
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Linear Regression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    linear_pred = linear_reg.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_pred)
    linear_r2 = r2_score(y_test, linear_pred)

    # Decision Tree Regression
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_train, y_train)
    tree_pred = tree_reg.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_pred)
    tree_r2 = r2_score(y_test, tree_pred)

    # Neural Network Regression
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    nn_pred = model.predict(X_test).flatten()
    nn_mse = mean_squared_error(y_test, nn_pred)
    nn_r2 = r2_score(y_test, nn_pred)

    # Scatter Matrix
    #numerical_features = pd.DataFrame(data)
    #sns.set(style="ticks")
    #scatter_matrix = sns.pairplot(numerical_features)
    #plt.show()

    # Report Writing
    print("\nSummary of Findings:")
    print("\nLinear Regression:")
    print("Mean Squared Error:", linear_mse)
    print("R-squared Score:", linear_r2)

    print("\nDecision Tree Regression:")
    print("Mean Squared Error:", tree_mse)
    print("R-squared Score:", tree_r2)

    print("\nNeural Network Regression:")
    print("Mean Squared Error:", nn_mse)
    print("R-squared Score:", nn_r2)

    # Additional Insights (if any)
    print("\nAdditional Insights:")
    # Add any additional insights or observations based on your analysis

    # Conclusion
    print("\nConclusion:")
    # Summarize the overall performance and any conclusions drawn from the analysis

else:
    print("No valid samples remaining after removing non-numeric values.")

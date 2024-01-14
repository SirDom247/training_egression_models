import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
# Add the target variable to the DataFrame
X['MEDV'] = california_housing.target

# Select numerical features
numerical_features = X.select_dtypes(include=[float])

# Create a scatter matrix
sns.set(style="ticks")
sns.pairplot(numerical_features)
plt.show()

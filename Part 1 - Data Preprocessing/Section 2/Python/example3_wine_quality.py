# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
# Note: The delimiter for this dataset is typically a semicolon (';')
wine_df = pd.read_csv('winequality-red.csv', delimiter=';')

# Separate features and target
X = wine_df.drop('quality', axis=1)  # 'quality' is the target variable
y = wine_df['quality']

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the StandardScaler class
sc = StandardScaler()
 
# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform(X_train)

# Apply the transform to the test set
X_test = sc.transform(X_test)

# Print the scaled training and test datasets
print("Scaled Training Set:")
print(X_train)
print("\nScaled Test Set:")
print(X_test) 
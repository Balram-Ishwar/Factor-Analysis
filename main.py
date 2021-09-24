

# Factor Analysis with Bartlett's test and KMO test


# Importing the dataset

from sklearn.datasets import load_diabetes
diabetes=load_diabetes()


# Displaying the details of the dataset

print("The description of the data set is: \n", diabetes.DESCR)
x= diabetes.data
y=diabetes.target
print("Dimansion of independent variables:",x.shape)

#Applying Bartlett's test of sphericity for determining adequency.
#importing necessary libraries

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chisquare,p_value = calculate_bartlett_sphericity(x)
print("Chi square value of Bartlett test: ",chisquare.round(3))
print("p value of Bartlett test: ", p_value.round(3))

# Applying KMO test for determining adequacy

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_value, kmo_model= calculate_kmo(x)
print("KMO Model: ", kmo_model.round(3))
print("KMO values: ", kmo_value.round(3))

# create factor analysis object and perform factor analysis

fa = FactorAnalyzer()
fa.fit(x)

# Determining Eigenvalues

eigen_value, value= fa.get_eigenvalues()
print("Eigen Values are: \n",eigen_value.round(3))
print("Values are: \n", value.round(3))

# Create scree plot for determining optimum number of factors

import matplotlib.pyplot as plt
plt.scatter(range(1,x.shape[1]+1),eigen_value)
plt.plot(range(1,x.shape[1]+1),eigen_value)
plt.title('Scree Plot for diabetes Dataset')
plt.xlabel('Number of factors')
plt.ylabel('Eigen Values')
plt.grid()
plt.show()

''' Note:- We can observed that from Scree plot that the eigenvalues are greater then 1 for only 3 variables. It means we 
need to perform FA considering only 3 factors(unobserved Variables).
'''

# Performing factor analysis for optimum number (3) of factors.

fa = FactorAnalyzer()
fa.fit(x,3)
print("Factor loadings are: \n", fa.loadings_.round(3))

# Display variance of all factors.
print("Variances of each factor :\n", fa.get_factor_variance())
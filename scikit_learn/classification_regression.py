## https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

import numpy as np
from sklearn import datasets


print("-- Sample 1: KNN (k nearest neighbors) classification example ---")
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print(np.unique(iris_y))

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
# Create and fit a nearest-neighbor classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 
##KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
#           weights='uniform')
##
vPredict = knn.predict(iris_X_test)
print("Predict:", vPredict)
#array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
print("Target: ", iris_y_test)
#array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])

print("\n-- Sample 2: linear regression example ---")
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

from sklearn import linear_model
regr = linear_model.LinearRegression()
vFit = regr.fit(diabetes_X_train, diabetes_y_train)
print("Linear reg fit:", vFit)
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
print("Coefficients: ", regr.coef_)
#[   0.30349955 -237.63931533  510.53060544  327.73698041 -814.13170937
#  492.81458798  102.84845219  184.60648906  743.51961675   76.09517222]

# The mean square error
vMean = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2)
print("Mean:", vMean)
#...                                                   
#2004.56760268...

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
vScore = regr.score(diabetes_X_test, diabetes_y_test)
print("Score:", vScore)
#0.5850753022690..
#vPredict = regr.predict(diabetes_y_train)
#print("Regr Predict: ", vPredict)
#print("Target: ", diabetes_y_test)


print("\nSample 3. Support vector machines (SVMs), linear SVM example")
from sklearn import svm
svc = svm.SVC(kernel='linear')          # Linear SVM
svc = svm.SVC(kernel='poly', degree=3)  # Polynomial SVM
svc = svm.SVC(kernel='rbf')             # RBF kernel (Radial Basis Function)
vFit = svc.fit(iris_X_train, iris_y_train) 
print("SVM fit:", vFit)

vPredict = svc.predict(iris_X_test)
print("SVC Predict:", vPredict)
print("Target: ", iris_y_test)


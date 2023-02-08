from sklearn import linear_model
from sklearn.linear_model import LinearRegression

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

polynomial_features = PolynomialFeatures(degree=3,
                                         include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
pipeline.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(linear_regression.coef_, linear_regression.intercept_)

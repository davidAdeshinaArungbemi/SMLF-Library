# SMLF - Simple Machine Learning Framework

A basic machine learning framework for regression and binary classification tasks.

## ML Algorithms Implemented

1. Linear Regression<br>
    a. Ordinary Least Squares Method<br>
    b. Gradient descent

2. Logistic Regression

## Dependencies
1. [Ourdataframe](https://github.com/davidAdeshinaArungbemi/OurDataframe) by davidAdeshinaArungbemi(for data manipulation)

2. [Eigen](https://gitlab.com/libeigen/eigen)(for linear algebra and matrix operations)


## Examples

### Linear Regression Ordinary Least Squares OLS

```cpp
#include "LinearRegression.hpp"
using namespace SMLF;
int main(int argc, char const *argv[])
{
    auto a = ODf::Table("/Users/david/SMLF-Library/OurDataframe/DataSource/winequality-red.csv");//use your full path
    auto y_train_table = a.SelectColumns((ODf::Vec_UInt){a.ColumnSize() - 1});
    auto X_train_table = a.ColumnCut(0, a.ColumnSize() - 1);

    auto X_train = X_train_table.ToMatrix();
    auto y_train = y_train_table.ToMatrix();

    LinearRegression::OLS lr;
    lr.train(X_train, y_train);
    auto prediction = lr.predict(X_train);
    return 0;
}
```

### Linear Regression with Gradient Descent


#include "LinearRegression.hpp"
using namespace SMLF;
int main(int argc, char const *argv[])
{
    auto a = ODf::Table("/Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/DataSource/fakeData.csv");
    auto y_train_table = a.SelectColumns((ODf::Vec_UInt){a.ColumnSize() - 1});
    auto X_train_table = a.ColumnCut(0, a.ColumnSize() - 1);

    auto X_train = X_train_table.ToMatrix();
    auto y_train = y_train_table.ToMatrix();

    LinearRegression::OLS lr;
    lr.train(X_train, y_train);
    auto prediction = lr.predict(X_train);
    std::cout << prediction << std::endl;
    return 0;
}

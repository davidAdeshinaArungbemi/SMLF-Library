#include "LinearRegression.hpp"
using namespace SMLF;

int main(int argc, char const *argv[])
{
    auto a = ODf::Table("/Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/DataSource/fakeData.csv");
    // std::cout << a;
    auto y_train = a.SelectColumns((ODf::Vec_UInt){a.ColumnSize() - 1});
    auto X_train = a.ColumnCut(0, a.ColumnSize() - 1);

    LinearRegression::GradientDescent lr(1e-2, 1000, true, 42, 0, 2);
    lr.train(X_train, y_train);
    std::cout << "Results:\n"
              << lr.predict(X_train) << std::endl;
    return 0;
}

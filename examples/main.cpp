#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
using namespace SMLF;

int main(int argc, char const *argv[])
{
    auto a = ODf::Table("/Users/davidadeshina/Documents/Git-projects/SMLF-Library/examples/framingham.csv");
    auto y_train = a.SelectColumns((ODf::Vec_UInt){a.ColumnSize() - 1});
    auto X_train = a.ColumnCut(0, a.ColumnSize() - 1);

    LinearRegression::GradientDescent lr(1e-7, 10000, true, 42, 0, 10000);
    lr.train(X_train, y_train);
    std::cout << "Results:\n"
              << lr.predict(X_train) << std::endl;

    return 0;
}

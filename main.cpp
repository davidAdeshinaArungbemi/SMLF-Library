#include "LinearRegression.hpp"
using namespace SMLF;
int main(int argc, char const *argv[])
{
    auto a = ODf::Table("/Users/davidadeshina/Documents/Git-projects/SMLF-Library/OurDataframe/DataSource/fakeData.csv");
    auto y_train_table = a.SelectColumns((ODf::Vec_UInt){a.ColumnSize() - 1});
    auto X_train_table = a.ColumnCut(0, a.ColumnSize() - 1);

    // std::cout << raw_matrix << std::endl;
    // LinearRegression::OLS lr;
    // auto X_train = raw_matrix.block()
    //                    lr.train();
    return 0;
}

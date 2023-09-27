// MSE
// MAE
// Accuracy
#ifndef MEASURES_DESC
#define MEASURES_DESC
#include "ODf.hpp"
namespace SMLF
{
    double mse(const Eigen::MatrixXd &prediction, const Eigen::MatrixXd &actual);
    double accuracy(const Eigen::MatrixXd &prediction, const Eigen::MatrixXd &actual);
}
#endif

#ifndef MEASURES_IMPL
#define MEASURES_IMPL
double SMLF::mse(const Eigen::MatrixXd &prediction, const Eigen::MatrixXd &actual)
{
    auto length = prediction.cols() * prediction.rows();
    Eigen::MatrixXd difference_mat = (prediction - actual);
    double *difference = difference_mat.data();

    for (size_t i = 0; i < length; i++)
    {
        difference[i] = difference[i] * difference[i]; // becomes squared difference
    }

    double sum = 0;

    for (size_t i = 0; i < length; i++)
    {
        sum += difference[i];
    }

    auto error = sum / prediction.rows() * 2;

    return sum;
}
#endif

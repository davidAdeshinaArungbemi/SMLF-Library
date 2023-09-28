#ifndef ACTIVATION_DESC
#define ACTIVATION_DESC
#include "ODf.hpp"
namespace SMLF
{
    Eigen::MatrixXd sigmoid_log(Eigen::MatrixXd mat);
}
#endif

#ifndef ACTIVATION_IMPL
#define ACTIVATION_IMPL
Eigen::MatrixXd SMLF::sigmoid_log(Eigen::MatrixXd mat)
{
    for (size_t i = 0; i < mat.rows(); i++)
    {
        for (size_t j = 0; j < mat.cols(); j++)
        {
            mat(i, j) = 1 / (1 + exp(-mat(i, j)));
        }
    }

    return mat;
}
#endif
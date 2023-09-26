#ifndef LINEAR_DESC
#define LINEAR_DESC
#include "ODf.hpp"
namespace SMLF
{
    namespace LinearRegression
    {
        class LinearRegression
        { // abstract class
        protected:
            Eigen::MatrixXd prediction;

        public:
            virtual void train(Eigen::MatrixXd X_train, Eigen::MatrixXd y_train) = 0;
            virtual Eigen::MatrixXd predict(Eigen::MatrixXd testingData) = 0;
        };

        class OLS : public LinearRegression
        {
        private:
            Eigen::MatrixXd hypothesisParameters;

        public:
            void train(Eigen::MatrixXd X_train, Eigen::MatrixXd y_train) override;
            Eigen::MatrixXd predict(Eigen::MatrixXd testingData) override;
        };

        class GradientDescent : public LinearRegression
        {
        };
    }
}

#endif

#ifndef LINEAR_IMPL
#define LINEAR_IMPL
void SMLF::LinearRegression::OLS::train(Eigen::MatrixXd X_train, Eigen::MatrixXd y_train)
{
    assert(X_train.rows() == y_train.rows());
    assert(y_train.cols() == 1);

    auto X_train_transpose = X_train.transpose();

    auto square_mat = (X_train_transpose * X_train);
    assert(square_mat.determinant() != 0 && "make use of gradient descent method");
    this->hypothesisParameters = square_mat.inverse() *
                                 X_train_transpose * y_train;
}

Eigen::MatrixXd SMLF::LinearRegression::OLS::predict(Eigen::MatrixXd testingData)
{
    assert(testingData.cols() == hypothesisParameters.rows());
    return testingData * hypothesisParameters;
}
#endif
#ifndef LINEAR_DESC
#define LINEAR_DESC
#include "ODf.hpp"
#include "Measures.hpp"
namespace SMLF
{
    namespace LinearRegression
    {
        class LinearRegression
        { // abstract class
        protected:
            Eigen::MatrixXd prediction;

        public:
            virtual void train(ODf::Table X_train, ODf::Table y_train) = 0;
            virtual Eigen::MatrixXd predict(ODf::Table X_test) = 0;
        };

        class OLS : public LinearRegression
        {
        private:
            Eigen::MatrixXd hypothesisParameters;

        public:
            void train(ODf::Table X_train, ODf::Table y_train) override;
            Eigen::MatrixXd predict(ODf::Table X_test) override;
        };

        class GradientDescent : public LinearRegression
        {
        private:
            double lr;
            double alpha;
            size_t random_state;
            size_t epochs;
            size_t callback_count;
            bool show_cost;
            Eigen::MatrixXd weights;

        public:
            GradientDescent(double lr = 0.001, size_t epochs = 100, bool show_cost = false, double random_state = 42, double alpha = 0, size_t callback_count = 100)
            {
                assert(epochs > 0);
                assert(lr != 0);
                assert(alpha >= 0 && alpha < 1);

                this->lr = lr;
                this->random_state = random_state;
                this->epochs = epochs;
                this->callback_count = callback_count;
                this->show_cost = show_cost;
                this->alpha = alpha;
            }

            void train(ODf::Table X_train, ODf::Table y_train) override;
            Eigen::MatrixXd predict(ODf::Table X_test) override;
        };
    }
}

#endif

#ifndef LINEAR_IMPL
#define LINEAR_IMPL
void SMLF::LinearRegression::OLS::train(ODf::Table X_train, ODf::Table y_train)
{
    assert(X_train.RowSize() == y_train.RowSize());
    assert(y_train.ColumnSize() == 1);

    auto X_train_raw = X_train.ToMatrix();
    auto y_train_raw = y_train.ToMatrix();

    auto X_train_raw_transpose = X_train_raw.transpose();

    auto square_mat = (X_train_raw_transpose * X_train_raw);
    assert(square_mat.determinant() != 0 && "make use of gradient descent method");
    this->hypothesisParameters = square_mat.inverse() *
                                 X_train_raw_transpose * y_train_raw;

    std::cout << hypothesisParameters << std::endl;
}

Eigen::MatrixXd SMLF::LinearRegression::OLS::predict(ODf::Table X_test)
{
    auto X_test_raw = X_test.ToMatrix();
    assert(X_test_raw.cols() == hypothesisParameters.rows());
    return X_test_raw * hypothesisParameters;
}

void SMLF::LinearRegression::GradientDescent::train(ODf::Table X_train, ODf::Table y_train)
{
    srand((unsigned)random_state);

    ODf::Table ones_column("1", X_train.RowSize(), 1);

    X_train = ODf::ColumnConcat(X_train, ones_column);

    auto X_train_raw = X_train.ToMatrix();
    auto y_train_raw = y_train.ToMatrix();

    this->weights.resize(X_train_raw.cols(), 1);
    this->weights.setRandom();

    ODf::VecDouble cost_history;
    std::vector<Eigen::MatrixXd> weight_history;

    Eigen::MatrixXd momentum(X_train_raw.cols(), 1);
    momentum.setConstant(0);

    for (size_t epoch = 0; epoch < this->epochs; epoch++)
    {
        auto gradient = (2.0 / X_train_raw.rows()) * X_train_raw.transpose() * (X_train_raw * this->weights - y_train_raw);
        if (this->alpha == 0)
        {
            this->weights -= (lr * gradient) + (alpha * momentum);
            momentum = gradient;
        }
        else
            this->weights -= (lr * gradient);

        if (this->show_cost)
            std::cout << SMLF::mse(X_train_raw * weights, y_train_raw) << std::endl;

        cost_history.push_back(SMLF::mse(X_train_raw * weights, y_train_raw));
        weight_history.push_back(weights);

        if (cost_history.size() == this->callback_count)
        {
            if (!(*(cost_history.end() - 1) < *cost_history.begin()))
            {
                weights = weight_history[0];
                break;
            }
            cost_history.clear();
            weight_history.clear();
        }
    }
}

Eigen::MatrixXd SMLF::LinearRegression::GradientDescent::predict(ODf::Table X_test)
{
    ODf::Table ones_column("1", X_test.RowSize(), 1);

    X_test = ODf::ColumnConcat(X_test, ones_column);

    auto X_test_raw = X_test.ToMatrix();

    assert(X_test_raw.cols() == this->weights.rows());

    return X_test_raw * this->weights;
}
#endif
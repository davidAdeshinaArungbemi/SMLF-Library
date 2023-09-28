#ifndef LOGISTIC_DESC
#define LOGISTIC_DESC
#include "ODf.hpp"
#include "Measures.hpp"
#include "activation.hpp"
namespace SMLF
{
    namespace LogisticRegression
    {
        class LogisticRegression
        { // abstract class
        protected:
            Eigen::MatrixXd prediction;

        public:
            virtual void train(ODf::Table X_train, ODf::Table y_train) = 0;
            virtual Eigen::MatrixXd predict(ODf::Table X_test) = 0;
        };

        class GradientDescent : public LogisticRegression
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

#ifndef LOGISTIC_IMPL
#define LOGISTIC_IMPL

void SMLF::LogisticRegression::GradientDescent::train(ODf::Table X_train, ODf::Table y_train)
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
        auto gradient = (1.0 / X_train_raw.rows()) * X_train_raw.transpose() *
                        (SMLF::sigmoid_log(X_train_raw * this->weights) -
                         y_train_raw);

        if (this->alpha == 0)
        {
            this->weights -= (lr * gradient) + (alpha * momentum);
            momentum = gradient;
        }

        else
            this->weights -= (lr * gradient);

        if (this->show_cost)
            std::cout << SMLF::mse(X_train_raw * weights, y_train_raw) << std::endl;

        // cost_history.push_back(SMLF::mse(X_train_raw * weights, y_train_raw));
        // weight_history.push_back(weights);

        // if (cost_history.size() == this->callback_count)
        // {
        //     if (!(*(cost_history.end() - 1) < *cost_history.begin()))
        //     {
        //         weights = weight_history[0];
        //         break;
        //     }
        //     cost_history.clear();
        //     weight_history.clear();
        // }
    }
}

Eigen::MatrixXd SMLF::LogisticRegression::GradientDescent::predict(ODf::Table X_test)
{
    ODf::Table ones_column("1", X_test.RowSize(), 1);

    X_test = ODf::ColumnConcat(X_test, ones_column);

    auto X_test_raw = X_test.ToMatrix();

    assert(X_test_raw.cols() == this->weights.rows());

    std::cout << SMLF::sigmoid_log(X_test_raw * this->weights) << std::endl;

    return X_test_raw * this->weights;
}
#endif
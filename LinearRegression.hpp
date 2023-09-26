#ifndef LINEAR_DESC
#define LINEAR_DESC

namespace SMLF
{
    namespace LinearRegression
    {
        class LinearRegression
        { // abstract class
        protected:
            MLA::DataFrame prediction;

        public:
            LinearRegression() = default;
            virtual void train(MLA::DataFrame trainingData,
                               long target_column = -1) = 0;
            virtual void predict(MLA::DataFrame testingData) = 0;
            virtual MLA::DataFrame getPrediction() = 0;
        };
    }
}
#endif

#ifndef LINEAR_IMPL
#define LINEAR_IMPL

#endif
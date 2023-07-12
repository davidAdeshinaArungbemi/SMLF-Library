#ifndef LINEAR_DESC
#define LINEAR_DESC
#include "../../DataFrame/DataFrame.hpp"

namespace MLA {
namespace Regression {
namespace LinearRegression {
class LinearRegression { //abstract class
protected:
	MLA::DataFrame prediction;

public:
	LinearRegression() = default;
	virtual void train(MLA::DataFrame trainingData,
			long target_column = -1) = 0;
	virtual void predict(MLA::DataFrame testingData) = 0;
	virtual MLA::DataFrame getPrediction() = 0;
};

class OLS: public LinearRegression { //OLS for 10,000 features or less
private:
	MLA::DataFrame hypothesisParameters;
public:
	OLS() = default;
	void train(MLA::DataFrame trainingData, long target_column = -1) override;
	void predict(MLA::DataFrame testingData) override;
	MLA::DataFrame getPrediction() override;
};

class LinearRegressionWithGradient: public LinearRegression {

};

}
}
}
#endif

#ifndef LINEAR_IMPLEMENTATION
#define LINEAR_IMPLEMENTATION
#include "LinearRegression.hpp"

void MLA::Regression::LinearRegression::OLS::train(MLA::DataFrame trainingData,
		long target_column) {
	assert(
			target_column != trainingData.colSize() - 1
					&& "If last column is already target do not bother to set the value");
	assert(
			target_column > -2
					&& "value of -1 indicating last column is allowable, but it must not be less");

	assert(
			target_column < (long )trainingData.colSize()
					&& "Cannot exceed number of columns"); //if not casted to long, assertion fails, because one supports negative and one doesn't

//	let last column be target(use column swapping)
	if (target_column != -1) {
		trainingData.swapColumns(target_column, trainingData.colSize() - 1);
	}

	MLA::DataFrame y_train( // @suppress("Invalid arguments")
			trainingData.locView(0, trainingData.rowSize() - 1,
					trainingData.colSize() - 1, trainingData.colSize() - 1));

	MLA::DataFrame X_train( // @suppress("Invalid arguments")
			trainingData.locView(0, trainingData.rowSize() - 1, 0,
					trainingData.colSize() - 2));

	MLA::DataFrame makeSquare(X_train.viewTranspose() * X_train); // @suppress("Invalid arguments")
	MLA::DataFrame inverseSquare(makeSquare.viewInverse()); // @suppress("Invalid arguments")
	MLA::DataFrame hyp( // @suppress("Invalid arguments")
			inverseSquare * X_train.viewTranspose()
					* y_train);
	hypothesisParameters = hyp;
}

void MLA::Regression::LinearRegression::OLS::predict(
		MLA::DataFrame testingData) {
//	this->hypothesisParameters.printShape();
//	this->testingData.printShape();
	MLA::DataFrame predict( // @suppress("Invalid arguments")
			hypothesisParameters.viewTranspose() * testingData.viewTranspose());
	this->prediction = predict;
}

MLA::DataFrame MLA::Regression::LinearRegression::OLS::getPrediction() {
	return this->prediction.viewTranspose();
}

#endif

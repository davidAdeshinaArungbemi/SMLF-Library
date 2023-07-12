/*
 * Cost_Func.hpp
 *
 *  Created on: 6 Jul 2023
 *      Author: davidadeshina
 */

#ifndef COST_FUNC_DESC
#define COST_FUNC_DESC
#include "../DataFrame/DataFrame.hpp"
#include <cmath>
namespace MLA {
namespace Cost {
double MAE(MLA::DataFrame predicted, MLA::DataFrame actual);
double MSE(MLA::DataFrame predicted, MLA::DataFrame actual);
double RMSE(MLA::DataFrame predicted, MLA::DataFrame actual);
double MLE(MLA::DataFrame predicted, MLA::DataFrame actual);
}

namespace Metrics{

}
}

#endif /* Cost_Cost_FUNC_HPP_ */

#ifndef COST_FUNC_IMPLEMENTATION
#define COST_FUNC_IMPLEMENTATION

double MLA::Cost::MAE(MLA::DataFrame predicted, MLA::DataFrame actual) {
	DataFrame difference_matrix(
			(predicted.getData() - actual.getData()).array().abs());

	return difference_matrix.getData().sum()
			/ difference_matrix.getData().size();

}

double MLA::Cost::MSE(MLA::DataFrame predicted, MLA::DataFrame actual) {
	DataFrame squared_difference_matrix(
			(predicted.getData() - actual.getData()).array().square());

	return squared_difference_matrix.getData().sum()
			/ squared_difference_matrix.getData().size();
}

double RMSE(MLA::DataFrame predicted, MLA::DataFrame actual){
	return sqrt(MLA::Cost::MSE(predicted,actual));
}

#endif

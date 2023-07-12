/*
 * main.cpp
 *
 *  Created on: 27 Jun 2023
 *      Author: davidadeshina
 */

#include "DataFrame/DataFrame.hpp"
#include "Algorithms/Regression/LinearRegression.hpp"

int main(int argc, char const *argv[])
{
	MLA::DataFrame Train_data("DataSource/perfectFakeData.csv"); // @suppress("Invalid arguments")
	std::cout << Train_data.head(10);
	std::cout << Train_data.dropColumnView({1}).head(10);

	MLA::Regression::LinearRegression::OLS lr; // @suppress("Invalid arguments")
	lr.train(Train_data);

	MLA::DataFrame X_train = Train_data.dropColumnView({Train_data.colSize() - 1});
	lr.predict(X_train);
	std::cout << lr.getPrediction().head(10);
	lr.getPrediction().createCSV("cool", // @suppress("Invalid arguments")
								 "/Users/davidadeshina/Desktop/C++/Python ML");

	lr.
	return 0;
}

// MLA::DataFrame Train_data("DataSource/fakeData.csv"); // @suppress("Invalid arguments")
// MLA::Regression::LinearRegression::OLS lr; // @suppress("Invalid arguments")
// lr.train(Train_data);
//
// MLA::DataFrame X_train(
//		Train_data.locView(0, Train_data.rowSize() - 1, 0,
//				Train_data.colSize() - 2));
// lr.predict(X_train);
// lr.getPrediction().displayData();
// lr.getPrediction().createCSV("cool","/Users/davidadeshina/Desktop/C++/Python ML"); // @suppress("Invalid arguments")

/*
 * Optimiser.hpp
 *
 *  Created on: 6 Jul 2023
 *      Author: davidadeshina
 */

#ifndef OPTIMISER_DESC
#define OPTIMISER_DESC
#include "../../DataFrame/DataFrame.hpp"
namespace MLA {
class Optimiser {
	void commonGradientDescent(MLA::DataFrame actual_result, size_t batch_size,
			double epsilon = 0.000001);
	void momentumOptimiser();
	void rmsPropOptimiser();
	void adaGradOptimiser();
};
}

#endif

#ifndef OPTIMISER_IMPLEMENTATION
#define OPTIMISER_IMPLEMENTATION

void MLA::Optimiser::commonGradientDescent(MLA::DataFrame actual_result,
		size_t batch_size, double epsilon) {

}

#endif

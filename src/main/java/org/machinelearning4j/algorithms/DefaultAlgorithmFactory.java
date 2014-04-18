/*
 * Copyright 2014 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.machinelearning4j.algorithms;

import org.machinelearning4j.algorithms.supervisedlearning.LinearRegressionNormalEquationAlgorithm;
import org.machinelearning4j.algorithms.supervisedlearning.LinearRegressionNormalEquationAlgorithmImpl;
import org.machinelearning4j.algorithms.supervisedlearning.LogisticRegressionAlgorithm;
import org.machinelearning4j.algorithms.supervisedlearning.LogisticRegressionAlgorithmImpl;
import org.machinelearning4j.algorithms.unsupervisedlearning.KMeansClusteringAlgorithm;


/**
 * Default algorithm Factory  encapsulating default algorithm creation
 * within a central component
 * 
 * @author Michael Lavelle
 */
public class DefaultAlgorithmFactory implements AlgorithmFactory {

	/**
	 * Create a linear regression algorithm, using the normal equation strategy
	 * 
	 * @param regularisationLambda The value of lambda to use for regularisation
	 */
	@Override
	public LinearRegressionNormalEquationAlgorithm createLinearRegressionNormalEquationAlgorithm(double regularisationLambda) {
		return new LinearRegressionNormalEquationAlgorithmImpl(regularisationLambda);
	}

	@Override
	public KMeansClusteringAlgorithm createKMeansClusteringAlgorithm(
			int numberOfClusters) {
		return new KMeansClusteringAlgorithm(numberOfClusters);
	}

	@Override
	public LogisticRegressionAlgorithm createLogisticRegressionAlgorithm(double regularizationLambda) {
		return new LogisticRegressionAlgorithmImpl(regularizationLambda);
	}

}

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
import org.machinelearning4j.algorithms.unsupervisedlearning.KMeansClusteringAlgorithm;


/**
 * Factory to allow machine learning algorithm creation to be encapsulated
 * within a central component
 * 
 * @author Michael Lavelle
 */
public interface AlgorithmFactory {

	/**
	 * Create a linear regression algorithm, using the normal equation strategy
	 * 
	 * @param regularisationLambda The value of lambda to use for regularisation
	 */
	LinearRegressionNormalEquationAlgorithm createLinearRegressionNormalEquationAlgorithm(double regularizationLambda);

	/**
	 * Create a K-Means clustering algorithm for the required number of clusters
	 * 
	 * @param numberOfClusters The required number of clusters
	 */
	KMeansClusteringAlgorithm createKMeansClusteringAlgorithm(int numberOfClusters);

}

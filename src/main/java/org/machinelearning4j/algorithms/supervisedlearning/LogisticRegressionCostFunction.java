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
package org.machinelearning4j.algorithms.supervisedlearning;

/**
 * Defines logistic regression cost function
 * 
 * @author Michael Lavelle
 */
public class LogisticRegressionCostFunction implements CostFunction<double[],Double,LogisticRegressionHypothesisFunction> {

	@Override
	public double getCost(LogisticRegressionHypothesisFunction h, double[][] features,Double[] labels) {		
		int trainingExamples = features.length;
		double cost = 0d;
		for (int i = 0; i < trainingExamples; i++)
		{
			cost = cost - labels[i].doubleValue() * Math.log(h.predict(features[i]))
			- (1d - labels[i].doubleValue()) * Math.log(1 - h.predict(features[i]));
		}
		double regularizationTerm = 0d;
		if (h.getRegularizationLambda() > 0)
		{
			double sumOfThetaSquares = 0d;
			for (int j = 0 ; j < h.thetas.length; j++)
			{
				sumOfThetaSquares = sumOfThetaSquares+ h.thetas[j] * h.thetas[j];
			}
			regularizationTerm =  h.getRegularizationLambda()/2d + sumOfThetaSquares;
		}
		cost = (cost + regularizationTerm)/trainingExamples;
		return cost;
	}

	

}

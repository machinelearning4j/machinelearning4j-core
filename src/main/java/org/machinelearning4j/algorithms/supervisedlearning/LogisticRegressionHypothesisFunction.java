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
 * 
 * @author Michael Lavelle
 * 
 * A NumericHypothesisFunction encapsulates a function h : double[] -> Double so that predict(double[] numericFeatures) is a "good" predictor
 * given an numeric features array for the corresponding value of a numeric label.
 * 
 * This LogisticRegressionHypothesisFunction uses a set of theta weights corresponding
 * to the feature values to predict a value from a the sigmoid function applied to a linear combination of the weighted values.
 */
public class LogisticRegressionHypothesisFunction implements NumericHypothesisFunction {

	protected double[] thetas;
	protected double regularizationLambda;

	public LogisticRegressionHypothesisFunction(double[] thetas,double regularizationLambda)
	{
		this.thetas = thetas;
		this.regularizationLambda = regularizationLambda;
	}

	public double getRegularizationLambda() {
		return regularizationLambda;
	}

	@Override
	public Double predict(double[] x) {
		double y = 0d;
		for (int index = 0; index < x.length; index++)
		{
			y = y +  thetas[index] * x[index];

		}
		y =  1d/(1d + Math.exp(-y));
		return y;
	}
}

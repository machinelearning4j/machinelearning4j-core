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
package org.machinelearning4j.supervisedlearning;

import org.machinelearning4j.algorithms.LinearRegressionNormalEquationAlgorithm;

/**
 *  Trainable component which learns to predict numeric labels from elements of type T,
 *  using a linear regression algorithm
 * 
 * @author Michael Lavelle
 */
public class NumericLabelPredictor<T> implements
		LabelPredictor<T, Number> {

	@SuppressWarnings("unused")
	private LinearRegressionNormalEquationAlgorithm linearRegressionAlgorithm;
	
	@SuppressWarnings("unused")
	private LabeledTrainingSet<T, Number> labeledTrainingSet;
	 
	public NumericLabelPredictor(
			LabeledTrainingSet<T, Number> labeledTrainingSet,
			LinearRegressionNormalEquationAlgorithm linearRegressionAlgorithm) {
		this.linearRegressionAlgorithm = linearRegressionAlgorithm;
		this.labeledTrainingSet = labeledTrainingSet;
	}

	/**
	 * Train the label predictor, using the linearRegressionAlgorithm and 
	 * the data within the labeledTrainingSet
	 */
	@Override
	public void train() {
		// TODO
	}

	/**
	 * @param The element we wish to predict a label for
	 * @return  The predicted numeric label
	 */
	@Override
	public Number predictLabel(T element) {
		// TODO
		return null;
	}

}

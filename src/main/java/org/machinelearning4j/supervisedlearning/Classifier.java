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

import org.machinelearning4j.algorithms.supervisedlearning.LogisticRegressionAlgorithm;
/**
 * Classifying LabelPredictor - predicts a classification of type T with an associated
 * prediction probability
 * 
 * @author Michael Lavelle
 */
public class Classifier<T,C> implements LabelPredictor<T,ClassificationProbability<C>> {

	private LabeledTrainingSet<T, C> labeledTrainingSet;
	
	public Classifier(
			LabeledTrainingSet<T, C> labeledTrainingSet,
			LogisticRegressionAlgorithm logisticRegressionAlgorithm) {
			if (!labeledTrainingSet.isFeatureScalingConfigured())
			{
				throw new IllegalStateException("Logistic regression algorithm requires " +
						"that feature scaling is configured for the training set");
			}
			this.labeledTrainingSet = labeledTrainingSet;
	}

	@Override
	public void train() {
		
		if (!labeledTrainingSet.isDataFeatureScaled())
		{
			throw new IllegalStateException("Logistic regression algorithm requires " +
					"that the data in the training set has been feature scaled");
		}	
	}

	@Override
	public ClassificationProbability<C> predictLabel(T element) {
		// TODO
		return null;
	}

}
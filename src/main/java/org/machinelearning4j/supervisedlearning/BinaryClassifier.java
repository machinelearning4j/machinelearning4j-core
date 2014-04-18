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
import org.machinelearning4j.algorithms.supervisedlearning.NumericHypothesisFunction;
/**
 * Classifying LabelPredictor - predicts a classification of type T (where there are exactly
 * two exclusive values the classification can take ) with an associated
 * prediction probability
 * 
 * @author Michael Lavelle
 */
public class BinaryClassifier<T,C> implements LabelPredictor<T,ClassificationProbability<C>> {

	private LabeledTrainingSet<T, C> labeledTrainingSet;
	private LogisticRegressionAlgorithm logisticRegressionAlgorithm;
	private NumericHypothesisFunction hypothesisFunction;
	private double decisionBoundaryProbabilityThreshold = 0.5d;
	private C negativeClass;
	private C positiveClass;
	private NumericLabelMapper<C> labelMapper;
	
	public BinaryClassifier(
			LabeledTrainingSet<T, C> labeledTrainingSet,
			LogisticRegressionAlgorithm logisticRegressionAlgorithm,NumericLabelMapper<C> labelMapper,C negativeClass,C positiveClass) {
			if (!labeledTrainingSet.isFeatureScalingConfigured())
			{
				throw new IllegalStateException("Logistic regression algorithm requires " +
						"that feature scaling is configured for the training set");
			}
			this.labeledTrainingSet = labeledTrainingSet;
			this.logisticRegressionAlgorithm = logisticRegressionAlgorithm;
			this.labelMapper = labelMapper;
			this.negativeClass = negativeClass;
			this.positiveClass = positiveClass;
	}

	@Override
	public void train() {
		
		double[][] featureMatrix = labeledTrainingSet.getFeatureMatrix();
		
		if (!labeledTrainingSet.isDataFeatureScaled())
		{
			throw new IllegalStateException("Logistic regression algorithm requires " +
					"that the data in the training set has been feature scaled");
		}	
		
		hypothesisFunction = logisticRegressionAlgorithm.train(featureMatrix, labelMapper.getLabelValues(labeledTrainingSet.getLabels()));
	}

	@Override
	public ClassificationProbability<C> predictLabel(T element) {

		double[] featureValues = labeledTrainingSet.getFeatureMapper().getFeatureValues(element);
		if (labeledTrainingSet.isFeatureScalingConfigured() && labeledTrainingSet.isDataFeatureScaled())
		{
			featureValues = labeledTrainingSet.getFeatureScaler().scaleFeatures(labeledTrainingSet, featureValues,true);
		}
		String featureValuesString = "";
		for (double featureValue : featureValues)
		{
			featureValuesString = featureValuesString + "," + featureValue;
		}
		Double positiveClassProbability = logisticRegressionAlgorithm.predictLabel(featureValues , hypothesisFunction);
		if (positiveClassProbability == null)
		{
			return null;
		}
		else
		{
			if (positiveClassProbability.doubleValue() >= decisionBoundaryProbabilityThreshold)
			{
				return new ClassificationProbability<C>(positiveClass,positiveClassProbability.doubleValue());
			}
			else
			{
				return new ClassificationProbability<C>(negativeClass,1d - positiveClassProbability.doubleValue());
			}
		}
	}

	public void setDecisionBoundaryProbabilityThreshold(
			double decisionBoundaryProbabilityThreshold) {
		this.decisionBoundaryProbabilityThreshold = decisionBoundaryProbabilityThreshold;
	}
	
	

}

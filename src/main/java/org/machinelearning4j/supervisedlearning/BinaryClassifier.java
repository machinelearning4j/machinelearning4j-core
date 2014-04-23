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

import java.util.Arrays;

import org.machinelearning4j.algorithms.supervisedlearning.LogisticRegressionAlgorithm;
import org.machinelearning4j.algorithms.supervisedlearning.NumericHypothesisFunction;
/**
 * Classifying LabelPredictor - predicts a classification of type L (where there are exactly
 * two exclusive values the classification can take ) with an associated
 * prediction probability
 * 
 * @author Michael Lavelle
 */
public class BinaryClassifier<T,L,C> implements Classifier<T,L,C> {

	private LabeledTrainingSet<T, L> labeledTrainingSet;
	private LogisticRegressionAlgorithm<C> logisticRegressionAlgorithm;
	private NumericHypothesisFunction hypothesisFunction;
	private double decisionBoundaryProbabilityThreshold = 0.5d;
	private L negativeClass;
	private L positiveClass;
	private NumericLabelMapper<L> labelMapper;
	
	public BinaryClassifier(
			LabeledTrainingSet<T, L> labeledTrainingSet,
			LogisticRegressionAlgorithm<C> logisticRegressionAlgorithm,NumericLabelMapper<L> labelMapper,L negativeClass,L positiveClass) {
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
	public void train(C trainingContext) {
		
		double[][] featureMatrix = labeledTrainingSet.getFeatureMatrix();
		
		if (!labeledTrainingSet.isDataFeatureScaled())
		{
			throw new IllegalStateException("Logistic regression algorithm requires " +
					"that the data in the training set has been feature scaled");
		}	
		
		hypothesisFunction = logisticRegressionAlgorithm.train(featureMatrix, labelMapper.getLabelValues(labeledTrainingSet.getLabels()),trainingContext);
	}
	
	protected ClassificationProbability<L> predictLabel(double[] featureValues)
	{
		Double positiveClassProbability = logisticRegressionAlgorithm.predictLabel(featureValues , hypothesisFunction);
		if (positiveClassProbability == null)
		{
			return null;
		}
		else
		{
			if (positiveClassProbability.doubleValue() >= decisionBoundaryProbabilityThreshold)
			{
				return new ClassificationProbability<L>(positiveClass,positiveClassProbability.doubleValue());
			}
			else
			{
				return new ClassificationProbability<L>(negativeClass,1d - positiveClassProbability.doubleValue());
			}
		}
	}
	
	@Override
	public ClassificationProbability<L> predictLabel(T element) {

		double[] featureValues = labeledTrainingSet.getFeatureMapper().getFeatureValues(element);
		if (labeledTrainingSet.isFeatureScalingConfigured() && labeledTrainingSet.isDataFeatureScaled())
		{
			featureValues = labeledTrainingSet.getFeatureScaler().scaleFeatures(labeledTrainingSet, featureValues,true);
		}
		return predictLabel(featureValues);
		
	}

	public void setDecisionBoundaryProbabilityThreshold(
			double decisionBoundaryProbabilityThreshold) {
		this.decisionBoundaryProbabilityThreshold = decisionBoundaryProbabilityThreshold;
	}
	
	public double getTrainingSetPredictionAccuracyPercentage()
	{
		double predictedCorrect = 0;
		double[] actualLabelValues = labelMapper.getLabelValues(labeledTrainingSet.getLabels());
		int trainingExampleIndex = 0;
		for (double[] elementFeatures :labeledTrainingSet.getFeatureMatrix())
		{
			ClassificationProbability<L> prediction = predictLabel(elementFeatures);
			double actualLabelValue = actualLabelValues[trainingExampleIndex++];
			@SuppressWarnings("unchecked")
			double predictedLabelValue = labelMapper.getLabelValues(Arrays.asList(prediction.getClassification()))[0];
			if (predictedLabelValue == actualLabelValue)
			{
				predictedCorrect++;
			}
		}
		return 100 * predictedCorrect/labeledTrainingSet.getFeatureMatrix().length;
	}

}

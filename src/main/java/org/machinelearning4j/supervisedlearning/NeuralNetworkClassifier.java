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

import org.machinelearning4j.algorithms.supervisedlearning.MultiNumericHypothesisFunction;
import org.machinelearning4j.algorithms.supervisedlearning.NeuralNetworkAlgorithm;
/**
 * Classifying LabelPredictor - predicts a classification of type L (where there are exactly
 * two exclusive values the classification can take ) with an associated
 * prediction probability
 * 
 * @author Michael Lavelle
 */
public class NeuralNetworkClassifier<T,L,C> implements Classifier<T,L,C> {

	private LabeledTrainingSet<T, L> labeledTrainingSet;
	private NeuralNetworkAlgorithm<C> neuralNetworkAlgorithm;
	private MultiNumericHypothesisFunction hypothesisFunction;
	private L[] classes;
	private MultiNumericLabelMapper<L> labelMapper;
	
	public NeuralNetworkClassifier(
			LabeledTrainingSet<T, L> labeledTrainingSet,
			NeuralNetworkAlgorithm<C> neuralNetworkAlgorithm,MultiNumericLabelMapper<L> labelMapper,L[] classes) {
			if (!labeledTrainingSet.isFeatureScalingConfigured())
			{
				//throw new IllegalStateException("Neural Network algorithm requires " +
					//	"that feature scaling is configured for the training set");
			}
			this.labeledTrainingSet = labeledTrainingSet;
			this.neuralNetworkAlgorithm = neuralNetworkAlgorithm;
			this.labelMapper = labelMapper;
			this.classes = classes;
	}

	@Override
	public void train(C trainingContext) {
		
		double[][] featureMatrix = labeledTrainingSet.getFeatureMatrix();
		
		if (!labeledTrainingSet.isDataFeatureScaled())
		{
			//throw new IllegalStateException("Neural Network algorithm requires " +
				//	"that the data in the training set has been feature scaled");
		}	
		
		hypothesisFunction = neuralNetworkAlgorithm.train(featureMatrix, labelMapper.getLabelValues(labeledTrainingSet.getLabels()),trainingContext);
	}
	
	protected ClassificationProbability<L> predictLabel(double[] featureValues)
	{
		double[] outputNeurons = neuralNetworkAlgorithm.getOutputNeuronVector(featureValues , hypothesisFunction);
		
		Double max = null;
		Integer maxIndex = null;
		
		for (int i = 0; i < outputNeurons.length; i++)
		{
			if (max == null || outputNeurons[i] > max.doubleValue())
			{
				max = outputNeurons[i];
				maxIndex = i;
			}
		}
		
		return new ClassificationProbability<L>(classes[maxIndex],max);

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

	
	
	public double getTrainingSetPredictionAccuracyPercentage()
	{
		double predictedCorrect = 0;
		int trainingExampleIndex = 0;
		for (double[] elementFeatures :labeledTrainingSet.getFeatureMatrix())
		{
			ClassificationProbability<L> prediction = predictLabel(elementFeatures);
			L actualLabelValue = labeledTrainingSet.getLabels().get(trainingExampleIndex++);
			if (actualLabelValue.equals(prediction.getClassification()))
			{
				predictedCorrect++;
			}
		}
		return 100 * predictedCorrect/labeledTrainingSet.getFeatureMatrix().length;
	}

}

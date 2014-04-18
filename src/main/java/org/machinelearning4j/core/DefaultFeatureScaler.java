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
package org.machinelearning4j.core;
/**
 * Encapsulates logic for scaling training set features
 * This default implementation scales each feature by subtracting the mean of the
 * feature for the training set and divides by the standard deviation for that feature.
 * 
 * @author Michael Lavelle
 */
public class DefaultFeatureScaler implements FeatureScaler {

	@Override
	public double[] scaleFeatures(TrainingSet<?> parentTrainingSet,
			double[] elementFeatureArrayToScale,boolean firstFeatureIsIntercept) {
		
		int nonInterceptStatisticsIndex = firstFeatureIsIntercept ? 0 : 1;
		for (int i = 0 ; i < elementFeatureArrayToScale.length; i++)
		{	
			if ((firstFeatureIsIntercept && i != 0) || !firstFeatureIsIntercept)
			{
				Statistics statistics = parentTrainingSet.getFeatureStatistics()[nonInterceptStatisticsIndex - 1];
				elementFeatureArrayToScale[i] = scaleFeatureValue(elementFeatureArrayToScale[i],statistics);
			}
			nonInterceptStatisticsIndex++;
		}
	
		return elementFeatureArrayToScale;
	}

	private double scaleFeatureValue(double featureValue, Statistics featureStatistics) {
		double featureMean = featureStatistics.getMean();
		double stdDev = featureStatistics.getStdDev();
		return (featureValue - featureMean)/stdDev;
	}
}

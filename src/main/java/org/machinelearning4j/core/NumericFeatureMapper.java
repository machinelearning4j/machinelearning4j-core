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

import java.util.ArrayList;
import java.util.List;

/**
 * An adapter which maps the features of an element of type T to an array of double values
 *
 *	@author Michael Lavelle
 */
public class NumericFeatureMapper<T> {

	private List<NumericFeatureDefinition<T>> featureDefinitions;
	private int featureValueCount=0;
	private boolean hasInterceptFeature;
	
	public NumericFeatureMapper(boolean createInterceptTermFeature)
	{
		this.featureDefinitions = new ArrayList<NumericFeatureDefinition<T>>();
		if (createInterceptTermFeature)
		{
			this.featureDefinitions.add(new InterceptFeatureDefinition<T>());
			featureValueCount = 1;
			this.hasInterceptFeature = true;
		}
	}
	
	public boolean isHasInterceptFeature() {
		return hasInterceptFeature;
	}

	public void addFeatureDefinition(NumericFeatureDefinition<T> featureDefinition)
	{
		featureDefinitions.add(featureDefinition);
		featureValueCount += featureDefinition.getFeatureValueCount();
	}
	
	public double[] getFeatureValues(T element)
	{
		double[] numericFeatureValues = new double[featureValueCount];
		int featureValueIndex = 0;
		for (NumericFeatureDefinition<T> featureDefinition : featureDefinitions)
		{
			for (double featureValue : featureDefinition.getFeatureValues(element))
			{
				numericFeatureValues[featureValueIndex++] = featureValue;
			}
		}
		return numericFeatureValues;
	}

}

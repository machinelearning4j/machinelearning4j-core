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
 * Default implementation of a TrainingSet
 * 
 * @author Michael Lavelle
 */
public class TrainingSetImpl<T> implements TrainingSet<T> {

	private NumericFeatureMapper<T> numericFeatureMapper;
	private List<double[]> elementFeatures;
	protected int size;
	protected boolean dataIsFeatureScaled;
	protected FeatureScaler featureScaler;
	
	public TrainingSetImpl(NumericFeatureMapper<T> numericFeatureMapper,int size)
	{
		this.elementFeatures = new ArrayList<double[]>();
		this.numericFeatureMapper = numericFeatureMapper;
		this.size = size;
	}
	
	public TrainingSetImpl(NumericFeatureMapper<T> numericFeatureMapper,FeatureScaler featureScaler,int size)
	{
		this.elementFeatures = new ArrayList<double[]>();
		this.numericFeatureMapper = numericFeatureMapper;
		this.featureScaler = featureScaler;
		this.size = size;
	}
	
	
	
	protected void addFeatureValuesForElement(T element)
	{
		elementFeatures.add(numericFeatureMapper.getFeatureValues(element));
	}
	
	/**
	 * @param elements Data elements to add to training set
	 */
	@Override
	public void add(Iterable<T> elements) {
		for (T element : elements)
		{
			addFeatureValuesForElement(element);
		}
	}

	@Override
	public double[][] getFeatureMatrix() {
		
		// TODO. Build up feature matrix as elements are added instead of on feature matrix access
		
		double[][] featureMatrix = new double[elementFeatures.size()][];
		int elementIndex = 0;
		for (double[] elementFeatureArray : elementFeatures)
		{
			featureMatrix[elementIndex++] = elementFeatureArray;
		}
		return featureMatrix;
	}

	@Override
	public NumericFeatureMapper<T> getFeatureMapper() {
		return numericFeatureMapper;
	}



	@Override
	public boolean isFeatureScalingConfigured() {
		return featureScaler != null;
	}



	@Override
	public boolean isDataFeatureScaled() {
		return dataIsFeatureScaled;
	}

}

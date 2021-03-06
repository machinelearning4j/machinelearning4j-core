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
 * Stores data in memory for training sets consisting of
 * elements of type T
 *  
 *  
 * 
 * @author Michael Lavelle
 */
public interface TrainingSet<T> {

	/**
	 * @param elements Data elements to add to training set
	 */
	void add(Iterable<T> elements);
	
	/**
	 * @return The numeric features for each element of the training set
	 */
	double[][] getFeatureMatrix();


	/**
	 * @return An adapter which maps the features of an element of type T to an array of double values
	 */
	NumericFeatureMapper<T> getFeatureMapper();
	
	boolean isFeatureScalingConfigured();
	
	boolean isDataFeatureScaled();
	
	
	
	public FeatureScaler getFeatureScaler();
		
	public Statistics[] getFeatureStatistics();
	
	

}

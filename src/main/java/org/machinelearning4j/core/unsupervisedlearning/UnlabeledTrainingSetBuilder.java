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
package org.machinelearning4j.core.unsupervisedlearning;

import org.machinelearning4j.core.NumericFeatureDefinition;
import org.machinelearning4j.core.NumericFeatureMapper;
import org.machinelearning4j.core.TrainingSet;
import org.machinelearning4j.core.TrainingSetImpl;

/**
 * Encapsulates the building of an unlabeled TrainingSet.  To define a training set
 * before it can be populated, we must specify definitions for features of the
 * element we wish to analyze
 * 
 * @author Michael Lavelle
 */
public class UnlabeledTrainingSetBuilder<T> {

	private NumericFeatureMapper<T> numericFeatureMapper;
	
	public UnlabeledTrainingSetBuilder()
	{
		this.numericFeatureMapper = new NumericFeatureMapper<T>(false);
	}
	
	/**
	 * 
	 * @param featureDefinition Defines a numeric feature of an element of this training set
	 * 
	 * @return the chained builder
	 */
	public UnlabeledTrainingSetBuilder<T> withFeatureDefinition(
			NumericFeatureDefinition<T> featureDefinition)
	{
		numericFeatureMapper.addFeatureDefinition(featureDefinition);
		return this;
	}

	

	/**
	 * @return An unlabeled TrainingSet for elements of type T
	 */
	public TrainingSet<T> build()
	{
		return new TrainingSetImpl<T>(numericFeatureMapper);
	}

}

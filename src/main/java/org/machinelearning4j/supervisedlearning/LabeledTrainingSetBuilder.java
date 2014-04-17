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

import org.machinelearning4j.core.NumericFeatureDefinition;

/**
 * Encapsulates the building of a LabeledTrainingSet.  To define a training set
 * before it can be populated, we must specify definitions for features of the
 * element we wish to analyze and for the label associated with each element
 * 
 * @author Michael Lavelle
 */
public class LabeledTrainingSetBuilder<T,L> {

	/**
	 * 
	 * @param featureDefinition Defines a numeric feature of an element of this training set
	 * 
	 * @return the chained builder
	 */
	public LabeledTrainingSetBuilder<T,L> withFeatureDefinition(
			NumericFeatureDefinition<T> featureDefinition)
	{
		// TODO
		return this;
	}

	/**
	 * @param priceLabelDefinition Defines a label of type T of an element of this training set
	 * @return
	 */
	public LabeledTrainingSetBuilder<T,L> withLabel(
			LabelDefinition<T,L> priceLabelDefinition)
	{
		// TODO
		return this;
	}

	/**
	 * @return A LabeledTrainingSet for elements of type T with labels of type L
	 */
	public LabeledTrainingSet<T,L> build()
	{
		// TODO
		return new LabeledTrainingSetImpl<T,L>();
	}

}

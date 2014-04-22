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
package org.machinelearning4j.algorithms.supervisedlearning;


/**
 * 
 * @author Michael Lavelle
 * 
 * A RegressionAlgorithm can be trained on a collection of elements, each of which has a list of
 * numeric features, and learns a NumericHypothesisFunction which can be used to predict the
 * value of numeric labels given the numeric features of a specific element.
 * 
 * The numeric features for a specific element are encapsulated in an array of doubles, with
 * the features for the input collection of training elements being encapsulated in a feature matrix with
 * columns for each feature, and rows for each element.
 *  * 
 * 
 */
public interface RegressionAlgorithm<C> {
	
	public NumericHypothesisFunction train(double[][] featureMatrix,double[] labelVector,C trainingContext);
	public Double predictLabel(double[] featureVector,NumericHypothesisFunction hypothesisFunction);
	public boolean isFeatureScaledDataRequired();
}

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
 * A NeuralNetworkAlgorithm can be trained on a fixed-size array of input neuron vectors, and learns a MultiNumericHypothesisFunction which can be used to predict the
 * value of the output neurons given the numeric features of a specific element.
 * 
 * 
 */
public interface NeuralNetworkAlgorithm<C>  {
	
	public MultiNumericHypothesisFunction train(double[][] inputNeuronVectors,double[][] outputNeuronVectors,C trainingContext);
	public double[] getOutputNeuronVector(double[] inputNeuronVector,MultiNumericHypothesisFunction hypothesisFunction);
	public boolean isFeatureScaledDataRequired();
}

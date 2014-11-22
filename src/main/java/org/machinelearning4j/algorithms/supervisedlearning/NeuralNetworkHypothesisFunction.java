package org.machinelearning4j.algorithms.supervisedlearning;

import org.dvincent1337.neuralNet.NeuralNetwork;
import org.jblas.DoubleMatrix;

public class NeuralNetworkHypothesisFunction implements
		MultiNumericHypothesisFunction {

	private NeuralNetwork neuralNetwork;
	
	public NeuralNetworkHypothesisFunction(NeuralNetwork neuralNetwork)
	{
		this.neuralNetwork = neuralNetwork;
	}

	
	public NeuralNetwork getNeuralNetwork() {
		return neuralNetwork;
	}

	@Override
	public double[] predict(double[] numericFeatures) {
		return neuralNetwork.predictFP(new DoubleMatrix(new double[][] {numericFeatures})).toArray();
	}

}

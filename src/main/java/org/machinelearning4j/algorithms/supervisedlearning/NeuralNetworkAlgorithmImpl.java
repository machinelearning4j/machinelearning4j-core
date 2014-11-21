package org.machinelearning4j.algorithms.supervisedlearning;

import org.dvincent1337.neuralNet.NeuralNetwork;
import org.jblas.DoubleMatrix;

public class NeuralNetworkAlgorithmImpl implements
		NeuralNetworkAlgorithm<NeuralNetworkAlgorithmTrainingContext> {


	
	
	@Override
	public MultiNumericHypothesisFunction train(double[][] inputNeuronVectors,
			double[][] outputNeuronVectors,
			NeuralNetworkAlgorithmTrainingContext trainingContext) {
		
		int[] topology = new int[2 + trainingContext.getHiddenLayerSizes().length];
		topology[0] = trainingContext.getInputLayerSize();
		for (int i = 0 ; i < trainingContext.getHiddenLayerSizes().length; i++)
		{
			topology[i + 1] = trainingContext.getHiddenLayerSizes()[i];
		}
		topology[topology.length - 1] = trainingContext.getOutputLayerSize();
		//double lambda = 0.905343;
		double lambda = trainingContext.getRegularizationLambda();
		NeuralNetwork neuralNetwork = new NeuralNetwork(topology,true);	//Cre
		
		DoubleMatrix X = new DoubleMatrix(inputNeuronVectors);
		DoubleMatrix Y = new DoubleMatrix(outputNeuronVectors);

		
		neuralNetwork.trainBP(X, Y, lambda, (int)trainingContext.getMaxIterations(),true); //Train the nerual network
		
		
		return new NeuralNetworkHypothesisFunction(neuralNetwork);
	}

	@Override
	public double[] getOutputNeuronVector(double[] inputNeuronVector,
			MultiNumericHypothesisFunction hypothesisFunction) {
		return hypothesisFunction.predict(inputNeuronVector);
	}

	@Override
	public boolean isFeatureScaledDataRequired() {
		return false;
	}

}

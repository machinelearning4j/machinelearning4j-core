package org.machinelearning4j.algorithms.supervisedlearning;


public class NeuralNetworkAlgorithmTrainingContext  {

	private int maxIterations;
	
	private double regularizationLambda;

	
	public NeuralNetworkAlgorithmTrainingContext( int maxIterations,int inputLayerSize,int[] hiddenLayerSizes,int outputLayerSize) {
		super();
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayerSizes = hiddenLayerSizes;
		this.outputLayerSize = outputLayerSize;
		this.maxIterations = maxIterations;
	}
	
	private int inputLayerSize;
	private int[] hiddenLayerSizes;
	private int outputLayerSize;
	
	
	
	public int getMaxIterations() {
		return maxIterations;
	}

	public int[] getLayerSizes()
	{
		int[] sizes = new int[2 + hiddenLayerSizes.length];
		sizes[0] = inputLayerSize;
		for (int i = 1; i < sizes.length - 1; i++)
		{
			sizes[i] = hiddenLayerSizes[i-1];
		}
		sizes[sizes.length - 1] = outputLayerSize;
		return sizes;
	}

	public int getInputLayerSize() {
		return inputLayerSize;
	}

	public int[] getHiddenLayerSizes() {
		return hiddenLayerSizes;
	}

	public int getOutputLayerSize() {
		return outputLayerSize;
	}

	public double getRegularizationLambda() {
		return regularizationLambda;
	}

	public void setRegularizationLambda(double regularizationLambda) {
		this.regularizationLambda = regularizationLambda;
	}

	

}

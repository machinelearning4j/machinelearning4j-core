package org.machinelearning4j.algorithms.supervisedlearning;


public class LinearRegressionHypothesisFunction implements NumericHypothesisFunction {

	private double[] thetas;

	public LinearRegressionHypothesisFunction(double[] thetas)
	{
		this.thetas = thetas;
	}
	
	@Override
	public Double predict(double[] numericFeatures) {
		double y = 0d;
		for (int index = 0; index < numericFeatures.length; index++)
		{
			y = y +  thetas[index] * numericFeatures[index];
		}
		return y;
	}

}

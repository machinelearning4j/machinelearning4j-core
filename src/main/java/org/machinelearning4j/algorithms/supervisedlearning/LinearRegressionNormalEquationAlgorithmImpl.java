package org.machinelearning4j.algorithms.supervisedlearning;

/**
 *  A linear regression algorithm using the normal equation, using regularization
 *  defined by the value of the specified regularisationLambda
 * 
 * @author Michael Lavelle
 */
public class LinearRegressionNormalEquationAlgorithmImpl implements
		LinearRegressionNormalEquationAlgorithm {

	@SuppressWarnings("unused")
	private double regularisationLambda;
	
	public LinearRegressionNormalEquationAlgorithmImpl(
			double regularisationLambda) {
		this.regularisationLambda = regularisationLambda;
	}

	@Override
	public NumericHypothesisFunction train(double[][] featureMatrix,
			double[] labelVector) {
		// TODO
		return null;
	}

	@Override
	public Double predictLabel(double[] featureVector,
			NumericHypothesisFunction hypothesisFunction) {
		// TODO
		return null;
	}

}

package org.machinelearning4j.algorithms.supervisedlearning;

import Jama.Matrix;

/**
 *  A linear regression algorithm using the normal equation, using regularization
 *  defined by the value of the specified regularisationLambda
 *  
 *  Warning - if specified regularisation lambda is zero, no regularisation
 *  with occur and there is a risk of matrix operations throwing a non-invertible
 *  exception.
 *  
 *  If regularisation is used, the necessary matrices are guaranteed to be invertible
 * 
 * @author Michael Lavelle
 */
public class LinearRegressionNormalEquationAlgorithmImpl implements
		LinearRegressionNormalEquationAlgorithm {

	private double regularisationLambda;
	
	public LinearRegressionNormalEquationAlgorithmImpl(
			double regularisationLambda) {
		this.regularisationLambda = regularisationLambda;
	}
	
	private Matrix createFeatureMatrix(double[][] featureMatrixValues)
	{

		return new Matrix(featureMatrixValues);
	}
	
	private Matrix createLabelVector(double[] labelVectorValues)
	{
		Matrix labelVector = new Matrix(labelVectorValues.length,1);
		int i  = 0;
		for (double label : labelVectorValues)
		{
			labelVector.set(i, 0, label);
			i++;
		}
		return labelVector;
	}
	
	private Matrix createRegularisationMatrix(Matrix preRegularisationMatrix,double lambda)
	{
		Matrix regularisationMatrix = Matrix.identity(preRegularisationMatrix.getRowDimension(), preRegularisationMatrix.getColumnDimension());
		regularisationMatrix.set(0, 0, 0);
		regularisationMatrix.times(lambda);
		return regularisationMatrix;
	}

	@Override
	public NumericHypothesisFunction train(double[][] featureMatrixValues,
			double[] labelVectorValues) {
		// TODO
		int featureCount = featureMatrixValues[0].length;
		double[] thetas = new double[featureCount];
		
		Matrix featureMatrix = createFeatureMatrix(featureMatrixValues);
		Matrix labelVector = createLabelVector(labelVectorValues);

		Matrix result = featureMatrix.transpose().times(featureMatrix);
		if (regularisationLambda > 0)
		{
			// Regularise if lambda specified is greater than zero
			result = result.plus(createRegularisationMatrix(result,regularisationLambda));
		}
		
		result = result.inverse().times(featureMatrix.transpose());
		Matrix thetasMatrix = result.times(labelVector);
		
		for (int j = 0 ; j < thetas.length; j++)
		{
			thetas[j] = thetasMatrix.get(j,0);
		}
		return new LinearRegressionHypothesisFunction(thetas);
	}

	@Override
	public Double predictLabel(double[] featureVector,
			NumericHypothesisFunction hypothesisFunction) {
		return hypothesisFunction.predict(featureVector);
	}

}

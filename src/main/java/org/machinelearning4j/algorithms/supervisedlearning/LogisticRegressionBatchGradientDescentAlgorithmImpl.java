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
 * A logistic regression algorithm implementation using batch gradient descent
 * 
 * @author Michael Lavelle
 */
public class LogisticRegressionBatchGradientDescentAlgorithmImpl implements
		LogisticRegressionAlgorithm<GradientDescentAlgorithmTrainingContext> {

	public LogisticRegressionBatchGradientDescentAlgorithmImpl() {
	}

	private boolean takeSnapshotOfCostFunctionValueIfApplicable(
			double[][] featureMatrix, double[] labelVector,
			GradientDescentAlgorithmTrainingContext trainingContext,
			LogisticRegressionHypothesisFunction hypothesisFunction) {
		if (trainingContext.getCostFunctionSnapshotIntervalInIterations() != null && trainingContext.getCurrentIteration()
				% trainingContext.getCostFunctionSnapshotIntervalInIterations() == 0) {
			Double[] labelsArray = new Double[labelVector.length];
			for (int i = 0; i < labelVector.length; i++) {
				labelsArray[i] = new Double(labelVector[i]);
			}
			trainingContext.addCostFunctionSnapshotValue(getCostFunction()
					.getCost(hypothesisFunction, featureMatrix, labelsArray));
			return true;
		}
		return false;
	}

	@Override
	public NumericHypothesisFunction train(double[][] featureMatrix,
			double[] labelVector,
			GradientDescentAlgorithmTrainingContext trainingContext) {
		LogisticRegressionHypothesisFunction hypothesisFunction = getInitialHypothesisFunction(featureMatrix[0].length);
		boolean snapshotTakenOfLastIteration = false;
		
		if (trainingContext.getLearningRateAlpha() == null)
		{
			throw new RuntimeException("No learning rate alpha specified on training context");
		}
		
		while (trainingContext.isTrainingRunning()
				&& !trainingContext.isTrainingSuccessful()) {
			snapshotTakenOfLastIteration = takeSnapshotOfCostFunctionValueIfApplicable(
					featureMatrix, labelVector, trainingContext,
					hypothesisFunction);

			trainingContext.incrementIterationNumber();

			hypothesisFunction = performHypothesisFunctionUpdateIteration(
					featureMatrix, labelVector, hypothesisFunction,
					trainingContext);

		}

		if (!snapshotTakenOfLastIteration) {
			takeSnapshotOfCostFunctionValueIfApplicable(featureMatrix,
					labelVector, trainingContext, hypothesisFunction);
		}

		if (trainingContext.isTrainingSuccessful()) {
			return hypothesisFunction;
		} else {
			if (trainingContext.getConvergenceCriteria() != null) {
				throw new RuntimeException(
						"Training has stopped running but has not satified convergence criteria");
			} else {
				throw new RuntimeException(
						"Training has stopped running but cannot be deemed to have converged as no convergence criteria have been specified on the training context");
			}
		}

	}

	protected LogisticRegressionHypothesisFunction performHypothesisFunctionUpdateIteration(
			double[][] featureMatrix, double[] labelVector,
			LogisticRegressionHypothesisFunction hypothesisFunction,
			GradientDescentAlgorithmTrainingContext trainingContext) {

		double[] newThetas = new double[hypothesisFunction.thetas.length];
		double[] gradients = getGradients(newThetas.length, featureMatrix,
				labelVector, hypothesisFunction);
		for (int j = 0; j < newThetas.length; j++) {
			newThetas[j] = hypothesisFunction.thetas[j]
					- trainingContext.getLearningRateAlpha() * gradients[j];
		}

		return new LogisticRegressionHypothesisFunction(newThetas);
	}

	private double[] getGradients(int thetaCount, double[][] featureMatrix,
			double[] labelVector,
			LogisticRegressionHypothesisFunction hypothesisFunction) {
		double[] gradients = new double[thetaCount];
		for (int j = 0; j < thetaCount; j++) {
			double m = featureMatrix.length;
			double s = 0;
			for (int i = 0; i < m; i++) {
				double t = (hypothesisFunction.predict(featureMatrix[i]) - labelVector[i])
						* featureMatrix[i][j];
				s = s + t;
			}
			double grad = s / m;
			gradients[j] = grad;
		}
		return gradients;
	}

	protected LogisticRegressionHypothesisFunction getInitialHypothesisFunction(
			int thetaCount) {
		double[] initialThetas = new double[thetaCount];
		return new LogisticRegressionHypothesisFunction(initialThetas);
	}

	@Override
	public Double predictLabel(double[] featureVector,
			NumericHypothesisFunction hypothesisFunction) {
		return hypothesisFunction.predict(featureVector);
	}

	@Override
	public boolean isFeatureScaledDataRequired() {
		return true;
	}

	public CostFunction<double[], Double, LogisticRegressionHypothesisFunction> getCostFunction() {
		return new LogisticRegressionCostFunction();
	}

}

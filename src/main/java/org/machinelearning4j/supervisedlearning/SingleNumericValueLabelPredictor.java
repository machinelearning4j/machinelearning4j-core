package org.machinelearning4j.supervisedlearning;

import org.machinelearning4j.algorithms.supervisedlearning.LinearRegressionNormalEquationAlgorithm;

public class SingleNumericValueLabelPredictor<T> extends
		NumericLabelPredictor<T, Number> {

	public SingleNumericValueLabelPredictor(
			LabeledTrainingSet<T, Number> labeledTrainingSet,
			LinearRegressionNormalEquationAlgorithm linearRegressionAlgorithm) {
		super(labeledTrainingSet, new SingleNumericValueLabelMapper(), linearRegressionAlgorithm);
	}

}

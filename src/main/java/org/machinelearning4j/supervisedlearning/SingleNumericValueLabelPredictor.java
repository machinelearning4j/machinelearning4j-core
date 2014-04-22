package org.machinelearning4j.supervisedlearning;

import org.machinelearning4j.algorithms.supervisedlearning.LinearRegressionAlgorithm;

public class SingleNumericValueLabelPredictor<T,C> extends
		NumericLabelPredictor<T, Number,C> {

	public SingleNumericValueLabelPredictor(
			LabeledTrainingSet<T, Number> labeledTrainingSet,
			LinearRegressionAlgorithm<C> linearRegressionAlgorithm) {
		super(labeledTrainingSet, new SingleNumericValueLabelMapper(), linearRegressionAlgorithm);
	}

}

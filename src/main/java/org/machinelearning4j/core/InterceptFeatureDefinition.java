package org.machinelearning4j.core;

public class InterceptFeatureDefinition<T> extends SingleNumericValuedFeatureDefinition<T> {

	@Override
	protected double getFeatureValue(T element) {
		return 1d;
	}

}

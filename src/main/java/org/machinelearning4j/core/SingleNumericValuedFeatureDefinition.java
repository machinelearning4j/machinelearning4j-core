package org.machinelearning4j.core;

public abstract class SingleNumericValuedFeatureDefinition<T> implements
		NumericFeatureDefinition<T> {

	@Override
	public int getFeatureValueCount() {
		return 1;
	}
	
	protected abstract double getFeatureValue(T element);

	@Override
	public double[] getFeatureValues(T element) {
		return new double[] {getFeatureValue(element)};
	}

}

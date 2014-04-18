package org.machinelearning4j.supervisedlearning;

import java.util.List;


public class SingleNumericValueLabelMapper implements NumericLabelMapper<Number> {

	@Override
	public double[] getLabelValues(List<Number> labels) {
		double[] labelValues = new double[labels.size()];
		int labelValueIndex = 0;
		for (Number label : labels)
		{
			labelValues[labelValueIndex++] = label.doubleValue();
		}
		return labelValues;
	}

}

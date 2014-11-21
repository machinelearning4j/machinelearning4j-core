package org.machinelearning4j.supervisedlearning;

import java.util.List;

public interface MultiNumericLabelMapper<L> {

	public double[][] getLabelValues(List<L> labels);
}

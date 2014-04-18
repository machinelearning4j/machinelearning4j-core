package org.machinelearning4j.supervisedlearning;

import java.util.List;

public interface NumericLabelMapper<L> {

	public double[] getLabelValues(List<L> labels);
}

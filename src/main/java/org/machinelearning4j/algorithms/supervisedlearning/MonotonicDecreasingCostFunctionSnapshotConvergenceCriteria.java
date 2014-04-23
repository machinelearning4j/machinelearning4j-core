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

import java.util.List;

import org.apache.log4j.Logger;
import org.machinelearning4j.algorithms.ConvergenceCriteria;
/**
 * Defines convergence criteria for the cost function of a gradient descent algorithm which is monotonically decreasing
 * with each iteration.
 * 
 * @author Michael Lavelle
 */
public class MonotonicDecreasingCostFunctionSnapshotConvergenceCriteria implements ConvergenceCriteria<GradientDescentAlgorithmTrainingContext> {

	static Logger LOG = Logger.getLogger(MonotonicDecreasingCostFunctionSnapshotConvergenceCriteria.class);

	
	private double minimumInitialCostSnapshotDecreasePercentage;
	private double declareConvergenceIfCostSnapshotDecreasesByLessThanPercentage;
	
	public MonotonicDecreasingCostFunctionSnapshotConvergenceCriteria(double minimumInitialCostSnapshotDecreasePercentage,double declareConvergenceIfCostSnapshotDecreasesByLessThanPercentage) {
		this.declareConvergenceIfCostSnapshotDecreasesByLessThanPercentage = declareConvergenceIfCostSnapshotDecreasesByLessThanPercentage;
		this.minimumInitialCostSnapshotDecreasePercentage = minimumInitialCostSnapshotDecreasePercentage;
	}

	@Override
	public boolean isPrerequisiteConditionViolated(GradientDescentAlgorithmTrainingContext trainingContext) {
		List<Double> costFunctionSnapshots = trainingContext.getCostFunctionSnapshotHistory();
	
		if (costFunctionSnapshots.size() > 1)
		{
			// Check initial cost decrease
			double initialCostSnapshot = costFunctionSnapshots.get(0).doubleValue();
			if (initialCostSnapshot == 0) return true;
			double nextCostSnapshot = costFunctionSnapshots.get(1).doubleValue();
			if (nextCostSnapshot != 0)
			{
				double initialCostDecreasePercentage = (1d - nextCostSnapshot/initialCostSnapshot) * 100d;
				if (initialCostDecreasePercentage < minimumInitialCostSnapshotDecreasePercentage) return true; 
			}
			
			// Check monotonic condition
			Double previousSnapshot = null;
			for (Double costSnapshot : costFunctionSnapshots)
			{
				if (previousSnapshot != null)
				{
					if (costSnapshot.doubleValue() >= previousSnapshot.doubleValue()) return true;
				}
				previousSnapshot = costSnapshot;
			}
		}
		return false;
	}

	@Override
	public boolean isConvergenceCompleteConditionSatisfied(GradientDescentAlgorithmTrainingContext trainingContext) {
		List<Double> costFunctionSnapshots = trainingContext.getCostFunctionSnapshotHistory();
		
		if (costFunctionSnapshots.size() == 0)
		{
			throw new RuntimeException("No cost function snapshots being captured.");
		}
			
		
		if (costFunctionSnapshots.size() > 2)
		{
			double lastCostSnapshot = costFunctionSnapshots.get(costFunctionSnapshots.size() - 1);
			double penultimateCostSnapshot = costFunctionSnapshots.get(costFunctionSnapshots.size() - 2);
			double lastCostDecreasePercentage = (1d - lastCostSnapshot/penultimateCostSnapshot) * 100d;
			return lastCostDecreasePercentage < declareConvergenceIfCostSnapshotDecreasesByLessThanPercentage;
		}
		else
		{
			return false;
		}
	}

	

}

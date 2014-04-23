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

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.machinelearning4j.algorithms.ConvergenceCriteria;

/**
 *  Training context for a gradient descent regression algorithm
 * 
 * @author Michael Lavelle
 */
public class GradientDescentAlgorithmTrainingContext {

	static Logger LOG = Logger.getLogger(GradientDescentAlgorithmTrainingContext.class);
	
	private double regularizationLambda;
	
	private Double learningRateAlpha;
	
	private Integer costFunctionSnapshotIntervalInIterations;

	private boolean trainingTerminated;
	
	private long maxIterations;
	
	private long currentIteration;
	
	public long getCurrentIteration() {
		return currentIteration;
	}
	
	private List<Double> costFunctionSnapshotHistory;
	
	
	public void addCostFunctionSnapshotValue(double costFunctionValue)
	{
		if (LOG.isDebugEnabled())
		{
			LOG.debug("Iteration: " + currentIteration + " : Cost: " +  costFunctionValue);
		}
		costFunctionSnapshotHistory.add(costFunctionValue);
	}
	
	
	private ConvergenceCriteria<GradientDescentAlgorithmTrainingContext> convergenceCriteria;
	
	public GradientDescentAlgorithmTrainingContext(long maxIterations)
	{
		this.maxIterations = maxIterations;
		this.costFunctionSnapshotHistory = new ArrayList<Double>();
	}
	
	public void incrementIterationNumber()
	{
		currentIteration++;
	}

	public boolean isTrainingRunning() {
		return !isTrainingTerminated() && (convergenceCriteria == null || !convergenceCriteria.isPrerequisiteConditionViolated(this)) && !isIterationLimitReached();
	}
	
	private boolean isIterationLimitReached() {
		return currentIteration >= maxIterations;
	}

	public boolean isTrainingTerminated() {
		return trainingTerminated;
	}
	public void setTrainingTerminated(boolean trainingTerminated) {
		this.trainingTerminated = trainingTerminated;
	}


	public boolean isTrainingSuccessful() {
		return convergenceCriteria != null && currentIteration > 0 && convergenceCriteria.isConvergenceCompleteConditionSatisfied(this);
	}

	 public void setConvergenceCriteria(ConvergenceCriteria<GradientDescentAlgorithmTrainingContext> convergenceCriteria) {
		 this.convergenceCriteria = convergenceCriteria;
	}

	public ConvergenceCriteria<GradientDescentAlgorithmTrainingContext> getConvergenceCriteria() {
		return convergenceCriteria;
	}


	public double getRegularizationLambda() {
		return regularizationLambda;
	}

	public void setRegularizationLambda(double regularizationLambda) {
		this.regularizationLambda = regularizationLambda;
	}

	public Double getLearningRateAlpha() {
		return learningRateAlpha;
	}

	public void setLearningRateAlpha(Double learningRateAlpha) {
		this.learningRateAlpha = learningRateAlpha;
	}
	

	public void setCostFunctionSnapshotIntervalInIterations(
			int costFunctionSnapshotItervalInIterations) {
		this.costFunctionSnapshotIntervalInIterations = costFunctionSnapshotItervalInIterations;
	}

	public Integer getCostFunctionSnapshotIntervalInIterations() {
		return costFunctionSnapshotIntervalInIterations;
	}

	public List<Double> getCostFunctionSnapshotHistory() {
		return costFunctionSnapshotHistory;
	}
	
	public long getMaxIterations() {
		return maxIterations;
	}

}

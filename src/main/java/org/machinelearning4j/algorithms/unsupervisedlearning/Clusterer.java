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
package org.machinelearning4j.algorithms.unsupervisedlearning;

import java.util.Set;

import org.machinelearning4j.core.TrainingSet;

/**
 * Obtains Centroids for a training set of elements of type T
 *
 * @author Michael Lavelle
 */
public class Clusterer<T> {

	@SuppressWarnings("unused")
	private ClusteringAlgorithm clusteringAlgorithm;
	
	@SuppressWarnings("unused")
	private TrainingSet<T> trainingSet;
	
	public Clusterer(TrainingSet<T> trainingSet,
			ClusteringAlgorithm clusteringAlgorithm) {
		this.trainingSet = trainingSet;
	}

	public Set<Centroid<T>> getCentroids() {
		// TODO
		return null;
	}

}

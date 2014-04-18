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
package org.machinelearning4j.unsupervisedlearning;

import java.util.Collection;
import java.util.Set;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.machinelearning4j.algorithms.AlgorithmFactory;
import org.machinelearning4j.algorithms.DefaultAlgorithmFactory;
import org.machinelearning4j.algorithms.unsupervisedlearning.Centroid;
import org.machinelearning4j.algorithms.unsupervisedlearning.Clusterer;
import org.machinelearning4j.algorithms.unsupervisedlearning.ClusteringAlgorithm;
import org.machinelearning4j.core.Builders;
import org.machinelearning4j.core.TrainingSet;
import org.machinelearning4j.supervisedlearning.BedroomsFeatureDefinition;
import org.machinelearning4j.supervisedlearning.House;
import org.machinelearning4j.supervisedlearning.HouseCsvDataExtractor;
import org.machinelearning4j.supervisedlearning.SquareFeetFeatureDefinition;
import org.machinelearning4j.util.CsvFileClassloaderDataSource;
import org.machinelearning4j.util.TrainingSetDataSource;

/**
 * Initial end-to-end integration test for clustering
 * 
 * @author Michael Lavelle
 */
@Ignore
public class ClusteringIntegrationTest {

	private Iterable<House> houses;
	private int trainingSetSize;
	private AlgorithmFactory algorithmFactory;

	
	@Before
	public void setUp() throws Exception
	{
		// Read housing data used as part of a programming assignment on
		// Stanford's Machine Learning course
		Collection<House> housesCollection = getHouseDataFromFile("ex1data2.txt");
		
		// Assert that we have read the training data correctly
		Assert.assertNotNull(housesCollection);
		Assert.assertEquals(47,housesCollection.size());
		House house1 = housesCollection.iterator().next();
		Assert.assertEquals(2104, house1.getSquareFeet());
		Assert.assertEquals(3, house1.getBedrooms());
		Assert.assertEquals(399900, house1.getPrice());
				
		
		this.houses = housesCollection;
		this.trainingSetSize = housesCollection.size();
		this.algorithmFactory = new DefaultAlgorithmFactory();	
	}
	
	@Test
	public void testClustering()
	{		
		// Create and configure the training set
		TrainingSet<House> trainingSet = 
				Builders.createUnlabeledTrainingSetBuilder(House.class,trainingSetSize)
				.withFeatureDefinition(new BedroomsFeatureDefinition())
				.withFeatureDefinition(new SquareFeetFeatureDefinition())
				.build();
					
		// Create a clustering algorithm that we can use to cluster the data
		// ( Passing in the number of required clusters )
		
		int requiredNumberOfClusters = 3;
		
		ClusteringAlgorithm clusteringAlgorithm = 
				algorithmFactory.createKMeansClusteringAlgorithm(requiredNumberOfClusters);
		
		// Create a clusterer for the training set, using this algorithm
		Clusterer<House> houseClusterer = 
				new Clusterer<House>(trainingSet,clusteringAlgorithm);

		// Add our housing data to the training set
		trainingSet.add(houses);

		// Get the element centroids for this training set
		Set<Centroid<House>> houseDataCentroids = houseClusterer.getCentroids();

		// Assert the centroids returned are not null
		Assert.assertNotNull("Centroids should not be null",houseDataCentroids);
		
		// Assert that we have obtained the required number of centroids
		Assert.assertEquals(requiredNumberOfClusters,houseDataCentroids.size());
	}
	
	private Collection<House> getHouseDataFromFile(String fileName)
	{
		TrainingSetDataSource<House,Collection<House>> houses = new CsvFileClassloaderDataSource<House>(fileName,getClass().getClassLoader(),new HouseCsvDataExtractor());
		return houses.getData();
	}
	
}

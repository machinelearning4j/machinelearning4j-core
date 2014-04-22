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
package org.machinelearning4j.supervisedlearning;

import java.util.Collection;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;
import org.machinelearning4j.algorithms.AlgorithmFactory;
import org.machinelearning4j.algorithms.DefaultAlgorithmFactory;
import org.machinelearning4j.algorithms.supervisedlearning.LinearRegressionAlgorithm;
import org.machinelearning4j.algorithms.supervisedlearning.LinearRegressionNormalEquationTrainingContext;
import org.machinelearning4j.core.Builders;
import org.machinelearning4j.util.CsvFileClassloaderDataSource;
import org.machinelearning4j.util.TrainingSetDataSource;

/**
 * Initial end-to-end integration test for numeric label prediction.
 * 
 * Uses data from a programming assigment on Stanford's Machine Learning
 * Coursera course to take data consisting of attributes of Houses and
 * applying Linear Regression to predict a price for the house.
 * 
 * @author Michael Lavelle
 */
public class NumericLabelPredictorIntegrationTest {

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
	public void testLabelPrediction_WithNormalEquationLinearRegressionAlgorithm()
	{		
		// Create and configure the training set
		LabeledTrainingSet<House,Number> labeledTrainingSet = 
				Builders.createLabeledTrainingSetBuilder(House.class,Number.class,trainingSetSize)
				.withFeatureDefinition(new BedroomsFeatureDefinition())
				.withFeatureDefinition(new SquareFeetFeatureDefinition())
				.withLabel(new PriceLabelDefinition())
				.build();
					
		// Create a linear regression algorithm that we can use to predict the price. 				
		LinearRegressionAlgorithm<LinearRegressionNormalEquationTrainingContext> linearRegressionAlgorithm = 
				algorithmFactory.createLinearRegressionNormalEquationAlgorithm();
				
		// Create a price (label) predictor for the training set, using this algorithm
		LabelPredictor<House,Number,LinearRegressionNormalEquationTrainingContext> pricePredictor = 
				new SingleNumericValueLabelPredictor<House,LinearRegressionNormalEquationTrainingContext>(labeledTrainingSet,linearRegressionAlgorithm);

		// Add our housing data to the training set
		labeledTrainingSet.add(houses);
		
		// Create a training context for our chosen algorithm, setting the necessary parameters
		LinearRegressionNormalEquationTrainingContext trainingContext = new LinearRegressionNormalEquationTrainingContext();
		trainingContext.setRegularizationLambda(0d);

		// Train the price predictor to learn from training set
		pricePredictor.train(trainingContext);

		// Predict a price for a specified house
		Number predictedPrice = pricePredictor.predictLabel(new House(1650,3));
		
		// Assert the price is not null
		Assert.assertNotNull("Predicted price should not be null",predictedPrice);
		
		// Assert the price matches the same price as predicted by the Octave
		// programming assignment from Stanford's Machine Learning course
		Assert.assertEquals(293081, predictedPrice.intValue());
	}
	
	private Collection<House> getHouseDataFromFile(String fileName)
	{
		TrainingSetDataSource<House,Collection<House>> houses = new CsvFileClassloaderDataSource<House>(fileName,getClass().getClassLoader(),new HouseCsvDataExtractor());
		return houses.getData();
	}
	
}

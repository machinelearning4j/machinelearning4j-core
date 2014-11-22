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
import org.junit.Ignore;
import org.junit.Test;
import org.machinelearning4j.algorithms.AlgorithmFactory;
import org.machinelearning4j.algorithms.DefaultAlgorithmFactory;
import org.machinelearning4j.algorithms.supervisedlearning.NeuralNetworkAlgorithm;
import org.machinelearning4j.algorithms.supervisedlearning.NeuralNetworkAlgorithmTrainingContext;
import org.machinelearning4j.core.Builders;
import org.machinelearning4j.core.DefaultFeatureScaler;
import org.machinelearning4j.util.CsvFileClassloaderDataSource;
import org.machinelearning4j.util.TrainingSetDataSource;

/**
 * Initial end-to-end integration test for classification label prediction.
 * 
 * Uses data from a programming assigment on Stanford's Machine Learning
 * Coursera course to take data consisting of students' exam scores and
 * their admission status, applying Logistic Regression to predict an admission status
 * 
 * @author Michael Lavelle
 */
public class NeuralNetworkClassifierIntegrationTest {

	private Iterable<Application> previousApplications;
	private int trainingSetSize;
	private AlgorithmFactory algorithmFactory;

	
	@Before
	public void setUp() throws Exception
	{
		// Read exam scores and admission status data used as part of a programming assignment on
		// Stanford's Machine Learning course
		Collection<Application> previousApplicationsCollection = getPreviousApplicationDataFromFile("ex2data1.txt");
		
		// Assert that we have read the training data correctly
		Assert.assertNotNull(previousApplicationsCollection);
		Assert.assertEquals(100,previousApplicationsCollection.size());
		Application application1 = previousApplicationsCollection.iterator().next();
		ExamScores examScores1 = application1.getExamScores();
		Assert.assertEquals(34.62365962451697, examScores1.getExamScore1());
		Assert.assertEquals(78.0246928153624, examScores1.getExamScore2());
		Assert.assertEquals(AdmissionStatus.NOT_ACCEPTED, application1.getAdmissionStatus());
	
		
		this.previousApplications = previousApplicationsCollection;
		this.trainingSetSize = previousApplicationsCollection.size();
		this.algorithmFactory = new DefaultAlgorithmFactory();	
	}
	
	@Test
	@Ignore // Ignore for now as libgfortran3 is not available on build machine
	public void testClassificationPrediction()
	{		
		// 1. Create and configure the training set
		LabeledTrainingSet<Application,AdmissionStatus> labeledTrainingSet = 
				Builders.createLabeledTrainingSetBuilder(Application.class,AdmissionStatus.class,trainingSetSize,false)
				.withFeatureDefinition(new ExamScoresFeatureDefinition())
				.withFeatureScaling(new DefaultFeatureScaler())
				.withLabel(new AdmissionStatusLabelDefinition())
				.build();
					
		// 2. Create a logistic regression algorithm that we can use to predict the AdmissionStatus. 
		NeuralNetworkAlgorithm<NeuralNetworkAlgorithmTrainingContext> neuralNetworkAlgorithm = 
				algorithmFactory.createNeuralNetworkAlgorithm();
		
		
		
		// 3. Create an admission status label predictor (classifier) for the training set, using this algorithm
		Classifier<Application,AdmissionStatus,NeuralNetworkAlgorithmTrainingContext> admissionStatusPredictor = 
				new NeuralNetworkClassifier<Application,AdmissionStatus,NeuralNetworkAlgorithmTrainingContext>(labeledTrainingSet,neuralNetworkAlgorithm,new AdmissionStatusMultiLabelMapper(),new AdmissionStatus[] {AdmissionStatus.NOT_ACCEPTED,AdmissionStatus.ACCEPTED});

		// 4. Add our previous applications data to the training set
		labeledTrainingSet.add(previousApplications);
		
		// 5. Create a training context for our chosen algorithm, passing in the max number of iterations we will allow the algorithm to run for
		NeuralNetworkAlgorithmTrainingContext trainingContext = new NeuralNetworkAlgorithmTrainingContext(100,2,new int[] {1},2);
		
		// 6. Add regularisation of our features
		trainingContext.setRegularizationLambda(0.9d);
		

		// 7. Train the admission status predictor to learn from training set
		admissionStatusPredictor.train(trainingContext);

		// 8. Predict a admission status for a specified set of exam scores
		ClassificationProbability<AdmissionStatus> predicitedAdmissionStatus = admissionStatusPredictor.predictLabel(new Application(new ExamScores(45,85)));
		
		// Assert the predicted admission status is not null
		Assert.assertNotNull("Predicted admission status classification probability wrapper should not be null",predicitedAdmissionStatus);
		
		// Assert that we have predicted the student will be accepted - this matches prediction from Stanford's Machine Learning course programming assignment
		Assert.assertEquals(AdmissionStatus.ACCEPTED,predicitedAdmissionStatus.getClassification());
		
		Assert.assertTrue(admissionStatusPredictor.getTrainingSetPredictionAccuracyPercentage() > 85d);

	}
	
	private Collection<Application> getPreviousApplicationDataFromFile(String fileName)
	{
		TrainingSetDataSource<Application,Collection<Application>> applicationsSource = new CsvFileClassloaderDataSource<Application>(fileName,getClass().getClassLoader(),new PreviousApplicationCsvDataExtractor());
		return applicationsSource.getData();
	}
	
}

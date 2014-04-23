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

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Collection;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;
import org.machinelearning4j.algorithms.AlgorithmFactory;
import org.machinelearning4j.algorithms.DefaultAlgorithmFactory;
import org.machinelearning4j.algorithms.supervisedlearning.GradientDescentAlgorithmTrainingContext;
import org.machinelearning4j.algorithms.supervisedlearning.LogisticRegressionAlgorithm;
import org.machinelearning4j.algorithms.supervisedlearning.MonotonicDecreasingCostFunctionSnapshotConvergenceCriteria;
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
public class ClassifierIntegrationTest {

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
	public void testClassificationPrediction()
	{		
		// 1. Create and configure the training set
		LabeledTrainingSet<Application,AdmissionStatus> labeledTrainingSet = 
				Builders.createLabeledTrainingSetBuilder(Application.class,AdmissionStatus.class,trainingSetSize)
				.withFeatureDefinition(new ExamScoresFeatureDefinition())
				.withFeatureScaling(new DefaultFeatureScaler())
				.withLabel(new AdmissionStatusLabelDefinition())
				.build();
					
		// 2. Create a logistic regression algorithm that we can use to predict the AdmissionStatus. 
		LogisticRegressionAlgorithm<GradientDescentAlgorithmTrainingContext> logisticRegressionAlgorithm = 
				algorithmFactory.createLogisticRegressionAlgorithm();
		
		// 3. Create an admission status label predictor (classifier) for the training set, using this algorithm
		Classifier<Application,AdmissionStatus,GradientDescentAlgorithmTrainingContext> admissionStatusPredictor = 
				new BinaryClassifier<Application,AdmissionStatus,GradientDescentAlgorithmTrainingContext>(labeledTrainingSet,logisticRegressionAlgorithm,new AdmissionStatusLabelMapper(),AdmissionStatus.NOT_ACCEPTED,AdmissionStatus.ACCEPTED);

		// 4. Add our previous applications data to the training set
		labeledTrainingSet.add(previousApplications);
		
		// 5. Create a training context for our chosen algorithm, passing in the max number of iterations we will allow the algorithm to run for
		GradientDescentAlgorithmTrainingContext trainingContext = new GradientDescentAlgorithmTrainingContext(1000);
		
		// We don't require regularisation of our features
		trainingContext.setRegularizationLambda(0d);
		
		// Pick an initial learning rate alpha
		trainingContext.setLearningRateAlpha(1d);
		
		// 6. Set convergence criteria
	
		// Collect a snapshot of the cost function's output value every 100 iterations
		trainingContext.setCostFunctionSnapshotIntervalInIterations(100);
		
		// Attempt to use convergence criteria which relies on the cost function snapshot decreasing monotonically
		// Not all algorithms, configurations, or runtime executions will satisfy this condition, in which case training 
		// will stop as the monotonic condition will fail and an alternative strategy for determining convergence can be selected

		// We will try this strategy, and if the monotonic condition does not fail, we will arbitrarily impose an additional requirement that the cost function output should
		// decrease by at least 5% in the first 100 iterations, and that convergence is complete once the cost function
		// output decrease by less than 0.0001% in 100 iterations
		
		//	If cost function does not decrease by 5% in the first 100 iterations, we may choose to revise this requirement,
		// (eg. setting max number of iterations higher), or choose another value of the learning rate.
		
		trainingContext.setConvergenceCriteria(new MonotonicDecreasingCostFunctionSnapshotConvergenceCriteria(5,0.0001d));
	

		// 7. Train the admission status predictor to learn from training set
		admissionStatusPredictor.train(trainingContext);

		// 8. Predict a admission status for a specified set of exam scores
		ClassificationProbability<AdmissionStatus> predicitedAdmissionStatus = admissionStatusPredictor.predictLabel(new Application(new ExamScores(45,85)));
		
		// Assert the predicted admission status is not null
		Assert.assertNotNull("Predicted admission status classification probability wrapper should not be null",predicitedAdmissionStatus);
		
		// Assert that we have predicted the student will be accepted - this matches prediction from Stanford's Machine Learning course programming assignment
		Assert.assertEquals(AdmissionStatus.ACCEPTED,predicitedAdmissionStatus.getClassification());
		
		// Assert that the confidence level we have in the prediction matches the same
		// probability as predicted by the Octave programming assigment from Stanford's Machine Learning course
		BigDecimal classificationProbabiliyValue = new BigDecimal(predicitedAdmissionStatus.getProbability(),new MathContext(3));
		Assert.assertEquals("0.776",classificationProbabiliyValue.toString());
		
		
		Assert.assertEquals(89d, admissionStatusPredictor.getTrainingSetPredictionAccuracyPercentage());

	}
	
	private Collection<Application> getPreviousApplicationDataFromFile(String fileName)
	{
		TrainingSetDataSource<Application,Collection<Application>> houses = new CsvFileClassloaderDataSource<Application>(fileName,getClass().getClassLoader(),new PreviousApplicationCsvDataExtractor());
		return houses.getData();
	}
	
}

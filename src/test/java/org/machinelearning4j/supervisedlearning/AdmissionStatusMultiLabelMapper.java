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
/**
 * A NumericLabelMapper which converts AdmissionStatus labels to 0 for the AdmissionStatus.NOT_ACCEPTED and 1 for AdmissionStatus_ACCEPTED
 * 
 * @author Michael Lavelle
 */
public class AdmissionStatusMultiLabelMapper extends
		BinaryClassificationMultiLabelMapper<AdmissionStatus> {


	@Override
	protected double[] getLabelAsVector(AdmissionStatus label) {
		
		double notAccepted = label == AdmissionStatus.NOT_ACCEPTED ? 1d: 0d;
		double accepted = label == AdmissionStatus.ACCEPTED ? 1d: 0d;

		return new double[] {notAccepted,accepted};
		
	}

}

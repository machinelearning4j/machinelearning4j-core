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

import org.machinelearning4j.util.CsvDataExtractor;
/**
 * Extracts a list of csv attributes into an Application pojo
 * 
 * @author Michael Lavelle
 */
public class PreviousApplicationCsvDataExtractor implements CsvDataExtractor<Application> {

	@Override
	public Application createData(String[] csvAttributes) {
		ExamScores examScores = new ExamScores(Double.parseDouble(csvAttributes[0]),Double.parseDouble(csvAttributes[1]));
		return new Application(examScores,createAdmissionStatus(Integer.parseInt(csvAttributes[2])));
	}
	
	private AdmissionStatus createAdmissionStatus(int flag)
	{
		return flag == 0 ? AdmissionStatus.NOT_ACCEPTED : AdmissionStatus.ACCEPTED;
	}

}

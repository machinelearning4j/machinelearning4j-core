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
 * Represents a student application
 * 
 * @author Michael Lavelle
 */
public class Application {

	private ExamScores examScores;
	private AdmissionStatus admissionStatus;
	
	public Application(ExamScores examScores)
	{
		this.examScores = examScores;
	}
	
	public ExamScores getExamScores() {
		return examScores;
	}

	public AdmissionStatus getAdmissionStatus() {
		return admissionStatus;
	}

	public Application(ExamScores examScores,AdmissionStatus admissionStatus)
	{
		this.examScores = examScores;
		this.admissionStatus = admissionStatus;
	}
}

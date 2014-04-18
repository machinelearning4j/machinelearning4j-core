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
package org.machinelearning4j.core;

import org.machinelearning4j.core.unsupervisedlearning.UnlabeledTrainingSetBuilder;
import org.machinelearning4j.supervisedlearning.LabeledTrainingSetBuilder;

/**
 * Provides static entrypoint for access to builders for configurable components.
 *
 * @author Michael Lavelle
 */
public class Builders {

	public static <T,L> LabeledTrainingSetBuilder<T,L> createLabeledTrainingSetBuilder(Class<T> elementClass,
			Class<L> labelClass, int numberOfTrainingExamplesToProcess) {
		return new LabeledTrainingSetBuilder<T,L>();
	}
	
	public static <T,L> UnlabeledTrainingSetBuilder<T> createUnlabeledTrainingSetBuilder(Class<T> elementClass, int numberOfTrainingExamplesToProcess) {
		return new UnlabeledTrainingSetBuilder<T>();
	}
	

}

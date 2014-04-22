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
 *  Trainable component which learns to predict labels of type L from elements of type T
 * 
 * @author Michael Lavelle
 */
public interface LabelPredictor<T, L,C> {

	/**
	 * Train the label predictor
	 */
	void train(C trainingContext);

	/**
	 * @param The element we wish to predict a label for
	 * @return  The predicted label
	 */
	L predictLabel(T element);

}

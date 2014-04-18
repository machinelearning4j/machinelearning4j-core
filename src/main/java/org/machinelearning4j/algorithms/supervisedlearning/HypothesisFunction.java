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
package org.machinelearning4j.algorithms.supervisedlearning;

/**
 * 
 * @param <X> the type of an input data element which contains the set of independent attributes
 * @param <Y> the type of the dependent output or target data attribute, which this function predicts, given an element of X 
 * 
 * @author Michael Lavelle
 * 
 * A HypothesisFunction<X,Y> encapsulates a function h : X -> Y so that predict(X x) is a "good" predictor
 * given an element of X for the corresponding value of Y
 */
public interface HypothesisFunction<X,Y> {

	public Y predict(X x);
}
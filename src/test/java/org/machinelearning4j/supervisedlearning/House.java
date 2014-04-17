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
 * Domain object representing housing data from an programming assignment from
 * Stanford's Machine Learning course
 * 
 * @author Michael Lavelle
 */
public class House {

	private int squareFeet;
	private int bedrooms;
	private int price;
	
	public House(int squareFeet,int bedrooms)
	{
		this.squareFeet = squareFeet;
		this.bedrooms = bedrooms;
	}
	
	public House(int squareFeet,int bedrooms,int price)
	{
		this.squareFeet = squareFeet;
		this.bedrooms = bedrooms;
		this.price = price;
	}

	public int getPrice() {
		return price;
	}

	public void setPrice(int price) {
		this.price = price;
	}
	
	public int getBedrooms() {
		return bedrooms;
	}

	public void setBedrooms(int bedrooms) {
		this.bedrooms = bedrooms;
	}

	public int getSquareFeet() {
		return squareFeet;
	}

	public void setSquareFeet(int squareFeet) {
		this.squareFeet = squareFeet;
	}
}

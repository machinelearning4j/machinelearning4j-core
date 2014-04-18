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
package org.machinelearning4j.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Creates a collection of elements of type T from a file accessible via a class loader.
 * 
 * Uses a CsvDataExtractor to parse csv attributes into an element
 * 
 * @author Michael Lavelle
 */
public class CsvFileClassloaderDataSource<T> implements TrainingSetDataSource<T,Collection<T>> {

	private CsvDataExtractor<T> dataExtractor;
	private String filePath;
	private ClassLoader classLoader;
	
	/**
	 * @param filePath The file path relative to the classLoader from which to load the data
	 * @param classLoader A classloader to be used to load files
	 * @param dataExtractor A data extractor defining how to parse csv attributes into elements
	 */
	public CsvFileClassloaderDataSource(String filePath,ClassLoader classLoader,CsvDataExtractor<T> dataExtractor) 
	{
		this.dataExtractor = dataExtractor;
		this.filePath = filePath;
		this.classLoader = classLoader;
	}
	
	/**
	 * Retrieve the elements from the csv file as a Collection
	 */
	public Collection<T> getData()
	{
		Collection<T> data = new ArrayList<T>();
		InputStreamReader inputStreamReader = null;
		BufferedReader bufferedReader =null;
		InputStream is = classLoader.getResourceAsStream(filePath);
		try
		{
			if (is == null) throw new FileNotFoundException(filePath);
			inputStreamReader = new InputStreamReader(is);
			bufferedReader = new BufferedReader(inputStreamReader);
			String line = bufferedReader.readLine();
			while(line != null)
			{
				String[] parts = line.split(",");
				data.add(dataExtractor.createData(parts));
				line = bufferedReader.readLine();
			}		
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
		finally
		{
			try {
				if (bufferedReader != null) bufferedReader.close();
			} catch (IOException e) {}
			
			try {
				if (inputStreamReader != null) inputStreamReader.close();
			} catch (IOException e) {}
			try {
				if (is != null) is.close();
			} catch (IOException e) {}
		}
		return data;
	}
	
}

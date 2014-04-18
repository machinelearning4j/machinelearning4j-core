package org.machinelearning4j.supervisedlearning;

import org.machinelearning4j.util.CsvDataExtractor;

public class HouseCsvDataExtractor implements CsvDataExtractor<House> {

	@Override
	public House createData(String[] csvAttributes) {
		return new House(Integer.parseInt(csvAttributes[0]),Integer.parseInt(csvAttributes[1])
				,Integer.parseInt(csvAttributes[2]));
	}

}

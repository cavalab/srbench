/**
 * Copyright (c) 2011-2013 Evolutionary Design and Optimization Group
 * 
 * Licensed under the MIT License.
 * 
 * See the "LICENSE" file for a copy of the license.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.  
 *
 */
package evogpj.evaluation.java;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import evogpj.evaluation.DataSizeRetreiver;

/**
 * Class for ingesting data from a Comma Separated Value (CSV) text file. All
 * data is assumed to be doubles.
 * 
 * @author Owen Derby
 */
public class CSVDataJava extends ScaledData {
	/**
	 * Parse given csvfile into set of input and target values.
	 * 
	 * @param csvfile file of comma-separated values, last value in each line is
	 *        the target value
	 */
	public CSVDataJava(String csvfile) {
		super(DataSizeRetreiver.num_fitness_cases(csvfile), DataSizeRetreiver.num_terminals(csvfile));
		BufferedReader f;
		try {
                    f = new BufferedReader(new FileReader(csvfile));
                    String[] token;
                    try {
                            /**
                             * Keep track of max/min target values (Vladislavleva suggested) to
                             * perform approximate scaling.
                             */
                            int fitnessCaseIndex = 0;
                            while (f.ready() && fitnessCaseIndex < numberOfFitnessCases) {
                                    //token = f.readLine().split("\\t");
                                    token = f.readLine().split(",");
                                    for (int i = 0; i < token.length - 1; i++) {
                                            this.fitnessCases[fitnessCaseIndex][i] = Double.valueOf(token[i]);
                                            if(fitnessCases[fitnessCaseIndex][i] < minFeatures[i]) minFeatures[i] = fitnessCases[fitnessCaseIndex][i];
                                            if(fitnessCases[fitnessCaseIndex][i] > maxFeatures[i]) maxFeatures[i] = fitnessCases[fitnessCaseIndex][i];
                                    }
                                    Double val = Double.valueOf(token[token.length - 1]);
                                    addTargetValue(val, fitnessCaseIndex);
                                    fitnessCaseIndex++;
                            }
                            this.scaleTarget();
                    } catch (NumberFormatException e) {
                    } catch (IOException e) {
                    }
                } catch (FileNotFoundException e) {
                	System.exit(-1);
                }
	}

	public CSVDataJava(List<String> data){
		//super(data.size(),data.get(0).split("\\t").length-1);
                super(data.size(),data.get(0).split(",").length-1);
		int fitnessCaseIndex = 0;
		String[] token;
		for(int index=0;index<numberOfFitnessCases;index++){
			//token=data.get(index).split("\\t");
                        token=data.get(index).split(",");
			for (int i = 0; i < token.length - 1; i++) {
                this.fitnessCases[fitnessCaseIndex][i] = Double.valueOf(token[i]);
                if(fitnessCases[fitnessCaseIndex][i] < minFeatures[i]) minFeatures[i] = fitnessCases[fitnessCaseIndex][i];
                if(fitnessCases[fitnessCaseIndex][i] > maxFeatures[i]) maxFeatures[i] = fitnessCases[fitnessCaseIndex][i];
			}
			Double val = Double.valueOf(token[token.length - 1]);
			addTargetValue(val, fitnessCaseIndex);
			fitnessCaseIndex++;
		}
		this.scaleTarget();
	}
	
	/**
	 * Display length/width of data
	 */
	public void printDataInfo() {
		System.out.println("We have " + this.numberOfFitnessCases + " fitness cases and " + this.numberOfFeatures+ " values");
	}
}
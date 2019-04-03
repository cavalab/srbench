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

import evogpj.math.means.ArithmeticMean;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Class which provides DataJava interface with target values scaled to the range
 * [0, 1]. Using data with this transformation frees GP from having to search
 * for the proper scale, allowing it to focus on just identifying the proper
 * shape. See E. Y. Vladislavleva, 창��밠odel-based problem solving through symbolic
 * regression via pareto genetic programming,창���CentER, Tilburg University, 2008.
 * 
 * @author Owen Derby
 */
public abstract class ScaledData implements DataJava {
	
	/**
	 * the number of fitness cases
	 */
        protected int numberOfFitnessCases;
        /**
	 * the number of features
	 */
        protected int numberOfFeatures;
        /**
	 * the datapoints (input values) to evaluate individual's fitness on.
	 */
	protected final double[][] fitnessCases;
	/**
	 * the datapoints (output values) to compare individual's output against to
	 * determine their fitness
	 */
	protected final double[] target;
	/**
	 * In addition, keep a copy of the target which is scaled to be in the range
	 * [0, 1] This is from Vladislavleva, to "allow the GP to focus on finding
	 * the structure of the solution, instead of the scale"
	 */
	protected final double[] scaled_target;
	private Double target_min;
	private Double target_max;
        protected double[] minFeatures;
        protected double[] maxFeatures;
        
	/**
	 * A running mean for target values, used in ModelScaler
	 */
	public ArithmeticMean targetMean;

	public ScaledData(int aNumberOfFitnessCases, int aNumberOfFeatures) {
                numberOfFitnessCases = aNumberOfFitnessCases;
                numberOfFeatures = aNumberOfFeatures;
		fitnessCases = new double[numberOfFitnessCases][numberOfFeatures];
		this.target = new double[numberOfFitnessCases];
		this.scaled_target = new double[numberOfFitnessCases];
		target_min = null;
		target_max = null;
		targetMean = new ArithmeticMean();
                minFeatures = new double[numberOfFeatures];
                maxFeatures = new double[numberOfFeatures];
                for(int i=0;i<numberOfFeatures;i++){
                    minFeatures[i] = Double.MAX_VALUE;
                    maxFeatures[i] = - Double.MAX_VALUE;
                }
                
                
	}

	protected void addTargetValue(Double val,int index) {
		this.target[index] = val;
		// keep a running mean
		targetMean.addValue(val);
		if (target_min == null || val < target_min) {
			target_min = val;
		}
		if (target_max == null || val > target_max) {
			target_max = val;
		}
	}

        @Override
	public Double getTargetMean() {
		return this.targetMean.getMean();
	}

	/**
	 * Call this after you've added all target values to target
	 */
	protected void scaleTarget() {
		double range = target_max - target_min;
		for (int i = 0; i < this.target.length; i++) {
                    Double val = (this.target[i] - target_min) / range;
                    this.scaled_target[i] = val;
		}
	}
        
    public void normalizeValues(String newFilePath,String pathToBounds) throws IOException{
        BufferedWriter bw = new BufferedWriter(new FileWriter(newFilePath));
        PrintWriter printWriter = new PrintWriter(bw);
        for(int i=0;i<numberOfFitnessCases;i++){
            for(int j=0;j<numberOfFeatures;j++){
                double range = maxFeatures[j] - minFeatures[j];
                Double val = (fitnessCases[i][j] - minFeatures[j]) / range;
                printWriter.write(val + ",");
            }
            double targetValue = this.scaled_target[i];
            printWriter.write(targetValue + "\n");
        }
        printWriter.flush();
        printWriter.close();
        
        BufferedWriter bwNormCoeffs = new BufferedWriter(new FileWriter(pathToBounds));
        PrintWriter printWriterNormCoeffs = new PrintWriter(bwNormCoeffs);
        for(int j=0;j<numberOfFeatures;j++){
            printWriterNormCoeffs.print(minFeatures[j] + " " + maxFeatures[j]+ "\n");
        }
        printWriterNormCoeffs.print(target_min + " " + target_max + "\n");
        printWriterNormCoeffs.flush();
        printWriterNormCoeffs.close();
    }
     
//    /**
//     * Partitions the given data into subsets according to the class labels.
//     * Call this after the targets are scaled. (Only for Binary Classification use)
//     * @return Map of subsets of data
//     */
//    public Map<Double,double[][]> partition() {
//    	Map<Double,double[][]> partitioned = new HashMap<Double,double[][]>();
//    	
//    	int numOfNegativeData = 0;
//    	int numOfPositiveData = 0;
//    	double negativeLabel = this.target_min;
//    	double positiveLabel = this.target_max;
//    	
//    	for (int i=0;i<numberOfFitnessCases;i++){
//    		if(this.scaled_target[i]==negativeLabel){
//    			numOfNegativeData++;
//    		}else if(this.scaled_target[i]==positiveLabel){
//    			numOfPositiveData++;
//    		}else{
//    			System.err.println("Something is wrong with class label!");
//    			System.exit(-1);
//    		}
//    	}
//    	
//    	double[][] negative_class = new double[numOfNegativeData][numberOfFeatures];
//    	double[][] positive_class = new double[numOfPositiveData][numberOfFeatures];
//    	
//    	int indexOfNegative = 0;
//    	int indexOfPositive = 0;
//    	
//    	for(int i=0;i<numberOfFitnessCases;i++){
//    		if(this.scaled_target[i]==negativeLabel){
//    			negative_class[indexOfNegative]=this.fitnessCases[i];
//    			indexOfNegative++;
//    		}else if(this.scaled_target[i]==positiveLabel){
//    			positive_class[indexOfPositive]=this.fitnessCases[i];
//    			indexOfPositive++;
//    		}else{
//    			System.err.println("Something is wrong with class label!");
//    			System.exit(-1);
//    		}
//    	}
//    	
//    	partitioned.put(negativeLabel,negative_class);
//    	partitioned.put(positiveLabel,positive_class);
//    	
//    	return partitioned;
//    }
    
    @Override
    public double[][] getInputValues(){
        return fitnessCases;
    }

    @Override
    public double[] getTargetValues(){
        return target;
    }

    @Override
    public double[] getScaledTargetValues(){
        return scaled_target;
    }
        


    @Override
    public Double getTargetMax() {
        return target_max;
    }

    @Override
    public Double getTargetMin() {
        return target_min;
    }

    /**
     * @return the numberOfFitnessCases
     */
    @Override
    public int getNumberOfFitnessCases() {
        return numberOfFitnessCases;
    }

    /**
     * @return the numberOfFeatures
     */
    @Override
    public int getNumberOfFeatures() {
        return numberOfFeatures;
    }
        
        
}
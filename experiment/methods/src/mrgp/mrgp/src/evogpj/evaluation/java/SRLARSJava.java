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
 * @author Ignacio Arnaldo
 * 
 */
package evogpj.evaluation.java;

import edu.uci.lasso.LassoFit;
import edu.uci.lasso.LassoFitGenerator;
import java.util.ArrayList;
import java.util.List;

import evogpj.genotype.Tree;
import evogpj.gp.Individual;
import evogpj.gp.Population;
import evogpj.math.Function;
import evogpj.math.means.ArithmeticMean;
import evogpj.math.means.Maximum;
import evogpj.math.means.Mean;
import evogpj.math.means.PowerMean;
import evogpj.algorithm.Parameters;
import evogpj.evaluation.FitnessFunction;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * MRGP evaluation of symbolic regression models
 * @author Ignacio Arnaldo
 */
public class SRLARSJava extends FitnessFunction {

    private final DataJava data;
    private int pow;
    private final boolean USE_INT;
    public static String FITNESS_KEY = Parameters.Operators.SR_JAVA_FITNESS;
    public Boolean isMaximizingFunction = true;
    private int numThreads;
    /**
     * Create a new fitness operator, using the provided data, for assessing
     * individual solutions to Symbolic Regression problems. There are two
     * parameters for this fitness evaluation:
     * <p>
     * <ul>
     * <li>The power <i>p</i> to use in computing the mean of the errors. See
     * {@link #getMeanFromP(int)} for recognized values and more information.
     * This parameter is set to the value specified by the key
     * {@value algorithm.Parameters.Names#MEAN_POW} in the properties file. The
     * default value is {@value algorithm.Parameters.Defaults#MEAN_POW}.
     * <li>A boolean flag specifying if the predictions should be forced to
     * integers because the output variable is integer-valued (ie you might want
     * to set this to true if your output is integer-valued). This parameter is
     * set to the boolean at the key
     * {@value algorithm.Parameters.Names#COERCE_TO_INT}. If there is no value
     * specified, the default is
     * {@value algorithm.Parameters.Defaults#COERCE_TO_INT}.
     * </ul>
     * 
     * @param data
     *            The dataset (training cases, output variable) to use in
     *            computing the fitness of individuals.
     */
    public SRLARSJava(DataJava data) {
        this(data, 2, false,1);
    }

    public SRLARSJava(DataJava aData, int aPow, boolean is_int,int anumThreads) {
        this.data = aData;
        pow = aPow;
        USE_INT = is_int;
        numThreads = anumThreads;
    }
    /**
     * Should this fitness function be minimized (i.e. mean squared error) or
     * maximized?
     * @return 
     */
    @Override
    public Boolean isMaximizingFunction() {
        return this.isMaximizingFunction;
    }

    /**
     * Simple "factory"-like method for returning the proper generalized mean
     * object to use, given the parameter p (see
     * http://en.wikipedia.org/wiki/Generalized_mean). Since we are only
     * concerned with means of errors, we don't utilize all the valid values of
     * p defined for generalized means.
     * 
     * @param p
     *            power to use in computing a mean
     * @return an instance of {@link PowerMean} if p > 1; an instance of
     *         {@link ArithmeticMean} if p == 1; otherwise, an instance of
     *         {@link Maximum}.
     */
    public static Mean getMeanFromP(int p) {
        if (p == 1) {
            return new ArithmeticMean();
        } else if (p > 1) {
            return new PowerMean(p);
        } else {
            return new Maximum();
        }
    }

    /**
     * @param ind
     * @see Function
     */
    public void eval(Individual ind) throws Exception {
        
        Tree genotype = (Tree) ind.getGenotype();
        Mean MEAN_FUNC = getMeanFromP(pow);
        Function func = genotype.generate();
        List<Double> d;
        ArrayList<Double> interVals;
        double[][] inputValuesAux = data.getInputValues();
        Tree tAux = (Tree) ind.getGenotype();
        //double[] targetAux = data.getScaledTargetValues();
        double[] targetAux = data.getTargetValues();
        float[][] intermediateValues = new float[data.getNumberOfFitnessCases()][tAux.getSize()];
        for (int i = 0; i < data.getNumberOfFitnessCases(); i++) {
            d = new ArrayList<Double>();
            for (int j = 0; j < data.getNumberOfFeatures(); j++) {
                d.add(j, inputValuesAux[i][j]);
            }
            interVals = new ArrayList<Double>();
            Double val = func.evalIntermediate(d,interVals);
            for(int t=0;t<interVals.size();t++){
                intermediateValues[i][t] = interVals.get(t).floatValue();
            }
            d.clear();
            interVals.clear();
        }

        /*
         * LassoFitGenerator is initialized
         */
        LassoFitGenerator fitGenerator = new LassoFitGenerator();
        int numObservations = data.getNumberOfFitnessCases();
        fitGenerator.init(tAux.getSize(), numObservations);
        for (int i = 0; i < numObservations; i++) {
            fitGenerator.setObservationValues(i,intermediateValues[i]);
            fitGenerator.setTarget(i, targetAux[i]);
        }

        /*
         * Generate the Lasso fit. The -1 arguments means that
         * there would be no limit on the maximum number of 
         * features per model
         */
        LassoFit fit = fitGenerator.fit(-1);

        
        // We pick the first value of lamda that includes all the features in the model
        int indexWeights = 0;
        int usedVars=0;
        for(int i=0;i<fit.lambdas.length;i++){
            if(fit.nonZeroWeights[i]>usedVars){
                indexWeights = i;
                usedVars = fit.nonZeroWeights[i];
            }
        }
        ArrayList<String> alWeights = new ArrayList<String>();
        double lassoIntercept = 0;
        double fitness = 0;
        //if(fit.nonZeroWeights[indexWeights]==tAux.getSize()){
            //double[] lassoWeights = fit.compressedWeights[indexWeights];
            double[] lassoWeights = fit.getWeights(indexWeights);
            for(int j=0;j<lassoWeights.length;j++){
                alWeights.add(Double.toString(lassoWeights[j]));
            }
            lassoIntercept = fit.intercepts[indexWeights];
            //SRPhenotype phenotype_tmp = new SRPhenotype();
            for (int i = 0; i < data.getTargetValues().length; i++) {
                double prediction = 0;
                for(int j=0;j<lassoWeights.length;j++){
                    prediction += intermediateValues[i][j]*lassoWeights[j];
                }
                prediction += lassoIntercept;
                //phenotype_tmp.addNewDataValue(prediction);
                if (this.USE_INT) {
                    prediction = Math.round(prediction);
                }
                MEAN_FUNC.addValue(Math.abs(targetAux[i] - prediction));
            }
            /*final ScaledTransform scalePrediction = new ScaledTransform(phenotype_tmp.min, phenotype_tmp.max);
            final ScaledTransform scaleTarget = new ScaledTransform(this.data.getTargetMin(), this.data.getTargetMax());
            for (int i = 0; i < this.getTarget().length; i++) {
                Double scaled_val = scalePrediction.scaleValue(phenotype_tmp.getDataValue(i));
                if (this.USE_INT) {
                    int rounded_val = scaleTarget.unScaleValue(scaled_val).intValue();
                    scaled_val = scaleTarget.scaleValue((double) rounded_val);
                }
                MEAN_FUNC.addValue(Math.abs(targetAux[i] - scaled_val));
            }
            phenotype_tmp = null;
            */
            Double error = MEAN_FUNC.getMean();
            fitness=errorToFitness(error);
            
        /*}else{
            for(int j=0;j<tAux.getSize();j++){
                alWeights.add("0");
            }
        }*/
        ind.setWeights(alWeights);
        ind.setLassoIntercept(Double.toString(lassoIntercept));
        ind.setFitness(SRLARSJava.FITNESS_KEY, fitness);
        func = null;
}
    
    /**
     * Transform errors to fitness values. For errors, smaller values are
     * better, while for fitness, values closer to 1 are better. This particular
     * transformation also places a greater emphasis on small changes close to
     * the solution (fitness == 1.0 represents a complete solution). However,
     * this transformation also assumes that the error is in the range [0,1].
     * 
     * @param error
     *            Error on training set; value in range [0,1].
     * @return 0 if error is not a number (NaN) or outside of the range [0,1].
     *         Otherwise, return the fitness.
     */
    private Double errorToFitness(Double error) {
        if(error==0){
            return 1.0;
        }else if (error.isNaN() || error < 0.0 ) {
            return 0.0;
        } else {
            return (1.0) / error;
        }
    }

    @Override
    public void evalPop(Population pop) {
        
        ArrayList<SRJavaThread> alThreads = new ArrayList<SRJavaThread>();
        for(int i=0;i<numThreads;i++){
            SRJavaThread threadAux = new SRJavaThread(i, pop,numThreads);
            alThreads.add(threadAux);
        }
        
        for(int i=0;i<numThreads;i++){
            SRJavaThread threadAux = alThreads.get(i);
            threadAux.start();
        }
        
        for(int i=0;i<numThreads;i++){
            SRJavaThread threadAux = alThreads.get(i);
            try {
                threadAux.join();
            } catch (InterruptedException ex) {
                Logger.getLogger(SRLARSJava.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        

    }

    /**
     * @return the fitnessCases
     */
    public double[][] getFitnessCases() {
        return data.getInputValues();
    }

    /**
     * @return the scaled_target
     */
    public double[] getTarget() {
        return data.getScaledTargetValues();
    }

    /**
     * Utility class for computing the element-wise normalization of a vector.
     * That is, for a given vector <code>A</code>, where all values are in the
     * range <code>[min(A), max(A)]</code>, we want an easy way to compute the
     * scaled vector <code>B</code> such that every value in B is in the range
     * <code>[0, 1]</code>.
     * <p>
     * In particular, if the ith element of <code>A</code> has value
     * <code>a_i</code>, then the ith element of <code>B</code> will have the
     * value <code>(a_i-min(A))/(max(A)-min(A))</code>.
     * <p>
     * Also, provide the inverse transformation, to recover the values of
     * <code>A</code>.
     * 
     * @author Owen Derby
     * @see ScaledData
     */
    private class ScaledTransform {
        private final double min, range;

        public ScaledTransform(double min, double max) {
            this.min = min;
            this.range = max - min;
        }

        public Double scaleValue(Double val) {
            return (val - min) / range;
        }

        public Double unScaleValue(Double scaled_val) {
            return (scaled_val * range) + min;
        }
    }
    
    public class SRJavaThread extends Thread{
        private int indexThread, totalThreads;
        private Population pop;
        
        public SRJavaThread(int anIndex, Population aPop,int aTotalThreads){
            indexThread = anIndex;
            pop = aPop;
            totalThreads = aTotalThreads;
        }

        @Override
        public void run(){
            int indexIndi = 0;
            for (Individual individual : pop) {
                if(indexIndi%totalThreads==indexThread){
                    try {
                        eval(individual);
                    } catch (Exception ex) {
                        Logger.getLogger(SRLARSJava.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
                indexIndi++;
            }
        }
     }
        
}

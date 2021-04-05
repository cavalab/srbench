/**
 * Copyright (c) 2011-2013 ALFA Group
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
package evogpj.test;

import evogpj.evaluation.java.CSVDataJava;
import evogpj.evaluation.java.DataJava;
import evogpj.genotype.Tree;
import evogpj.genotype.TreeGenerator;
import evogpj.gp.Individual;
import evogpj.gp.Population;

import java.util.ArrayList;
import java.util.List;

import evogpj.math.Function;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * Test MRGP models.
 * 
 * @author Ignacio Arnaldo
 */
public class TestRGPModels {
    
    private String pathToData;
    private final DataJava data;
        
    private String pathToPop;
    private Population models;
    
    private boolean round;
    
    private double minTarget, maxTarget;
    /**
     * Create a new fitness operator, using the provided data, for assessing
     * individual solutions to Symbolic Regression problems. There is one
     * parameter for this fitness evaluation:
     * @param aPathToData
     * @param aPathToPop
     * @param aRound
     * @throws java.io.IOException
     * @throws java.lang.ClassNotFoundException
     */
    public TestRGPModels(String aPathToData, String aPathToPop,boolean aRound) throws IOException, ClassNotFoundException {
        pathToData = aPathToData;
        pathToPop = aPathToPop;
        round = aRound;
        this.data = new CSVDataJava(pathToData+"-test");
        //this.data = new CSVDataJava(pathToData);
        readScaledModels(pathToPop);
    }

    
    private void readScaledModels(String filePath) throws IOException, ClassNotFoundException{
        models = new Population();
        ArrayList<String> alModels = new ArrayList<String>();
        Scanner sc = new Scanner(new FileReader(filePath));
        int indexModel =0;
        while(sc.hasNextLine()){
            String sAux = sc.nextLine();
            alModels.add(indexModel, sAux);
            indexModel++;
        }
        int popSize = alModels.size();
        for(int i=0;i<popSize;i++){
            String scaledModel = alModels.get(i);
            String[] tokens = scaledModel.split(",");
            minTarget = Double.parseDouble(tokens[0]);
            maxTarget = Double.parseDouble(tokens[1]);
            String[] weightsArrayS = tokens[2].split(" ");
            ArrayList<String> alWeights = new ArrayList<String>();
            
            
            for(int j=0;j<weightsArrayS.length;j++){
                alWeights.add(weightsArrayS[j]);
            }
            String interceptS = tokens[3];
            String model = tokens[4];
            Tree g = TreeGenerator.generateTree(model);
            Individual iAux = new Individual(g);
            iAux.setWeights(alWeights);
            iAux.setLassoIntercept(interceptS);
            models.add(i, iAux);
        }
    }
    
    /**
     * @see Function
     */
    public void predictionsPop(String filePath) throws IOException {
        
        int indexIndi = 0;
        
        double[] targets = data.getTargetValues();
        for(Individual ind:models){
            BufferedWriter bw = new BufferedWriter(new FileWriter(filePath + "_" + indexIndi + ".csv"));
            PrintWriter printWriter = new PrintWriter(bw);
            ArrayList<String> alWeights = ind.getWeights();
            double[] lassoWeights = new double[alWeights.size()];
            for(int i=0;i<alWeights.size();i++){
                lassoWeights[i] = Double.parseDouble(alWeights.get(i));
            }
            double lassoIntercept = Double.parseDouble(ind.getLassoIntercept());
            double sqDiff = 0;
            double absDiff = 0;
            Tree genotype = (Tree) ind.getGenotype();
            Function func = genotype.generate();
            List<Double> d;
            ArrayList<Double> interVals;
            double[][] inputValuesAux = data.getInputValues();
            float[][] intermediateValues = new float[data.getNumberOfFitnessCases()][genotype.getSize()];
            for (int i = 0; i < data.getNumberOfFitnessCases(); i++) {
                d = new ArrayList<Double>();
                for (int j = 0; j < data.getNumberOfFeatures(); j++) {
                    d.add(j, inputValuesAux[i][j]);
                }
                interVals = new ArrayList<Double>();
                func.evalIntermediate(d,interVals);
                for(int t=0;t<interVals.size();t++){
                    intermediateValues[i][t] = interVals.get(t).floatValue();
                }
                double prediction = 0;
                for(int j=0;j<lassoWeights.length;j++){
                    prediction += intermediateValues[i][j]*lassoWeights[j];
                }
                prediction += lassoIntercept;
                if (round) prediction = Math.round(prediction);          
                if(prediction<minTarget) prediction = minTarget;
                if(prediction>maxTarget) prediction = maxTarget;
                d.clear();
                interVals.clear();
                printWriter.println(prediction);
            }
            printWriter.flush();
            printWriter.close();
            func = null;
            indexIndi++;
        }
    }    
    
    /**
     * @see Function
     */
    public void evalPop() {
        double[] targets = data.getTargetValues();
        for(Individual ind:models){
            ArrayList<String> alWeights = ind.getWeights();
            double[] lassoWeights = new double[alWeights.size()];
            for(int i=0;i<alWeights.size();i++){
                lassoWeights[i] = Double.parseDouble(alWeights.get(i));
            }
            double lassoIntercept = Double.parseDouble(ind.getLassoIntercept());
            double sqDiff = 0;
            double absDiff = 0;
            Tree genotype = (Tree) ind.getGenotype();
            Function func = genotype.generate();
            List<Double> d;
            ArrayList<Double> interVals;
            double[][] inputValuesAux = data.getInputValues();
            float[][] intermediateValues = new float[data.getNumberOfFitnessCases()][genotype.getSize()];
            for (int i = 0; i < data.getNumberOfFitnessCases(); i++) {
                d = new ArrayList<Double>();
                for (int j = 0; j < data.getNumberOfFeatures(); j++) {
                    d.add(j, inputValuesAux[i][j]);
                }
                interVals = new ArrayList<Double>();
                func.evalIntermediate(d,interVals);
                for(int t=0;t<interVals.size();t++){
                    intermediateValues[i][t] = interVals.get(t).floatValue();
                }
                double prediction = 0;
                for(int j=0;j<lassoWeights.length;j++){
                    prediction += intermediateValues[i][j]*lassoWeights[j];
                }
                prediction += lassoIntercept;
                //phenotype_tmp.addNewDataValue(prediction);
                if (round) prediction = Math.round(prediction);
                if(prediction<minTarget) prediction = minTarget;
                if(prediction>maxTarget) prediction = maxTarget;
                d.clear();
                interVals.clear();
                System.out.print(prediction+" ");
                sqDiff += Math.pow(targets[i] - prediction, 2);
                absDiff += Math.abs(targets[i] - prediction);
                d.clear();
            }
            sqDiff = sqDiff / data.getNumberOfFitnessCases();
            absDiff= absDiff / data.getNumberOfFitnessCases();
            ind.setScaledMSE(sqDiff);
            ind.setScaledMAE(absDiff);
            func = null;
        }
    }
    
    public void saveModelsToFile(String filePath) throws IOException{
        BufferedWriter bw = new BufferedWriter(new FileWriter(filePath));
        PrintWriter printWriter = new PrintWriter(bw);
        
        for(Individual ind:models){
////            System.out.print( ind.toString() + "\nMSE: " + ind.getScaledMSE() + "\nMAE: " + ind.getScaledMAE() + "\n");            printWriter.write(ind.toString() + "\nMSE: " + ind.getScaledMSE() + "\nMAE: " + ind.getScaledMAE() + "\n"); 
        }

        printWriter.flush();
        printWriter.close();
    }



}
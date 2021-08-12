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
package main;

import evogpj.algorithm.Parameters;
import evogpj.algorithm.SymbRegMOO;
import evogpj.gp.Individual;
import evogpj.test.TestRGPModels;
import java.io.File;
import java.io.IOException;
import java.util.Properties;

/**
 *
 * @author Ignacio Arnaldo
 */
public class SRLearnerMenuManager {
    
    public SRLearnerMenuManager(){
        
    }
    
    public void printUsage(){
        System.err.println();
        System.err.println("USAGE:");
        System.err.println();
        System.err.println("TRAIN:");
        System.err.println("java -jar sr.jar -train path_to_data -minutes min [-properties path_to_properties]");
        System.err.println();
        System.err.println("OBTAIN PREDICTIONS:");
        System.err.println("java -jar sr.jar -predict path_to_data -o path_to_predictions -integer true -scaled path_to_scaled_models");
        System.err.println();
        System.err.println("TEST:");
        System.err.println("java -jar sr.jar -test path_to_data");
        System.err.println("or");
        System.err.println("java -jar sr.jar -test path_to_data -integer true -scaled path_to_scaled_models");
        System.err.println();
    }
    
    public void parseSymbolicRegressionTrain(String args[]) throws IOException{
        String dataPath;
        int numMinutes = 10*60;
        String propsFile = "";
        SymbRegMOO srEvoGPj;
        dataPath = args[1];
        String generations = args[2];
        String pop_size = args[3];
        String  mut_rate = args[4];
        String  crossover_rate = args[5];
        String  max_length = args[6];
        String  external_threads = "4";
        String  rng_seed = "";
        // Add a time limit if specified by user
        if (args.length >= 8){
        	    numMinutes = Integer.parseInt(args[7]);
	    }
	// Add a maximum number of threads if specified by user
        if (args.length >= 9){
                   external_threads = args[8];
           }
      	// Add a seed if specified by user
        if (args.length >= 10){
                   rng_seed = args[9];
           }

        // run evogpj with standard properties
        Properties props = new Properties();
        //System.out.print(dataPath);
        props.put(Parameters.Names.PROBLEM, dataPath);
        props.put(Parameters.Names.NUM_GENS, generations);
        props.put(Parameters.Names.POP_SIZE, pop_size);      
        props.put(Parameters.Names.MUTATION_RATE, mut_rate);
        props.put(Parameters.Names.XOVER_RATE, crossover_rate);
        props.put(Parameters.Names.TREE_INIT_MAX_DEPTH, max_length);
        props.put(Parameters.Names.TREE_MUTATE_MAX_DEPTH, max_length);
        props.put(Parameters.Names.TREE_XOVER_MAX_DEPTH, max_length);
        if (args.length >= 9){
                   props.put(Parameters.Names.EXTERNAL_THREADS, external_threads);
           }
        if (args.length >= 10){
                   props.put(Parameters.Names.SEED, rng_seed);
           }

/*            if (args[2].equals("-minutes")) {
                ;
                if(args.length==4){// JAVA NO PROPERTIES
                    // run evogpj with standard properties
                    srEvoGPj = new SymbRegMOO(props,numMinutes*60);
                    Individual bestIndi = srEvoGPj.run_population();
                }else if(args.length==6){
                    if(args[4].equals("-properties")){ // JAVA WITH PROPERTIES
                        propsFile = args[5];
                        // run evogpj with properties file and modified properties
                        srEvoGPj = new SymbRegMOO(props,propsFile,numMinutes*60);
                        Individual bestIndi = srEvoGPj.run_population();
                    }else{
                        System.err.println("Error: wrong argument. Expected -cpp flag");
                        printUsage();
                        System.exit(-1);
                    }
                }
            }else{
                System.err.println("Error: must specify the optimization time in minutes");
                printUsage();
                System.exit(-1);
            }

        }else{
            System.err.println("Error: wrong number of arguments");
            printUsage();
            System.exit(-1);
        }
     */
        //allow each job to run around numMinutes
        srEvoGPj = new SymbRegMOO(props,numMinutes);
        Individual bestIndi = srEvoGPj.run_population();

    }
    
    
    //java -jar evogpj.jar -predictions path_to_data -o filename -integer true -scaled path_to_scaled_models
    public void parseSymbolicRegressionPredictions(String args[]) throws IOException, ClassNotFoundException{
        String dataPath;
        String popPath;
        String predPath;
        boolean integerTarget;
        if(args.length==8){
            dataPath = args[1];
            if(args[2].equals("-o")){
                predPath = args[3];
                if(args[4].equals("-integer")){
                    integerTarget = Boolean.valueOf(args[5]);
                    popPath = args[7];
                    if(args[6].equals("-scaled")){
                        TestRGPModels tsm = new TestRGPModels(dataPath, popPath,integerTarget);
                        tsm.predictionsPop(predPath);
                    }else{
                        System.err.println("Error: wrong argument. Expected -scaled flag");
                        printUsage();
                        System.exit(-1);
                    }
                }else{
                    System.err.println("Error: wrong argument. Expected -integer flag");
                    printUsage();
                    System.exit(-1);
                }
            }else{
                System.err.println("Error: wrong argument. Expected -o flag");
                printUsage();
                System.exit(-1);
            }
        }else {
            System.err.println("Error: wrong number of arguments");
            printUsage();
            System.exit(-1);
        }
        
    }
    
    //java -jar evogpj.jar -test path_to_data -integer true -scaled path_to_scaled_models
    public void parseSymbolicRegressionTest(String args[]) throws IOException, ClassNotFoundException{
        String dataPath;
        String popPath;
        boolean integerTarget;
        if (args.length==2){
            // by default integer targets = false
            integerTarget = false;
            dataPath = args[1];
            // check if knee model exists
            /*
            popPath = "knee.txt";
            System.out.println();
            if(new File(popPath).isFile()){
                System.out.println("TESTING KNEE MODEL:");
                TestRGPModels tsm = new TestRGPModels(dataPath, popPath,integerTarget);
                tsm.evalPop();
                tsm.saveModelsToFile("test"+popPath);
                System.out.println();
            }*/
            //popPath = "mostAccurate.txt";
            popPath = dataPath+"-best";
            //System.out.print(popPath);
            //System.out.print(dataPath);
            if(new File(popPath).isFile()){
                //System.out.println("TESTING MOST ACCURATE MODEL: ");
                TestRGPModels tsm = new TestRGPModels(dataPath, popPath,integerTarget);
                tsm.evalPop();
                ////tsm.saveModelsToFile(popPath);
                //System.out.println();
            }
            /*
            popPath = "leastComplex.txt";
            if(new File(popPath).isFile()){
                System.out.println("TESTING SIMPLEST MODEL: ");
                TestRGPModels tsm = new TestRGPModels(dataPath, popPath,integerTarget);
                tsm.evalPop();
                tsm.saveModelsToFile("test"+popPath);
                System.out.println();
            }
            popPath = "pareto.txt";
            if(new File(popPath).isFile()){
                System.out.println("TESTING PARETO MODELS: ");
                TestRGPModels tsm = new TestRGPModels(dataPath, popPath,integerTarget);
                tsm.evalPop();
                tsm.saveModelsToFile("test"+popPath);
                System.out.println();
            }
*/
        }else if(args.length==6){
            dataPath = args[1];
            if(args[2].equals("-integer")){
                integerTarget = Boolean.valueOf(args[3]);
                popPath = args[5];
                if(args[4].equals("-scaled")){
                    TestRGPModels tsm = new TestRGPModels(dataPath, popPath,integerTarget);
                    tsm.evalPop();
                    tsm.saveModelsToFile("test"+popPath);
                }else{
                    System.err.println("Error: wrong argument. Expected -scaled or -fused flag");
                    printUsage();
                    System.exit(-1);
                }
            }else{
                System.err.println("Error: wrong argument. Expected -integer flag");
                printUsage();
                System.exit(-1);
            }
        }else {
            System.err.println("Error: wrong number of arguments");
            printUsage();
            System.exit(-1);
        }
        
    }
    
    
    public static void main(String args[]) throws IOException, ClassNotFoundException, InterruptedException{
        SRLearnerMenuManager m = new SRLearnerMenuManager();
        if (args.length == 0) {
            System.err.println("Error: too few arguments");
            m.printUsage();
            System.exit(-1);
        }else{
            switch (args[0]) {
                case "-train":
                    m.parseSymbolicRegressionTrain(args);
                    break;
                case "-predict":
                    m.parseSymbolicRegressionPredictions(args);
                    break;
                case "-test":
                    m.parseSymbolicRegressionTest(args);
                    break;
                default:
                    System.err.println("Error: unknown argument");
                    m.printUsage();
                    System.exit(-1);
                    break;
            }
        }
    }
}

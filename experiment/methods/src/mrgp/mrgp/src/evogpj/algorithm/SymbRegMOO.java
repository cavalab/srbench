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
package evogpj.algorithm;

import evogpj.evaluation.java.CSVDataJava;
import evogpj.evaluation.java.DataJava;
import evogpj.evaluation.FitnessFunction;
import evogpj.evaluation.java.SubtreeComplexityFitness;
import evogpj.evaluation.java.SRLARSJava;
import evogpj.genotype.Tree;
import evogpj.genotype.TreeGenerator;
import evogpj.gp.GPException;
import evogpj.gp.Individual;
import evogpj.gp.MersenneTwisterFast;
import evogpj.gp.Population;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Properties;

import evogpj.operator.Crossover;
import evogpj.operator.CrowdedTournamentSelection;
import evogpj.operator.Initialize;
import evogpj.operator.Mutate;
import evogpj.operator.Select;
import evogpj.operator.SinglePointKozaCrossover;
import evogpj.operator.SinglePointUniformCrossover;
import evogpj.operator.SubtreeMutate;
import evogpj.operator.TournamentSelection;
import evogpj.operator.TreeInitialize;
import evogpj.sort.CrowdingSort;
import evogpj.sort.DominatedCount;
import evogpj.sort.DominatedCount.DominationException;

/**
 * This class contains the main method that runs the GP algorithm.
 * 
 * @author Owen Derby and Ignacio Arnaldo
 **/
public class SymbRegMOO {
    
    /* NUMBER OF THREADS EMPLOYED IN THE EXERNAL EVALUATION */
    protected int EXTERNAL_THREADS = Parameters.Defaults.EXTERNAL_THREADS;
    
    /* DATA */
    // TRAINING SET
    protected String PROBLEM;
    // INTEGER TARGETS
    protected boolean COERCE_TO_INT = Parameters.Defaults.COERCE_TO_INT;
    protected int TARGET_NUMBER = 1;
    // FEATURES
    protected List<String> TERM_SET;    
    
    /* PARAMETERS GOVERNING THE GENETIC PROGRAMMING PROCESS */
    // POPULATION SIZE
    protected int POP_SIZE = Parameters.Defaults.POP_SIZE;
    // NUMBER OF GENERATIONS
    protected int NUM_GENS = Parameters.Defaults.NUM_GENS;
    // start time
    protected Long startTime;
    // TIME OUT
    protected Long TIMEOUT;
    
    // MUTATION RATE
    protected double MUTATION_RATE = Parameters.Defaults.MUTATION_RATE;
    // CROSSOVER RATE
    protected double XOVER_RATE = Parameters.Defaults.XOVER_RATE;
    
    
    // DEFAULT MUTATION OPERATOR
    protected String INITIALIZE = Parameters.Defaults.INITIALIZE;
    // DEFAULT MUTATION OPERATOR
    protected String SELECT = Parameters.Defaults.SELECT;
    // DEFAULT CROSSOVER OPERATOR
    protected String XOVER = Parameters.Defaults.XOVER;
    // DEFAULT MUTATION OPERATOR
    protected String MUTATE = Parameters.Defaults.MUTATE;
    // DEFAULT MUTATION OPERATOR
    protected String FITNESS = Parameters.Defaults.SRFITNESS;
    // METHOD EMPLOYED TO AGGREGATE THE FITNESS OF CANDIDATE SOLUTIONS
    protected int MEAN_POW = Parameters.Defaults.MEAN_POW;
    // METHOD EMPLOYED TO SELECT A SOLUTION FROM A PARETO FRONT
    protected String FRONT_RANK_METHOD = Parameters.Defaults.FRONT_RANK_METHOD;
    
    
    // ALL THE OPERATORS USED TO BUILD GP TREES
    protected List<String> FUNC_SET = Parameters.Defaults.FUNCTIONS;
    
    // UNARY OPERATORS USED TO BUILD GP TREES
    protected List<String> UNARY_FUNC_SET;
    
    // RANDOM SEED
    protected Long SEED = Parameters.Defaults.SEED;
    
    /* LOG FILES*/
    // LOG BEST INDIVIDUAL PER GENERATION
    
    protected String MOST_ACCURATE_PATH; //Parameters.Defaults.MOST_ACCURATE_PATH;
    
    
    
    protected String MODELS_PATH = Parameters.Defaults.MODELS_PATH;
    
   
    // LOG FINAL PARETO FRONT
    //protected String PARETO_PATH = Parameters.Defaults.PARETO_PATH;
    // LOG least complex ind
    /*
    protected String LEAST_COMPLEX_PATH = Parameters.Defaults.LEAST_COMPLEX_PATH;
    // LOG most accurate individual
    
    // LOG the model on the knee of the Pareto Front
    protected String KNEE_PATH = Parameters.Defaults.KNEE_PATH;
    // LOG the fused Pareto Front
    protected String FUSED_PATH = Parameters.Defaults.FUSED_PATH;
    */
    /* CANDIDATE SOLUTIONS MAINTAINED DURING THE SEARCH */
    // CURRENT POPULATION
    protected Population pop;
    // OFFSPRING
    protected Population childPop;
    // OFFSPRING + PARENTS
    protected Population totalPop;
    // CURRENT NON-DOMINATED SOLUTIONS
    protected Population paretoFront;
    // CURRENT GENERATION'S BEST INDIVIDUAL
    protected Individual best;
    // BEST INDIVIDUAL OF EACH GENERATION
    protected Population bestPop;
    
    /* OPERATORS EMPLOYED IN THE SEARCH PROCESS */
    // RANDOM NUMBER GENERATOR
    protected MersenneTwisterFast rand;
    // INITIALIZATION METHOD
    protected Initialize initialize;
    // CROSSOVER
    protected Crossover xover;
    // SELECTION
    protected Select select;
    // MUTATION
    protected Mutate mutate;

    // FITNESS FUNCTIONS
    protected LinkedHashMap<String, FitnessFunction> fitnessFunctions;
    
    
    /* CONTROL FOR END OF EVOLUTIONARY PROCESS*/
    // CURRENT GENERATION
    protected Integer generation;
    // CONTROL FOR END OF PROCESS
    protected Boolean finished;
    // NUMBER OF GENERATIONS WITHOUT FITNESS IMPROVEMENT
    protected int counterConvergence;
    // CURRENT FITNESS OF BEST INDIVIDUAL
    protected double lastFitness;
    
    private Properties props;
    
    double minTarget, maxTarget;
    
    /**
     * Empty constructor, to allow subclasses to override
     */
    public SymbRegMOO() {
        fitnessFunctions = new LinkedHashMap<String, FitnessFunction>();
        finished = false;
        generation = 0;
        counterConvergence = 0;
        lastFitness = 0;
        startTime = System.currentTimeMillis();
    }
    
    /**
     * Create an instance of the algorithm. This simply initializes all the
     * operators to the default parameters or whatever they are set to in the
     * passed in properties object. Use {@link #run_population()} to actually
     * run the population for the specified number of generations.
     * <p>
     * If an invalid operator type is specified, then the program will
     * terminate, indicating which parameter is incorrect.
     * 
     * @param props
     *            Properties object created from a .properties file specifying
     *            parameters for the algorithm
     * @param timeout
     * @throws java.io.IOException
     */
    public SymbRegMOO(Properties props,long timeout) throws IOException {
        this();
        if (timeout > 0)
            TIMEOUT = startTime + (timeout * 1000);
        this.props = props;
        loadParams(props);
        create_operators(props,SEED);
    }

    public SymbRegMOO(String propFile,long timeout) throws IOException {
        this();
        if (timeout > 0)
            TIMEOUT = startTime + (timeout * 1000);
        this.props = loadProps(propFile);
        loadParams(props);
        create_operators(props,SEED);
    }
    
    public SymbRegMOO(Properties aProps,String propFile,long timeout) throws IOException {
        this();
        this.props = loadProps(propFile);
        if (timeout > 0)
            TIMEOUT = startTime + (timeout * 1000);
        loadParams(props);
        Object[] presetProperties = (Object[])aProps.stringPropertyNames().toArray();
        for(int i=0;i<presetProperties.length;i++){
            String keyAux = (String)presetProperties[i];
            String valueAux = aProps.getProperty(keyAux);
            props.setProperty(keyAux, valueAux);
        }
        loadParams(props);
        create_operators(props,SEED);
    }

    /**
     * Read parameters from the Property object and set Algorithm variables.
     * 
     * @see Parameters
     */
    private void loadParams(Properties props) {
        if (props.containsKey(Parameters.Names.SEED))
            SEED = Long.valueOf(props.getProperty(Parameters.Names.SEED)).longValue();
        if (props.containsKey(Parameters.Names.PROBLEM)){
            PROBLEM = props.getProperty(Parameters.Names.PROBLEM);
            System.out.print("Problem: "+PROBLEM);
            MOST_ACCURATE_PATH = PROBLEM+"-best";
        }else if(props.containsKey("data")){
            PROBLEM = props.getProperty("data");
        }
        if (props.containsKey(Parameters.Names.EXTERNAL_THREADS))
            EXTERNAL_THREADS = Integer.valueOf(props.getProperty(Parameters.Names.EXTERNAL_THREADS));
        if (props.containsKey(Parameters.Names.FITNESS))
            FITNESS = props.getProperty(Parameters.Names.FITNESS);            
        if (props.containsKey(Parameters.Names.MUTATION_RATE))
            MUTATION_RATE = Double.valueOf(props.getProperty(Parameters.Names.MUTATION_RATE));
        if (props.containsKey(Parameters.Names.XOVER_RATE))
            XOVER_RATE = Double.valueOf(props.getProperty(Parameters.Names.XOVER_RATE));
        if (props.containsKey(Parameters.Names.POP_SIZE))
            POP_SIZE = Integer.valueOf(props.getProperty(Parameters.Names.POP_SIZE));
        if (props.containsKey(Parameters.Names.POP_DATA_PATH))
            MODELS_PATH = props.getProperty(Parameters.Names.MODELS_PATH);
        if (props.containsKey(Parameters.Names.NUM_GENS))
            NUM_GENS = Integer.valueOf(props.getProperty(Parameters.Names.NUM_GENS));
        if (props.containsKey(Parameters.Names.MEAN_POW))
            MEAN_POW = Integer.valueOf(props.getProperty(Parameters.Names.MEAN_POW));
        if (props.containsKey(Parameters.Names.COERCE_TO_INT))
            COERCE_TO_INT = Boolean.parseBoolean(props.getProperty(Parameters.Names.COERCE_TO_INT));
        if (props.containsKey(Parameters.Names.FUNCTION_SET)) {
            String funcs[] = props.getProperty(Parameters.Names.FUNCTION_SET).split(" ");
            FUNC_SET = new ArrayList<String>();
            FUNC_SET.addAll(Arrays.asList(funcs));
        }
        UNARY_FUNC_SET = new ArrayList<String>();
        for(String func:FUNC_SET){
            if(func.equals("mylog") || func.equals("exp") || func.equals("sin") || func.equals("cos") || 
                    func.equals("sqrt") || func.equals("square") || func.equals("cube") || func.equals("quart")){
                UNARY_FUNC_SET.add(func);
            }
        }
        if (props.containsKey(Parameters.Names.TERMINAL_SET)) {
            String term = props.getProperty(Parameters.Names.TERMINAL_SET);
            if (term.equalsIgnoreCase("all")) {
                // defer populating terminal list until we know our problem
                // size!
                TERM_SET = null;
            } else {
                String terms[] = term.split(" ");
                TERM_SET = new ArrayList<String>();
                TERM_SET.addAll(Arrays.asList(terms));
            }
        }
        if (props.containsKey(Parameters.Names.MUTATE))
                MUTATE = props.getProperty(Parameters.Names.MUTATE);
        if (props.containsKey(Parameters.Names.XOVER))
                XOVER = props.getProperty(Parameters.Names.XOVER);
        if (props.containsKey(Parameters.Names.SELECTION))
                SELECT = props.getProperty(Parameters.Names.SELECTION);
        if (props.containsKey(Parameters.Names.INITIALIZE))
                INITIALIZE = props.getProperty(Parameters.Names.INITIALIZE);
        if (props.containsKey(Parameters.Names.FRONT_RANK_METHOD))
                FRONT_RANK_METHOD = props.getProperty(Parameters.Names.FRONT_RANK_METHOD);
        
    }

    /**
     * Handle parsing the FITNESS field (fitness_op), which could contain
     * multiple fitness operators
     * 
     * @return a LinkedHashMap with properly ordered operators and null
     *         FitnessFunctions. This enforces the iteration order
     */
    protected LinkedHashMap<String, FitnessFunction> splitFitnessOperators(String fitnessOpsRaw) {
        LinkedHashMap<String, FitnessFunction> fitnessOperators = new LinkedHashMap<String, FitnessFunction>();
        List<String> fitnessOpsSplit = Arrays.asList(fitnessOpsRaw.split("\\s*,\\s*"));
        for (String f : fitnessOpsSplit) {
            fitnessOperators.put(f, null);
        }
        return fitnessOperators;
    }

    /**
     * Create all the operators from the loaded params. Seed is the seed to use
     * for the rng. If specified, d_in is some DataJava to use. Otherwise, d_in
     * should be null and fitness will load in the appropriate data.
     * 
     * @param seed
     * 
     */
    private void create_operators(Properties props, long seed) throws IOException {
        System.out.println("Running evogpj with seed: " + seed);
        rand = new MersenneTwisterFast(seed);
        fitnessFunctions = splitFitnessOperators(FITNESS);
        for (String fitnessOperatorName : fitnessFunctions.keySet()) {
            if (fitnessOperatorName.equals(Parameters.Operators.SR_JAVA_FITNESS)) {
                DataJava data = new CSVDataJava(PROBLEM+"-train");
                //DataJava data = new CSVDataJava(PROBLEM);
                minTarget = data.getTargetMin();
                maxTarget = data.getTargetMax();
                
                if (TERM_SET == null) {
                    TERM_SET = new ArrayList<String>();
                    for (int i = 0; i < data.getNumberOfFeatures(); i++) TERM_SET.add("X" + (i + 1));
                    System.out.println(TERM_SET);
                }
                fitnessFunctions.put(fitnessOperatorName,new SRLARSJava(data, MEAN_POW, COERCE_TO_INT,EXTERNAL_THREADS));
                //modelScalerJava = new SRModelScalerJava(data);
            } else if (fitnessOperatorName.equals(Parameters.Operators.SUBTREE_COMPLEXITY_FITNESS)) {
                fitnessFunctions.put(fitnessOperatorName,new SubtreeComplexityFitness());
            } else {
                System.err.format("Invalid fitness function %s specified for problem type %s%n",fitnessOperatorName);
                System.exit(-1);
            }
        }

        TreeGenerator treeGen = new TreeGenerator(rand, FUNC_SET, TERM_SET);
        if (INITIALIZE.equals(Parameters.Operators.TREE_INITIALIZE)) {
            initialize = new TreeInitialize(rand, props, treeGen);
        } else {
            System.err.format("Invalid initialize function %s specified%n",INITIALIZE);
            System.exit(-1);
        }

        // Set up operators.
        if (SELECT.equals(Parameters.Operators.TOURNEY_SELECT)) {
            select = new TournamentSelection(rand, props);
        } else if (SELECT.equals(Parameters.Operators.CROWD_SELECT)) {
            select = new CrowdedTournamentSelection(rand, props);
        } else {
            System.err.format("Invalid select function %s specified%n", SELECT);
            System.exit(-1);
        }

        mutate = new SubtreeMutate(rand, props, treeGen);
        //mutate = new SubtreeMutateConstants(rand, props, treeGen);

        if (XOVER.equals(Parameters.Operators.SPU_XOVER)) {
            xover = new SinglePointUniformCrossover(rand, props);
        } else if (XOVER.equals(Parameters.Operators.SPK_XOVER)) {
            xover = new SinglePointKozaCrossover(rand, props);
        } else {
            System.err.format("Invalid crossover function %s specified%n",XOVER);
            System.exit(-1);
        }

        Tree t = treeGen.generateLinearModel(TERM_SET);
        Individual linearModelInd = new Individual(t);
        // to set up equalization operator, we need to evaluate all the
        // individuals first
        pop = initialize.initialize(POP_SIZE);
        pop.set(0,linearModelInd);
        // initialize totalPop to simply the initial population
        for (FitnessFunction f : fitnessFunctions.values())
            f.evalPop(pop);
        // calculate domination counts of initial population for tournament selection
        try {
            DominatedCount.countDominated(pop, fitnessFunctions);
        } catch (DominationException e) {
            System.exit(-1);
        }
        // save first front of initial population
        paretoFront = new Population();
        for (int index = 0; index < pop.size(); index++) {
            Individual individual = pop.get(index);
            if (individual.getDominationCount().equals(0))
                paretoFront.add(individual);
        }
        // calculate crowding distances of initial population for crowding sort
        if (SELECT.equals(Parameters.Operators.CROWD_SELECT)) {
            CrowdingSort.computeCrowdingDistances(pop, fitnessFunctions);
        }
    }

    /**
     * Accept potential migrants into the population
     * @param migrants
     */
    protected void acceptMigrants(Population migrants) {
            pop.addAll(migrants);
    }
	
    /**
     * This is the heart of the algorithm. This corresponds to running the
     * {@link #pop} forward one generation
     * <p>
     * Basically while we still need to produce offspring, we select an
     * individual (or two) as parent(s) and perform a genetic operator, chosen
     * at random according to the parameters, to apply to the parent(s) to
     * produce children. Then evaluate the fitness of the new child(ren) and if
     * they are accepted by the equalizer, add them to the next generation.
     * <p>
     * The application of operators is mutually exclusive. That is, for each
     * iteration of this algorithm, we will choose exactly one of crossover,
     * mutation and replication. However, which one we choose is determined by
     * sampling from the distribution specified by the mutation and crossover
     * rates.
     * 
     * @returns a LinkedHashMap mapping fitness function name to the best
     *          individual for that fitness function
     * @throws GPException
     *             if any of the operators receive a individual with an
     *             unexpected genotype, this is an error.
     */
    protected void step() throws GPException {
        // generate children from previous population. don't use elitism
        // here since that's done later
        childPop = new Population();
        Population children;
        while (childPop.size() < POP_SIZE) {
            Individual p1 = select.select(pop);
            double prob = rand.nextDouble();
            // Select exactly one operator to use
            if (prob < XOVER_RATE) {
                Individual p2 = select.select(pop);
                children = xover.crossOver(p1, p2);
                for (Individual ind : children) {    
                    if(!ind.equals(p1) && !ind.equals(p2) && (childPop.size() < POP_SIZE)){
                        childPop.add(ind);
                    }
                }
            } else if (prob < MUTATION_RATE + XOVER_RATE) {
                Individual ind = mutate.mutate(p1);
                if(!ind.equals(p1) && (childPop.size() < POP_SIZE)){
                    childPop.add(ind);
                }
            }
        }
        // evaluate all children
        for (String fname : fitnessFunctions.keySet()) {
            FitnessFunction f = fitnessFunctions.get(fname);
            f.evalPop(childPop);
        }
        // combine the children and parents for a total of 2*POP_SIZE
        totalPop = new Population(pop, childPop);
        try {
            // for each individual, count number of individuals that dominate it
            DominatedCount.countDominated(totalPop, fitnessFunctions);
        } catch (DominationException e) {
                System.exit(-1);
        }
        // if crowding tournament selection is enabled, calculate crowding distances
        if (SELECT.equals(Parameters.Operators.CROWD_SELECT)) {
            CrowdingSort.computeCrowdingDistances(totalPop, fitnessFunctions);
        }
        // sort the entire 2*POP_SIZE population by domination count and by crowding distance if enabled
        totalPop.sort(SELECT.equals(Parameters.Operators.CROWD_SELECT));

        // use non-dominated sort to take the POP_SIZE best individuals
        // also find the latest pareto front
        pop = new Population();
        paretoFront = new Population();
        for (int index = 0; index < POP_SIZE; index++) {
            Individual individual = totalPop.get(index);
            pop.add(individual);
            // also save the first front for later use
            if (individual.getDominationCount().equals(0))
                paretoFront.add(individual);
        }
        // find best individual
        pop.calculateEuclideanDistances(fitnessFunctions);
        best = pop.get(0);
        for (int index = 0; index < POP_SIZE; index++) {
            Individual individual = pop.get(index);
            // two methods for selecting the best here from the entire population:
            // 1) euclidean distance
            // 2) "first fitness", which for dynamic equalization is simply
            // the individual with the best fitness, and for multi-objective optimization
            // is the individual with the best first fitness
            if (FRONT_RANK_METHOD.equals(Parameters.Names.EUCLIDEAN)) {
                if (individual.getEuclideanDistance() < best.getEuclideanDistance()) {
                    best = individual;
                }
            } else if (FRONT_RANK_METHOD.equals(Parameters.Names.FIRST_FITNESS)) {
                if(individual.getFitness() > best.getFitness()){
                    best = individual;
                }
            } else {
                System.err.format("No such selection method \"%s\"%n", FRONT_RANK_METHOD);
                System.exit(-1);
            }
        }
    }

    /**
     * get the best individual per generation in a Population object
     * 
     * @return the best individual per generation.
     */
    public Population getBestPop(){
        return bestPop;
    }
                
    /**
    * Run the current population for the specified number of generations.
    * 
    * @return the best individual found.
    */
    public Individual run_population() throws IOException {
        Individual bestOnCrossVal = null;
        bestPop = new Population();
        // get the best individual
        best = pop.get(0);
        System.out.println(best.getFitnesses());
        // record the best individual in models.txt
        bestPop.add(best);
        long timeStamp = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("ELAPSED TIME: " + timeStamp);
        while ((generation <= NUM_GENS) && (!finished)) {
            System.out.format("Generation %d\n", generation);
            System.out.flush();
            try {
                step();
            } catch (GPException e) {
                System.exit(-1);
            }
            // print information about this generation
            //System.out.format("Statistics: %d " + calculateStats() + "%n", generation);
            System.out.format("Best individual for generation %d:%n", generation);
            double MSE = best.getFitness();
            MSE = ((1-MSE) / (MSE + 1));
            System.out.println(best.getFitnesses());
            System.out.println(MSE);
            System.out.flush();
            bestPop.add(best);
            timeStamp = (System.currentTimeMillis() - startTime) / 1000;
            System.out.println("ELAPSED TIME: " + timeStamp);
            generation++;
            finished = stopCriteria();
            
        }

        String firstFitnessFunction = fitnessFunctions.keySet().iterator().next();
        
        
        
        if (firstFitnessFunction.equals(Parameters.Operators.SR_JAVA_FITNESS) ){
            /*
            this.saveText(MODELS_PATH, "", false);
            for(Individual ind:bestPop){
                for(int j = 0;j<ind.getWeights().size()-1;j++){
                    this.saveText(MODELS_PATH, ind.getWeights().get(j) + " ", true);
                }
                this.saveText(MODELS_PATH, ind.getWeights().get(ind.getWeights().size()-1) + ",", true);
                this.saveText(MODELS_PATH, ind.getLassoIntercept() + ",", true);
                this.saveText(MODELS_PATH, ind.toString() + "\n", true);
            }
            */
            Individual acc = paretoFront.get(0);
            //Individual comp = paretoFront.get(0);
            //Individual knee = paretoFront.get(0);
            paretoFront.calculateEuclideanDistances(fitnessFunctions);
            //this.saveText(PARETO_PATH, "", false);
            for(Individual ind:paretoFront){
                if(ind.getFitness(Parameters.Operators.SR_JAVA_FITNESS) > acc.getFitness(Parameters.Operators.SR_JAVA_FITNESS)){
                    acc = ind;
                }
                /*
                if(ind.getFitness(Parameters.Operators.SUBTREE_COMPLEXITY_FITNESS) < comp.getFitness(Parameters.Operators.SUBTREE_COMPLEXITY_FITNESS)){
                    comp = ind;
                }
                if(ind.getEuclideanDistance()<knee.getEuclideanDistance()){
                    knee = ind;
                }
                
                this.saveText(PARETO_PATH, minTarget + "," + maxTarget + ",", true);
                for(int j = 0;j<ind.getWeights().size()-1;j++){
                    this.saveText(PARETO_PATH, ind.getWeights().get(j) + " ", true);
                }
                this.saveText(PARETO_PATH, ind.getWeights().get(ind.getWeights().size()-1) + ",", true);
                this.saveText(PARETO_PATH, ind.getLassoIntercept() + ",", true);
                this.saveText(PARETO_PATH, ind.toString() + "\n", true);
                */
                
            }
            // SAVE LEAST COMPLEX MODEL OF THE PARETO FRONT
            /*
            this.saveText(LEAST_COMPLEX_PATH, "", false);
            this.saveText(LEAST_COMPLEX_PATH, minTarget + "," + maxTarget + ",", true);
            for(int j = 0;j<comp.getWeights().size()-1;j++){
                this.saveText(LEAST_COMPLEX_PATH, comp.getWeights().get(j) + " ", true);
            }
            this.saveText(LEAST_COMPLEX_PATH, comp.getWeights().get(comp.getWeights().size()-1) + ",", true);
            this.saveText(LEAST_COMPLEX_PATH, comp.getLassoIntercept() + ",", true);
            this.saveText(LEAST_COMPLEX_PATH, comp.toString() + "\n", true);
            */
            // SAVE MOST ACCURATE MODEL OF THE PARETO FRONT
            MOST_ACCURATE_PATH=PROBLEM+"-best";
            //System.out.println(MOST_ACCURATE_PATH);
            this.saveText(MOST_ACCURATE_PATH, "", false);
            this.saveText(MOST_ACCURATE_PATH, minTarget + "," + maxTarget + ",", true);
            for(int j = 0;j<acc.getWeights().size()-1;j++){
                this.saveText(MOST_ACCURATE_PATH, acc.getWeights().get(j) + " ", true);
            }
            this.saveText(MOST_ACCURATE_PATH, acc.getWeights().get(acc.getWeights().size()-1) + ",", true);
            this.saveText(MOST_ACCURATE_PATH, acc.getLassoIntercept() + ",", true);
            this.saveText(MOST_ACCURATE_PATH, acc.toString() + "\n", true);
            
            MOST_ACCURATE_PATH=PROBLEM+"-bestall";
            //System.out.println(MOST_ACCURATE_PATH);
            this.saveText(MOST_ACCURATE_PATH, "", true);
            this.saveText(MOST_ACCURATE_PATH, minTarget + "," + maxTarget + ",", true);
            for(int j = 0;j<acc.getWeights().size()-1;j++){
                this.saveText(MOST_ACCURATE_PATH, acc.getWeights().get(j) + " ", true);
            }
            this.saveText(MOST_ACCURATE_PATH, acc.getWeights().get(acc.getWeights().size()-1) + ",", true);
            this.saveText(MOST_ACCURATE_PATH, acc.getLassoIntercept() + ",", true);
            this.saveText(MOST_ACCURATE_PATH, acc.toString() + "\n", true);
            //System.out.print(MOST_ACCURATE_PATH);
            
            // SAVE KNEE MODEL OF THE PARETO FRONT
            /*
            this.saveText(KNEE_PATH, "", false);
            this.saveText(KNEE_PATH, minTarget + "," + maxTarget + ",", true);
            for(int j = 0;j<knee.getWeights().size()-1;j++){
                this.saveText(KNEE_PATH, knee.getWeights().get(j) + " ", true);
            }
            this.saveText(KNEE_PATH, knee.getWeights().get(knee.getWeights().size()-1) + ",", true);
            this.saveText(KNEE_PATH, knee.getLassoIntercept() + ",", true);
            this.saveText(KNEE_PATH, knee.toString() + "\n", true);
            */
        }  
        return bestOnCrossVal;
    }
    
    public boolean stopCriteria(){
        boolean stop = false;
        if( System.currentTimeMillis() >= TIMEOUT){
            System.out.println("Timout exceeded, exiting.");
            return true;
        }
        return stop;
    }
        
    public static Properties loadProps(String propFile) {
        Properties props = new Properties();
        BufferedReader f;
        try {
            f = new BufferedReader(new FileReader(propFile));
        } catch (FileNotFoundException e) {
            return null;
        }
        try {
            props.load(f);
        } catch (IOException e) {
        }
        System.out.println(props.toString());
        return props;
    }
       
    /**
     * Save text to a filepath
     * @param filepath
     * @param text
     * @param append
     */
    protected void saveText(String filepath, String text, Boolean append) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(filepath,append));
            PrintWriter printWriter = new PrintWriter(bw);
            printWriter.write(text);
            printWriter.flush();
            printWriter.close();
        } catch (IOException e) {
            System.exit(-1);
        }
    }
        
    public List<String> getFuncs(){
        return FUNC_SET;
    }

    public List<String> getUnaryFuncs(){
        return UNARY_FUNC_SET;
    }

    public boolean running() {
        return (generation <= NUM_GENS) && (!finished);
    }
  
}

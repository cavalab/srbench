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
package evogpj.gp;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import evogpj.evaluation.FitnessComparisonStandardizer;
import evogpj.evaluation.FitnessFunction;

/**
 * Class representing collection of all individuals in the run. Just a wrapper
 * around the ArrayList class.
 * 
 * @author Owen Derby
 */
public class Population extends ArrayList<Individual> implements Serializable {
	private static final long serialVersionUID = 6111020814262385165L;

	public Population() {
		super();
	}

	// a constructor which yields the combination of populations
	public Population(Population... populations) {
            super();
            for (Population population : populations) {
                this.addAll(population);
            }
	}

	@Override
	public boolean equals(Object other) {
		if (!this.getClass().equals(other.getClass()))
			return false;
		ArrayList<Individual> otherL = (ArrayList<Individual>) other;
		if (otherL.size() != this.size())
			return false;
		for (int i = 0; i < otherL.size(); i++) {
			if (!this.get(i).equals(otherL.get(i)))
				return false;
		}
		return true;
	}

	public Population deepCopy() {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream oos;
		try {
			oos = new ObjectOutputStream(bos);
			oos.writeObject(this);
			ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bos.toByteArray()));
			return (Population) ois.readObject();
		} catch (IOException e) {
			System.exit(-1);
		} catch (ClassNotFoundException e) {
			System.exit(-1);
		}
		return null;
	}

	/**
	 * Calculates the euclidean distance of all individuals in the first front. Others will keep a distance of Double.MAX_VALUE
	 * @param fitnessFunctions
	 */
	public void calculateEuclideanDistances(LinkedHashMap<String, FitnessFunction> fitnessFunctions) {
		// first get the mins and maxes for the first front only
		LinkedHashMap<String, Double>[] minMax = getMinMax(this, fitnessFunctions, true);
		LinkedHashMap<String, Double> mins = minMax[0];
		LinkedHashMap<String, Double> maxes = minMax[1];
		// convert the mins and maxes to standardized form
		for (String key : mins.keySet()) {
			// get new standardized min and max. swap values if this fitness function isn't minimizing already
			if (fitnessFunctions.get(key).isMaximizingFunction()) { // swap min and max since they've both been inverted
				Double standardizedMin = FitnessComparisonStandardizer.getFitnessForMinimization(maxes, key, fitnessFunctions);
				Double standardizedMax = FitnessComparisonStandardizer.getFitnessForMinimization(mins, key, fitnessFunctions);
				mins.put(key, standardizedMin);
				maxes.put(key, standardizedMax);
			} else {
				Double standardizedMin = FitnessComparisonStandardizer.getFitnessForMinimization(mins, key, fitnessFunctions);
				Double standardizedMax = FitnessComparisonStandardizer.getFitnessForMinimization(maxes, key, fitnessFunctions);
				mins.put(key, standardizedMin);
				maxes.put(key, standardizedMax);
			}
		}
		// create the ranges needed for scaling
		LinkedHashMap<String, Double> ranges = new LinkedHashMap<String, Double>();
		for (String key : mins.keySet()) {
			ranges.put(key, Math.abs(maxes.get(key) - mins.get(key)));
		}
		// compute euclidean distances for the first front only
		for (Individual individual : this) {
			if (individual.getDominationCount() > 0) continue;
			individual.calculateEuclideanDistance(fitnessFunctions, mins, ranges);
		}
	}
	
        /**
	 * Find the min and max value for each fitness function
	 * 
	 * @param pop
	 * @param onlyFirstFront if true, mins/maxes calculated only for individuals in the first front
	 */
	public static LinkedHashMap<String, Double>[] getMinMax(Population pop,LinkedHashMap<String, FitnessFunction> fitnessFunctions, Boolean onlyFirstFront) {
		LinkedHashMap<String, Double>[] minMax = new LinkedHashMap[2];
		LinkedHashMap<String, Double> mins = new LinkedHashMap<String, Double>();
		LinkedHashMap<String, Double> maxes = new LinkedHashMap<String, Double>();
		
		// establish order of fitness functions in mins and maxes
		for (String id : fitnessFunctions.keySet()) {
			mins.put(id, pop.get(0).getFitness(id));
			maxes.put(id, pop.get(0).getFitness(id));
		}

		// find mins and maxes
		for (Individual i : pop) {
			if (onlyFirstFront && i.getDominationCount() > 0) continue;
			for (String funcName : i.getFitnessNames()) {
				Double iFitness = i.getFitness(funcName);
				if (iFitness < mins.get(funcName)) // lower min
					mins.put(funcName, iFitness);
				if (iFitness > maxes.get(funcName)) // higher max
					maxes.put(funcName, iFitness);
			}
		}
		minMax[0] = mins;
		minMax[1] = maxes;
		return minMax;
	}
	
        public void sort() {
            Collections.sort(this, new DominationCrowdingSortComparator(false));
	}
                
	public class DominationCrowdingSortComparator implements Comparator<Individual> {
		// can't use crowding distance for comparison if it's not being computed a level above
		boolean crowdingDistanceEnabled;

		// default constructor
		public DominationCrowdingSortComparator() {
			this(false);
		}
		
		public DominationCrowdingSortComparator(boolean _crowdingDistanceEnabled) {
			crowdingDistanceEnabled = _crowdingDistanceEnabled;
		}
		
		@Override
		public int compare(Individual a, Individual b) {
			Integer ad = a.getDominationCount();
			Integer bd = b.getDominationCount();
			// sort by domination count: lower is better
			if (ad > bd) return 1;
			else if (ad < bd) return -1;
			// don't proceed to crowding distance sorting if we're not computing crowding distance
			if (!crowdingDistanceEnabled) return 0;
			// now sort by crowding distance: higher is better
			Double ac = a.getCrowdingDistance();
			Double bc = b.getCrowdingDistance();
			if (ac < bc) return 1;
			else if (ac > bc) return -1;
			else return 0;
		}
	}
	
	public class CrossValSortComparator implements Comparator<Individual>{
		// default constructor
		public CrossValSortComparator() {
		}
		
		//@Override
                @Override
		public int compare(Individual a, Individual b) {
			double aCVF = a.getCrossValFitness();
			double bCVF = b.getCrossValFitness();
			// sort by domination count: lower is better
                        if (aCVF == bCVF) {
                            return 0;
                        } else if (aCVF > bCVF){
                            return 1;
                        }else {
                            return -1;
                        }
		}
	}
        
	public void sort(boolean crowdingSortEnabled) {
            Collections.sort(this, new DominationCrowdingSortComparator(crowdingSortEnabled));
	}
        
        public void sortCrossVal() {
            Collections.sort(this, new CrossValSortComparator());
	}
                
}

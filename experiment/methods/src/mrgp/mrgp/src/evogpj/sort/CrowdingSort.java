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
package evogpj.sort;

import evogpj.evaluation.FitnessFunction;
import evogpj.gp.Individual;
import evogpj.gp.Population;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;

/**
 * Computes crowding distance of each individual.
 * 
 * @author Dylan Sherry and Ignacio Arnaldo
 *
 */
public class CrowdingSort {

	// arbitrary large distance assigned to boundary solutions
	public static final Double BOUNDARY_DISTANCE = 1000000000.0;

	/**
	 * Calculate cuboid region which is linearly proportional to crowding distance
	 */
	public static void computeCrowdingDistances(Population p, LinkedHashMap<String, FitnessFunction> f) {
		// for each fitness function:
		// first use comparator to generate sort indices. Pass population to
		// comparator, use it to access the population when making comparisons
		// get difference between first and last individual for normalization
		// then for each individual, update the crowding distance with the distance between neighbors, divided by max-min
		for (String fitnessFuncName : f.keySet()) {
			IntegerComparator ic = new IntegerComparator(p, fitnessFuncName);
			ArrayList<Integer> indices = getRange(0, p.size());
			Collections.sort(indices, ic);
			for (int metaIndex = 0; metaIndex < indices.size(); metaIndex++) {
				Integer previousMetaIndex = metaIndex - 1;
				Integer nextMetaIndex = metaIndex + 1;
				Double localCrowdingDistance;
				if ((previousMetaIndex < 0) || (nextMetaIndex >= p.size()))
					localCrowdingDistance = CrowdingSort.BOUNDARY_DISTANCE;
				else {
                                    Individual prev = p.get(indices.get(previousMetaIndex));
                                    Individual next = p.get(indices.get(nextMetaIndex));
                                    localCrowdingDistance = Math.abs(next.getFitness(fitnessFuncName) - prev.getFitness(fitnessFuncName));
				}
				// update the crowding distance for this individual
				Individual current = p.get(indices.get(metaIndex));
				current.updateCrowdingDistance(localCrowdingDistance);
			}
		}
	}

	public static ArrayList<Integer> getRange(Integer min, Integer max) {
            ArrayList<Integer> r = new ArrayList<Integer>();
            for (int i = min; i < max; i++)
                    r.add(i);
            return r;
	}

	public static class IntegerComparator implements Comparator<Integer> {
		Population p;
		String fitnessFuncName;

		public IntegerComparator(Population _p, String _fitnessFuncName) {
			p = _p;
			fitnessFuncName = _fitnessFuncName;
		}
		
		@Override
		public int compare(Integer index1, Integer index2) {
			// fetch corresponding individuals from Population
			Individual i1 = p.get(index1);
			Individual i2 = p.get(index2);
			Double f1 = i1.getFitness(fitnessFuncName);
			Double f2 = i2.getFitness(fitnessFuncName);
			return (f1 < f2) ? -1 : (f1 > f2) ? 1 : 0;
		}
	}
}
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
package evogpj.operator;

import evogpj.gp.Individual;
import evogpj.gp.MersenneTwisterFast;
import evogpj.gp.Population;

import java.util.Properties;

import evogpj.algorithm.Parameters;

/**
 * Perform selection by running a tournament of fitnesses between several
 * randomly chosen individuals and choosing the best one.
 * 
 * @author Owen Derby
 */
public class TournamentSelection extends RandomOperator implements Select {

    /**
     * Size of the tournament to run.
     */
    protected final int TOURNEY_SIZE;

    /**
     * Create a select operator. The tournament size is specified by the
     * {@value algorithm.Parameters.Names#TOURNEY_SIZE} key in the properties
     * object. If no value is provided, then the default of
     * {@value algorithm.Parameters.Defaults#TOURNEY_SIZE} is used.
     * 
     * @param rand random number generator
     * @param props properties for system.
     *
    public TournamentSelection(MersenneTwisterFast rand, Properties props) {
        this(rand, props, Integer.valueOf(props.getProperty(Parameters.Names.TOURNEY_SIZE)));    
    }*/

    /**
     * Create a select operator, of the specified tournament size.
     * 
     * @param rand instance of {@link MersenneTwisterFast} to use for random
     *        number generation.
     * @param tourney_size
     */
    //public TournamentSelection(MersenneTwisterFast rand, Properties props, int tourney_size) {
    public TournamentSelection(MersenneTwisterFast rand, Properties props) {
        super(rand);
        if (props.containsKey(Parameters.Names.TOURNEY_SIZE))
            TOURNEY_SIZE = Integer.valueOf(props.getProperty(Parameters.Names.TOURNEY_SIZE));
        else
            TOURNEY_SIZE = Parameters.Defaults.TOURNEY_SIZE;
    }

    @Override
    public Individual select(Population pop) {
        int n = pop.size();
        // want newPop of size n, but have to account for elitism
        Individual best, challenger;
        best = pop.get(rand.nextInt(n));
        if (best.getDominationCount() == null) {
            System.err.println("Individual has no non-domination rank");
            System.exit(-1);
        }
        for (int j = 0; j < TOURNEY_SIZE - 1; j++) {
            challenger = pop.get(rand.nextInt(n));
            if (challenger.getDominationCount() == null) {
                System.err.println("Individual has no non-domination rank");
                System.exit(-1);
            }
            if (challenger.getDominationCount() < best.getDominationCount())
                best = challenger;
            // if the individuals are in the same front, randomize selection
            else if (challenger.getDominationCount() == best.getDominationCount()) {
                if (rand.nextBoolean(0.5)) {
                        best = challenger;
                }
            }
        }
        return best;
    }

}

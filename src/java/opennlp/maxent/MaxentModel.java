///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2001 Jason Baldridge and Gann Bierner
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//////////////////////////////////////////////////////////////////////////////   
package opennlp.maxent;

/**
 * Interface for maximum entropy models.
 *
 * @author      Jason Baldridge
 * @version     $Revision: 1.1 $, $Date: 2001/10/23 14:06:53 $
 */
public interface MaxentModel {

    /**
     * Evaluates a context.
     *
     * @param context A list of String names of the contextual predicates
     *                which are to be evaluated together.
     * @return an array of the probabilities for each of the different
     *         outcomes, all of which sum to 1.
     *
     */
    public double[] eval (String[] context);

    /**
     * Simple function to return the outcome associated with the index
     * containing the highest probability in the double[].
     *
     * @param outcomes array of probabilities obtained from the eval method.
     * @return the String name of the best outcome
     *
     */
    public String getBestOutcome (double[] outcomes);

    /**
     * Gets the String name of the outcome associated with the index i.
     *
     * @param i the index for which the name of the associated outcome is
     *          desired.
     * @return the String name of the outcome
     */
    public String getOutcome (int i);

    public Object[] getDataStructures ();
    
}

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

import cern.colt.list.*;
import cern.colt.map.*;
import java.util.*;

/**
 * A maximum entropy model which has been trained using the Generalized
 * Iterative Scaling procedure (implemented in GIS.java).
 *
 * @author      Tom Morton and Jason Baldridge
 * @version     $Revision: 1.1 $, $Date: 2001/10/23 14:06:53 $
 */
public final class GISModel implements MaxentModel {
    private final OpenIntDoubleHashMap[] params;
    private final HashMap pmap = new HashMap();
    private final String[] ocNames;
    private final int correctionConstant;
    private final double correctionParam;

    private final int numOutcomes;
    private final double iprob;
    private final double fval;

    //private final double[] OUTSUMS;
    //private final int[] NUMFEATS;

    
    public GISModel (OpenIntDoubleHashMap[] _params,
		     String[] predLabels,
		     String[] _ocNames,
		     int _correctionConstant,
		     double _correctionParam) {

	for (int i=0; i<predLabels.length; i++)
	    pmap.put(predLabels[i], new Integer(i));

	params = _params;
	ocNames =  _ocNames;
	correctionConstant = _correctionConstant;
	correctionParam = _correctionParam;
	
	numOutcomes = ocNames.length;
	iprob = Math.log(1.0/numOutcomes);
	fval = 1.0/correctionConstant;
	
	//OUTSUMS= new double[numOutcomes];
	//NUMFEATS= new int[numOutcomes];
    }
    

    
    /**
     * Use this model to evaluate a context and return an array of the
     * likelihood of each outcome given that context.
     *
     * @param context The names of the predicates which have been observed at
     *                the present decision point.
     * @return        The normalized probabilities for the outcomes given the
     *                context. The indexes of the double[] are the outcome
     *                ids, and the actual string representation of the
     *                outcomes can be obtained from the method
     *  	      getOutcome(int i).
     */
    public final double[] eval(String[] context) {
	double[] outsums = new double[numOutcomes];
	int[] numfeats = new int[numOutcomes];

	for (int oid=0; oid<numOutcomes; oid++) {
	    outsums[oid] = iprob;
	    numfeats[oid] = 0;
	}

	IntArrayList activeOutcomes = new IntArrayList(0);
	for (int i=0; i<context.length; i++) {
	    if (pmap.containsKey(context[i])) {
		OpenIntDoubleHashMap predParams =
		    params[((Integer)pmap.get(context[i])).intValue()];
		predParams.keys(activeOutcomes);
		for (int j=0; j<activeOutcomes.size(); j++) {
		    int oid = activeOutcomes.getQuick(j);
		    numfeats[oid]++;
		    outsums[oid] += fval * predParams.get(oid);
		}
		//params[((Integer)pmap.get(context[i])).intValue()].forEachPair(updateSums);
	    }
	}

	double normal = 0.0;
	for (int oid=0; oid<numOutcomes; oid++) {
	    outsums[oid] = Math.exp(outsums[oid]
				    + ((1.0 -
					(numfeats[oid]/correctionConstant))
				       * correctionParam));
	    normal += outsums[oid];
	}

	for (int oid=0; oid<numOutcomes; oid++)
	    outsums[oid] /= normal;

	return outsums;
    }

    
    /**
     * Return the name of the outcome corresponding to the highest likelihood
     * in the parameter ocs.
     *
     * @param ocs A double[] as returned by the eval(String[] context)
     *            method.
     * @return    The name of the most likely outcome.
     */    
    public final String getBestOutcome(double[] ocs) {
	int best = 0;
	for (int i = 1; i<ocs.length; i++)
	    if (ocs[i] > ocs[best]) best = i;
	return ocNames[best];
    }

    
    /**
     * Return the name of an outcome corresponding to an int id.
     *
     * @param i An outcome id.
     * @return  The name of the outcome associated with that id.
     */
    public final String getOutcome(int i) {
	return ocNames[i];
    }

    
    /**
     * Provides the fundamental data structures which encode the maxent model
     * information.  This method will usually only be needed by
     * GISModelWriters.  The following values are held in the Object array
     * which is returned by this method:
     *
     * <li>index 0: cern.colt.map.OpenIntDoubleHashMap[] containing the model
     *            parameters  
     * <li>index 1: java.util.HashMap containing the mapping of model predicates
     *            to unique integers
     * <li>index 2: java.lang.String[] containing the names of the outcomes,
     *            stored in the index of the array which represents their
     * 	          unique ids in the model.
     * <li>index 3: java.lang.Integer containing the value of the models
     *            correction constant
     * <li>index 4: java.lang.Double containing the value of the models
     *            correction parameter
     *
     * @return An Object[] with the values as described above.
     */
    public final Object[] getDataStructures () {
	Object[] data = new Object[5];
	data[0] = params;
	data[1] = pmap;
	data[2] = ocNames;
	data[3] = new Integer(correctionConstant);
	data[4] = new Double(correctionParam);
	return data;
    }
    


}

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
package opennlp.maxent.io;

import opennlp.maxent.*;
import cern.colt.map.*;

/**
 * Abstract parent class for readers of GISModels.
 *
 * @author      Jason Baldridge
 * @version     $Revision: 1.1 $, $Date: 2001/10/23 14:06:53 $
 */
public abstract class GISModelReader {
    /**
     * The number of predicates contained in the model.
     */
    protected int NUM_PREDS;

    /**
     * Implement as needed for the format the model is stored in.
     */
    protected abstract int readInt () throws java.io.IOException;

    /**
     * Implement as needed for the format the model is stored in.
     */
    protected abstract double readDouble () throws java.io.IOException;

    /**
     * Implement as needed for the format the model is stored in.
     */
    protected abstract String readUTF () throws java.io.IOException;

    /**
     * Retrieve a model from disk. It assumes that models are saved in the
     * following sequence:
     * 
     * <br>GIS (model type identifier)
     * <br>1. # of parameters (int)
     * <br>2. the correction constant (int)
     * <br>3. the correction constant parameter (double)
     * <br>4. # of outcomes (int)
     * <br>  * list of outcome names (String)
     * <br>5. # of different types of outcome patterns (int)
     * <br>   * list of (int int[])
     * <br>   [# of predicates for which outcome pattern is true] [outcome pattern]
     * <br>6. # of predicates (int)
     * <br>   * list of predicate names (String)
     *
     * <p>If you are creating a reader for a format which won't work with this
     * (perhaps a database or xml file), override this method and ignore the
     * other methods provided in this abstract class.
     *
     * @return The GISModel stored in the format and location specified to
     *         this GISModelReader (usually via its the constructor).
     */
    public GISModel getModel () throws java.io.IOException {
	checkModelType();
	int correctionConstant = getCorrectionConstant();
	double correctionParam = getCorrectionParameter();
	String[] outcomeLabels = getOutcomes();
	int[][] outcomePatterns = getOutcomePatterns();
	String[] predLabels = getPredicates();
	OpenIntDoubleHashMap[] params = getParameters(outcomePatterns);
 	
	return new GISModel(params,
			    predLabels,
			    outcomeLabels,
			    correctionConstant,
			    correctionParam);

    }

    protected void checkModelType () throws java.io.IOException {
	String modelType = readUTF();
	if (!modelType.equals("GIS"))
	    System.out.println("Error: attempting to load a "+modelType+
			       " model as a GIS model."+
			       " You should expect problems.");
    }

    protected int getCorrectionConstant () throws java.io.IOException {
	return readInt();
    }
    
    protected double getCorrectionParameter () throws java.io.IOException {
	return readDouble();
    }
    
    protected String[] getOutcomes () throws java.io.IOException {
	int numOutcomes = readInt();
	String[] outcomeLabels = new String[numOutcomes];
	for (int i=0; i<numOutcomes; i++) outcomeLabels[i] = readUTF();
	return outcomeLabels;
    }

    protected int[][] getOutcomePatterns () throws java.io.IOException {
	int numOCTypes = readInt();
	int[][] outcomePatterns = new int[numOCTypes][];
	for (int i=0; i<numOCTypes; i++) {
	    String[] info = PerlHelp.split(readUTF());
	    int[] infoInts = new int[info.length];
	    for (int j=0; j<info.length; j++)
		infoInts[j] = Integer.parseInt(info[j]);
	    outcomePatterns[i] = infoInts;
	}
	return outcomePatterns;
    }

    protected String[] getPredicates () throws java.io.IOException {
	NUM_PREDS = readInt();
	String[] predLabels = new String[NUM_PREDS];
	for (int i=0; i<NUM_PREDS; i++)
	    predLabels[i] = readUTF();
	return predLabels;
    }

    protected OpenIntDoubleHashMap[] getParameters (int[][] outcomePatterns)
	throws java.io.IOException {
	
 	OpenIntDoubleHashMap[] params = new OpenIntDoubleHashMap[NUM_PREDS];

	int pid=0;
	for (int i=0; i<outcomePatterns.length; i++) {
	    for (int j=0; j<outcomePatterns[i][0]; j++) {
		params[pid] = new OpenIntDoubleHashMap();
		for (int k=1; k<outcomePatterns[i].length; k++) {
		    double d = readDouble();
		    params[pid].put(outcomePatterns[i][k], d);
		}
		params[pid].trimToSize();
		pid++;
	    }
	}
	return params;
    }

}

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

import cern.colt.function.*;
import cern.colt.list.*;
import cern.colt.map.*;

import java.io.*;
import java.util.*;
import java.util.zip.*;


/**
 * An implementation of Generalized Iterative Scaling.  The reference paper
 * for this implementation was Adwait Ratnaparkhi's tech report at the
 * University of Pennsylvania's Institute for Research in Cognitive Science,
 * and is available at <a href ="ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z"><code>ftp://ftp.cis.upenn.edu/pub/ircs/tr/97-08.ps.Z</code></a>. 
 *
 * @author  Jason Baldridge
 * @version $Revision: 1.1 $, $Date: 2001/10/23 14:06:53 $
 */
class GISTrainer {
    private boolean printMessages = false;
    private int numTokens;   // # of event tokens
    private int numPreds;    // # of predicates
    private int numOutcomes; // # of outcomes
    private int TID;         // global index variable for Tokens
    private int PID;         // global index variable for Predicates    
    private int OID;         // global index variable for Outcomes

    // a global variable for adding probabilities in an array
    private double PABISUM; 

    // records the array of predicates seen in each event
    private int[][] contexts; 

    // records the num of times an event has been seen, paired to
    // int[][] contexts
    private int[] numTimesEventsSeen;

    // stores the String names of the outcomes.  The GIS only tracks outcomes
    // as ints, and so this array is needed to save the model to disk and
    // thereby allow users to know what the outcome was in human
    // understandable terms.
    private String[] outcomeLabels;

    // stores the String names of the predicates. The GIS only tracks
    // predicates as ints, and so this array is needed to save the model to
    // disk and thereby allow users to know what the outcome was in human
    // understandable terms.
    private String[] predLabels;

    // stores the observed expections of each of the events
    private OpenIntDoubleHashMap[] observedExpects;

    // stores the estimated parameter value of each predicate during iteration
    private OpenIntDoubleHashMap[] params;

    // stores the modifiers of the parameter values, paired to params
    private OpenIntDoubleHashMap[] modifiers;

    // a helper object for storing predicate indexes
    private IntArrayList predkeys; 

    // a boolean to track if all events have same number of active features
    private boolean needCorrection;
    // initialize the GIS constant
    private int constant = 1;
    // stores inverse of constant after it is determined
    private double constantInverse;
    // the correction parameter of the model
    private double correctionParam = 0.0; 
    // observed expectation of correction feature
    private double cfObservedExpect;
    // a global variable to help compute the amount to modify the correction
    // parameter
    private double CFMOD;

    // stores the value of corrections feature for each event's predicate list,
    // expanded to include all outcomes which might come from those predicates.
    private OpenIntIntHashMap[] cfvals;

    // Normalized Probabilities Of Outcomes Given Context: p(a|b_i)
    // Stores the computation of each iterations for the update to the
    // modifiers (and therefore the params)
    private OpenIntDoubleHashMap[] pabi;

    // make all values in an OpenIntDoubleHashMap return to 0.0
    private DoubleFunction backToZeros =
        new DoubleFunction() {
                public double apply(double arg) { return 0.0; }
            };

    // divide all values in the OpenIntDoubleHashMap pabi[TID] by the sum of
    // all values in the map.
    private DoubleFunction normalizePABI =
        new DoubleFunction() {
                public double apply(double arg) { return arg / PABISUM; }
            };

    // add the previous iteration's parameters to the computation of the
    // modifiers of this iteration.
    private IntDoubleProcedure addParamsToPABI =
        new IntDoubleProcedure() {
                public boolean apply(int oid, double arg) {
                    pabi[TID].put(oid, pabi[TID].get(oid) + arg);
                    return true;
                }
            };

    // add the correction parameter and exponentiate it
    private IntDoubleProcedure addCorrectionToPABIandExponentiate =
        new IntDoubleProcedure() {
                public boolean apply(int oid, double arg) {
                    if (needCorrection)
                        arg = arg + (correctionParam * cfvals[TID].get(oid));
                    arg = Math.exp(arg);
                    PABISUM += arg;
                    pabi[TID].put(oid, arg);
                    return true;
                }
            };

    // update the modifiers based on the new pabi values
    private IntDoubleProcedure updateModifiers =
        new IntDoubleProcedure() {
                public boolean apply(int oid, double arg) {
                    modifiers[PID].put(oid,
                                       arg
                                       + (pabi[TID].get(oid)
                                          * numTimesEventsSeen[TID]));
                    return true;
                }
            };

    // update the params based on the newly computed modifiers
    private IntDoubleProcedure updateParams =
        new IntDoubleProcedure() {
                public boolean apply(int oid, double arg) {
                    params[PID].put(oid,
                                    arg
                                    + (constantInverse *
                                       (observedExpects[PID].get(oid)
                                        - Math.log(modifiers[PID].get(oid)))));
                    return true;
                }
            };

    // update the correction feature modifier, which will then be used to
    // updated the correction parameter
    private IntDoubleProcedure updateCorrectionFeatureModifier =
        new IntDoubleProcedure() {
                public boolean apply(int oid, double arg) {
                    CFMOD +=  arg * cfvals[TID].get(oid) * numTimesEventsSeen[TID];
                    return true;
                }
            };

    /**
     * Creates a new <code>GISTrainer</code> instance which does
     * not print progress messages about training to STDOUT.
     *
     */
    GISTrainer() {
        super();
    }

    /**
     * Creates a new <code>GISTrainer</code> instance.
     *
     * @param printMessages sends progress messages about training to
     *                      STDOUT when true; trains silently otherwise.
     */
    GISTrainer(boolean printMessages) {
        this();
        this.printMessages = printMessages;
    }

    /**
     * Train a model using the GIS algorithm.
     *
     * @param eventStream The EventStream holding the data on which this model
     *                    will be trained.
     * @param iterations  The number of GIS iterations to perform.
     * @param cutoff      The number of times a feature must be seen in order
     *                    to be relevant for training.
     * @return The newly trained model, which can be used immediately or saved
     *         to disk using an opennlp.maxent.io.GISModelWriter object.
     */
    public GISModel trainModel(EventStream eventStream,
                               int iterations,
                               int cutoff) {

        DataIndexer di = new DataIndexer(eventStream, cutoff);
	
        /************** Incorporate all of the needed info ******************/
        display("Incorporating indexed data for training...  \n");
        contexts = di.contexts;
        numTimesEventsSeen = di.numTimesEventsSeen;
        numTokens = contexts.length;

        //printTable(contexts);

        needCorrection = false; 

        // determine the correction constant and its inverse, and check to see
        // whether we need the correction features
        constant = contexts[0].length;
        for (TID=1; TID<contexts.length; TID++) {
            if (contexts[TID].length < constant) {
                needCorrection = true;
            }
            else if (contexts[TID].length > constant) {
                needCorrection = true;
                constant = contexts[TID].length;
            }
        }

        constantInverse = 1.0/constant;
	
        outcomeLabels = di.outcomeLabels;
        numOutcomes = outcomeLabels.length;

        predLabels = di.predLabels;
        numPreds = predLabels.length;
	
        display("\tNumber of Event Tokens: " + numTokens +"\n");
        display("\t    Number of Outcomes: " + numOutcomes +"\n");
        display("\t  Number of Predicates: " + numPreds +"\n");

        // set up feature arrays
        int[][] predCount = new int[numPreds][numOutcomes];
        for (TID=0; TID<numTokens; TID++)
            for (int j=0; j<contexts[TID].length; j++)
                predCount[contexts[TID][j]][di.outcomeList[TID]]
                    += numTimesEventsSeen[TID];

        //printTable(predCount);

        di = null; // don't need it anymore

        // Get the observed expectations of the features. Strictly speakings,
        // we should divide the counts by the number of Tokens, but because of
        // the way the model's expectations are approximated in the
        // implementation, this is cancelled out when we compute the next
        // iteration of a parameter, making the extra divisions wasteful.
        params = new OpenIntDoubleHashMap[numPreds];
        modifiers = new OpenIntDoubleHashMap[numPreds];
        observedExpects = new OpenIntDoubleHashMap[numPreds];
        for (PID=0; PID<numPreds; PID++) {
            params[PID] = new OpenIntDoubleHashMap();
            modifiers[PID] = new OpenIntDoubleHashMap();
            observedExpects[PID] = new OpenIntDoubleHashMap();
            for (OID=0; OID<numOutcomes; OID++) {
                if (predCount[PID][OID] > 0) {
                    params[PID].put(OID, 0.0);
                    modifiers[PID].put(OID, 0.0);
                    observedExpects[PID].put(OID,Math.log(predCount[PID][OID]));
                }
            }
            params[PID].trimToSize();
            modifiers[PID].trimToSize();
            observedExpects[PID].trimToSize();
        }

        predCount = null; // don't need it anymore
	
        display("...done.\n");

        pabi = new OpenIntDoubleHashMap[numTokens];

        if (needCorrection) {
            // initialize both the pabi table and the cfvals matrix
            display("Computing correction feature matrix... ");
	
            cfvals = new OpenIntIntHashMap[numTokens];
            for (TID=0; TID<numTokens; TID++) {
                cfvals[TID] = new OpenIntIntHashMap();
                pabi[TID] = new OpenIntDoubleHashMap();
                for (int j=0; j<contexts[TID].length; j++) {
                    PID = contexts[TID][j];
                    predkeys = params[PID].keys();
                    for (int i=0; i<predkeys.size(); i++) {
                        OID = predkeys.get(i);
                        if (cfvals[TID].containsKey(OID)) {
                            cfvals[TID].put(OID, cfvals[TID].get(OID) + 1);
                        } else {
                            cfvals[TID].put(OID, 1);
                            pabi[TID].put(OID, 0.0);
                        }
                    }
                }
                cfvals[TID].trimToSize();
                pabi[TID].trimToSize();
            }
	
            for (TID=0; TID<numTokens; TID++) {
                predkeys = cfvals[TID].keys();
                for (int i=0; i<predkeys.size(); i++) {
                    OID = predkeys.get(i);
                    cfvals[TID].put(OID, constant - cfvals[TID].get(OID));
                }
            }

            // compute observed expectation of correction feature (E_p~ f_l)
            int cfvalSum = 0;
            for (TID=0; TID<numTokens; TID++)
                cfvalSum += (constant - contexts[TID].length)
		            * numTimesEventsSeen[TID];
	    
            cfObservedExpect = Math.log(cfvalSum);
	    
            display("done.\n");

        }
        else {
            // initialize just the pabi table
            pabi = new OpenIntDoubleHashMap[numTokens];
            for (TID=0; TID<numTokens; TID++) {
                pabi[TID] = new OpenIntDoubleHashMap();
                for (int j=0; j<contexts[TID].length; j++) {
                    PID = contexts[TID][j];
                    predkeys = params[PID].keys();
                    for (int i=0; i<predkeys.size(); i++)
                        pabi[TID].put(predkeys.get(i), 0.0);
                }
                pabi[TID].trimToSize();
            }
        }

        /***************** Find the parameters ************************/
        display("Computing model parameters...\n");
        findParameters(iterations);
	
        /*************** Create and return the model ******************/
        return new GISModel(params,
                            predLabels,
                            outcomeLabels,
                            constant,
                            correctionParam);
	
    }
    
    
    /* Estimate and return the model parameters. */
    private void findParameters(int iterations) {
        display("Performing " + iterations + " iterations.\n");
        for (int i=1; i<=iterations; i++) {
            if (i<10) display("  " + i + ":  ");
            else if (i<100) display(" " + i + ":  ");
            else display(i + ":  ");
            nextIteration();
        }

        // kill a bunch of these big objects now that we don't need them
        observedExpects = null;
        pabi = null;
        modifiers = null;
        cfvals = null;
        numTimesEventsSeen = null;
        contexts = null;
    }


    /* Compute one iteration of GIS */
    private void nextIteration() {

        // compute table probabilities of outcomes given contexts 
        CFMOD = 0.0;
        for (TID=0; TID<numTokens; TID++) {
            pabi[TID].assign(backToZeros);

            for (int j=0; j<contexts[TID].length; j++)
                params[contexts[TID][j]].forEachPair(addParamsToPABI);

            PABISUM = 0.0; // PABISUM is computed in the next line's procedure
            pabi[TID].forEachPair(addCorrectionToPABIandExponentiate);
            if (PABISUM > 0.0) pabi[TID].assign(normalizePABI);

            if (needCorrection)
                pabi[TID].forEachPair(updateCorrectionFeatureModifier);
        }
        display(".");

        // compute contribution of p(a|b_i) for each feature and the new
        // correction parameter
        for (TID=0; TID<numTokens; TID++) {
            for (int j=0; j<contexts[TID].length; j++) {
                // do not remove the next line since we need to know PID
                // globally for the updateModifiers procedure used after it
                PID = contexts[TID][j]; 
                modifiers[PID].forEachPair(updateModifiers);
            }
        }
        display(".");
	
        // compute the new parameter values
        for (PID=0; PID<numPreds; PID++) {
            params[PID].forEachPair(updateParams);
            modifiers[PID].assign(backToZeros); // re-initialize to 0.0's
        }

        if (CFMOD > 0.0) 
            correctionParam +=
                constantInverse * (cfObservedExpect - Math.log(CFMOD));

        display(".\n");
	
    }    

    private void display (String s) {
        if (printMessages) System.out.print(s);
    }
    
}

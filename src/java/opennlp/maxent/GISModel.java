///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2004 Jason Baldridge, Gann Bierner, and Tom Morton
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

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.text.DecimalFormat;


/**
 * A maximum entropy model which has been trained using the Generalized
 * Iterative Scaling procedure (implemented in GIS.java).
 *
 * @author      Tom Morton and Jason Baldridge
 * @version     $Revision: 1.16 $, $Date: 2006/11/08 22:03:43 $
 */
public final class GISModel implements MaxentModel {
    /** Maping between predicates/contexts and an integer representing them. */
    private final TObjectIndexHashMap pmap;
    /** The names of the outcomes. */
    private final String[] ocNames;
    /** The number of outcomes. */
    //private final int numOutcomes;
    private DecimalFormat df;
    private EvalParameters evalParams;

    public GISModel (Context[] _params,
                     String[] predLabels,
                     String[] _ocNames,
                     int _correctionConstant,
                     double _correctionParam) {

        pmap = new TObjectIndexHashMap(predLabels.length);
        for (int i=0; i<predLabels.length; i++)
            pmap.put(predLabels[i], i);

        ocNames =  _ocNames;
        evalParams = new EvalParameters(_params,_correctionParam,_correctionConstant,ocNames.length);
    }
        
    private static Context[] convertToContexts(TIntParamHashMap[] params) {
      Context[] contexts = new Context[params.length];
      for (int pi=0;pi<params.length;pi++) {
        int[] activeOutcomes = params[pi].keys();
        double[] activeParameters = new double[activeOutcomes.length];
        for (int oi=0;oi<activeParameters.length;oi++) {
          activeParameters[oi] = params[pi].get(activeOutcomes[oi]);
        }
        contexts[pi] = new Context(activeOutcomes,activeParameters);
      }
      return contexts;
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
      return(eval(context,new double[evalParams.numOutcomes]));
    }
    
    /**
     * Use this model to evaluate a context and return an array of the
     * likelihood of each outcome given the specified context and the specified parameters.
     * @param context The integer values of the predicates which have been observed at
     *                the present decision point.
     * @param outsums This is where the distribution is stored.
     * @param model The set of parametes used in this computation.
     * @return The normalized probabilities for the outcomes given the
     *                context. The indexes of the double[] are the outcome
     *                ids, and the actual string representation of the
     *                outcomes can be obtained from the method
     *                getOutcome(int i).
     */
    public static double[] eval(int[] context, double[] outsums, EvalParameters model) {
      int numOutcomes = model.numOutcomes;
      Context[] params = model.params;
      double constant = model.correctionConstant;
      double constantInverse = model.constantInverse;
      double correctionParam = model.correctionParam;
      double iprob = model.iprob;
      for (int oid = 0; oid < numOutcomes; oid++) {
        outsums[oid] = iprob;
        model.numfeats[oid] = 0;
      }
      int[] activeOutcomes;
      double[] activeParameters; 
      for (int i = 0; i < context.length; i++) {
        if (context[i] >= 0) {
          Context predParams = params[context[i]];
          activeOutcomes = predParams.getOutcomes();
          activeParameters = predParams.getParameters();
          for (int j = 0; j < activeOutcomes.length; j++) {
            int oid = activeOutcomes[j];
            model.numfeats[oid]++;
            outsums[oid] += constantInverse * activeParameters[j];
          }
        }
      }

      double SUM = 0.0;
      for (int oid = 0; oid < numOutcomes; oid++) {
        outsums[oid] = Math.exp(outsums[oid]+((1.0 - ((double) model.numfeats[oid] / constant)) * correctionParam));
        SUM += outsums[oid];
      }

      for (int oid = 0; oid < numOutcomes; oid++)
        outsums[oid] /= SUM;
      return outsums;
    }
    
    /**
     * Use this model to evaluate a context and return an array of the
     * likelihood of each outcome given that context.
     *
     * @param context The names of the predicates which have been observed at
     *                the present decision point.
     * @param outsums This is where the distribution is stored.
     * @return        The normalized probabilities for the outcomes given the
     *                context. The indexes of the double[] are the outcome
     *                ids, and the actual string representation of the
     *                outcomes can be obtained from the method
     *                getOutcome(int i).
     */
    public final double[] eval(String[] context, double[] outsums) {
      int[] scontexts = new int[context.length];
      for (int i=0; i<context.length; i++) {
        scontexts[i] = pmap.get(context[i]);
      }
      return GISModel.eval(scontexts,outsums,evalParams);
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
     * Return a string matching all the outcome names with all the
     * probabilities produced by the <code>eval(String[] context)</code>
     * method.
     *
     * @param ocs A <code>double[]</code> as returned by the
     *            <code>eval(String[] context)</code>
     *            method.
     * @return    String containing outcome names paired with the normalized
     *            probability (contained in the <code>double[] ocs</code>)
     *            for each one.
     */    
    public final String getAllOutcomes (double[] ocs) {
        if (ocs.length != ocNames.length) {
            return "The double array sent as a parameter to GISModel.getAllOutcomes() must not have been produced by this model.";
        }
        else {
            if (df == null) { //lazy initilazation
              df = new DecimalFormat("0.0000");
            }
            StringBuffer sb = new StringBuffer(ocs.length*2);
            sb.append(ocNames[0]).append("[").append(df.format(ocs[0])).append("]");
            for (int i = 1; i<ocs.length; i++) {
                sb.append("  ").append(ocNames[i]).append("[").append(df.format(ocs[i])).append("]");
            }
            return sb.toString();
        }
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
     * Gets the index associated with the String name of the given outcome.
     *
     * @param outcome the String name of the outcome for which the
     *          index is desired
     * @return the index if the given outcome label exists for this
     * model, -1 if it does not.
     **/
    public int getIndex (String outcome) {
        for (int i=0; i<ocNames.length; i++) {
            if (ocNames[i].equals(outcome))
                return i;
        }
        return -1;
    } 

    public int getNumOutcomes() {
      return(evalParams.numOutcomes);
    }

    
    /**
     * Provides the fundamental data structures which encode the maxent model
     * information.  This method will usually only be needed by
     * GISModelWriters.  The following values are held in the Object array
     * which is returned by this method:
     *
     * <li>index 0: gnu.trove.TIntDoubleHashMap[] containing the model
     *            parameters  
     * <li>index 1: java.util.Map containing the mapping of model predicates
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
        data[0] = evalParams.params;
        data[1] = pmap;
        data[2] = ocNames;
        data[3] = new Integer((int)evalParams.correctionConstant);
        data[4] = new Double(evalParams.correctionParam);
        return data;
    }
    
    public static void main(String[] args) throws java.io.IOException {
      if (args.length == 0) {
        System.err.println("Usage: GISModel modelname < contexts");
        System.exit(1);
      }
      GISModel m = new opennlp.maxent.io.SuffixSensitiveGISModelReader(new File(args[0])).getModel();
      BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
      DecimalFormat df = new java.text.DecimalFormat(".###");
      for (String line = in.readLine(); line != null; line = in.readLine()) {
        String[] context = line.split(" ");
        double[] dist = m.eval(context);
        for (int oi=0;oi<dist.length;oi++) {
          System.out.print("["+m.getOutcome(oi)+" "+df.format(dist[oi])+"] ");
        }
        System.out.println();
      }
    }
}

/**
 * This class encapsulates the varibales used in producing probabilities from a model 
 * and facilitaes passing these variables to the eval method.  Variables are declared
 * non-private so that they may be accessed and updated without a method call for efficiency
 * reasons.
 * @author Tom Morton
 *
 */
class EvalParameters {
  
 /** Mapping between outcomes and paramater values for each context. 
   * The integer representation of the context can be found using <code>pmap</code>.*/
  Context[] params;
  /** The number of outcomes being predicted. */
  final int numOutcomes;
  /** The maximum number of feattures fired in an event. Usually refered to a C.
   * This is used to normalize the number of features which occur in an event. */
  double correctionConstant;
  
  /**  Stores inverse of the correction constant, 1/C. */
  final double constantInverse;
  /** The correction parameter of the model. */
  double correctionParam;
  /** Log of 1/C; initial value of probabilities. */
  final double iprob;
  
  /** Stores the number of features that get fired for each outcome in an event. 
   * This is over-written for each event evaluation, but declared once for efficiency.*/
  int[] numfeats;
  
  /**
   * Creates a set of paramters which can be evaulated with the eval method.
   * @param params The parameters of the model.
   * @param correctionParam The correction paramter.
   * @param correctionConstant The correction constant.
   * @param numOutcomes The number of outcomes.
   */
  public EvalParameters(Context[] params, double correctionParam, double correctionConstant, int numOutcomes) {
    this.params = params;
    this.correctionParam = correctionParam;
    this.numOutcomes = numOutcomes;
    this.numfeats = new int[numOutcomes];
    this.correctionConstant = correctionConstant;
    this.constantInverse = 1.0 / correctionConstant;
    this.iprob = Math.log(1.0/numOutcomes);
  }
  
}

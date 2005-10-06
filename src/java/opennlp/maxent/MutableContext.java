package opennlp.maxent;

import java.util.Arrays;

/**
 * Class used to store parameters or expected values associated with this context which
 * can be updated or assigned. 
 * @author Tom Morton
 *
 */
public class MutableContext extends Context {

  /**
   * Creates a new parametes object with the specifed parameters associated with the specified
   * outcome pattern.
   * @param outcomePattern Array of outcomes for which parameters exists for this context.
   * @param parameters Paramaters for the outcomes specified.
   */
  public MutableContext(int[] outcomePattern, double[] parameters) {
    super(outcomePattern, parameters);
  }
  
  /**
   * Assigns the parameter or expected value at the specified outcomeIndex the specified value. 
   * @param outcomeIndex The index of the parameter or expected value to be updated. 
   * @param value The value to be assigned.
   */
  public void setParameter(int outcomeIndex, double value) {
    parameters[outcomeIndex]=value;
  }
  
  /**
   * Updated the parameter or expected value at the specified outcomeIndex by adding the specified value to its current value.
   * @param outcomeIndex The index of the parameter or expected value to be updated.
   * @param value The value to be added.
   */
  public void updateParameter(int outcomeIndex, double value) {
    parameters[outcomeIndex]+=value;
  }
  
  public boolean contains(int outcome) {
    return(Arrays.binarySearch(outcomes,outcome) >= 0);
  }
}

package opennlp.maxent;

/**
 * Class which associates a real valueed parameter or expected value with a particular contextual
 * predicate or feature.  This is used to store maxent model parameters as well as model and emperical
 * expected values.  
 * @author Tom Morton
 *
 */
public class Context {

  /** The real valued parameters or expected values for this context. */
  protected double[] parameters;
  /** The outcomes which occur with this context. */
  protected int[] outcomes;
  
  /**
   * Creates a new parametes object with the specifed parameters associated with the specified
   * outcome pattern.
   * @param outcomePattern Array of outcomes for which parameters exists for this context.
   * @param parameters Paramaters for the outcomes specified.
   */
  public Context(int[] outcomePattern, double[] parameters) {
    this.outcomes = outcomePattern;
    this.parameters = parameters;
  }
  
  /**
   * Returns the outcomes for which parameters exists for this context.
   * @return Array of outcomes for which parameters exists for this context.
   */
  public int[] getOutcomes() {
    return outcomes;
  }
  
  /**
   * Returns the paramaters or expected values for the outcomes which occur with this context.
   * @return Array of paramaters for the outcomes of this context.
   */
  public double[] getParameters() {
    return parameters;
  }
}

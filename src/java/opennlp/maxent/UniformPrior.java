package opennlp.maxent;

/**
 * Provide a maximum entropy model with a uniform prior.
 * @author Tom Morton
 *
 */
public class UniformPrior implements Prior {

  private int numOutcomes;
  private double r;
  
  /**
   * Creates a uniform prior of 1/N for each outcome where N is the specified number
   * of outcomes.
   * @param numOutcomes
   */
  public UniformPrior(int numOutcomes) {
    this.numOutcomes = numOutcomes;
    r = Math.log(1.0/numOutcomes);
  }
  
  public void logPrior(double[] dist, int[] context) {
    for (int oi=0;oi<numOutcomes;oi++) {
      dist[oi] = r;
    }
  }

}

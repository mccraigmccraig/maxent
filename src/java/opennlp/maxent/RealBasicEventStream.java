package opennlp.maxent;

public class RealBasicEventStream implements EventStream {
  ContextGenerator cg = new BasicContextGenerator();
  DataStream ds;
  Event next;
  
  public RealBasicEventStream(DataStream ds) {
    this.ds = ds;
    if (this.ds.hasNext())
      next = createEvent((String)this.ds.nextToken());
    
  }

  public Event nextEvent() {
    while (next == null && this.ds.hasNext())
      next = createEvent((String)this.ds.nextToken());
    
    Event current = next;
    if (this.ds.hasNext()) {
      next = createEvent((String)this.ds.nextToken());
    }
    else {
      next = null;
    }
    return current;
  }

  public boolean hasNext() {
    while (next == null && ds.hasNext())
      next = createEvent((String)ds.nextToken());
    return next != null;
  }
  
  private Event createEvent(String obs) {
    int lastSpace = obs.lastIndexOf(' ');
    if (lastSpace == -1) 
      return null;
    else {
      String[] contexts = obs.substring(lastSpace+1).split("\\s+");
      float[] values = new float[contexts.length];
      boolean hasRealValue = false;
      for (int ci=0;ci<contexts.length;ci++) {
        int ei = contexts[ci].lastIndexOf("=");
        if (ei > 0 && ei+1 < contexts[ci].length()) {
          values[ci] = Float.parseFloat(contexts[ci].substring(ei+1));
          if (values[ci] < 0) {
            // TODO: Throw corrpurt data exception
            return null;
          }
          contexts[ci] = contexts[ci].substring(0,ei);
          hasRealValue = true;
        }
        else {
          values[ci] = 1;
        }
      }
      if (!hasRealValue) {
        values = null;
      }
      return new Event(obs.substring(lastSpace+1),contexts,values);
    }
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub

  }

}

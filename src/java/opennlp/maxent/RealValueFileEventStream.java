package opennlp.maxent;

import java.io.File;
import java.io.IOException;
import java.util.StringTokenizer;

import opennlp.maxent.io.SuffixSensitiveGISModelWriter;

public class RealValueFileEventStream extends FileEventStream {

  public RealValueFileEventStream(String fileName) throws IOException {
    super(fileName);
  }
  
  public RealValueFileEventStream(File file) throws IOException {
    super(file);
  }
  
  public Event nextEvent() {
    StringTokenizer st = new StringTokenizer(line);
    String outcome = st.nextToken();
    int count = st.countTokens();
    String[] contexts = new String[count];
    float[] values = new float[count];
    boolean hasRealValue = false;
    for (int ci = 0; ci < count; ci++) {
      contexts[ci] = st.nextToken();
      int ei = contexts[ci].lastIndexOf("=");
      if (ei > 0 && ei+1 < contexts[ci].length()) {
        boolean gotReal = true;
        try {
          values[ci] = Float.parseFloat(contexts[ci].substring(ei+1));
        }
        catch (NumberFormatException e) {
          gotReal = false;
          System.err.println("Unable to determine value in context:"+contexts[ci]);
          values[ci] = 1;
        }
        if (gotReal) {
          if (values[ci] < 0) {
            throw new RuntimeException("Negitive values are not allowed: "+contexts[ci]);
          }
          contexts[ci] = contexts[ci].substring(0,ei);
          hasRealValue = true;
        }
      }
      else {
        values[ci] = 1;
      }
    }
    if (!hasRealValue) {
      values = null;
    }
    return (new Event(outcome, contexts, values));
  }  
  
  /**
   * Trains and writes a model based on the events in the specified event file.
   * the name of the model created is based on the event file name.
   * @param args eventfile [iterations cuttoff]
   * @throws IOException when the eventfile can not be read or the model file can not be written.
   */
  public static void main(String[] args) throws IOException {
    if (args.length == 0) {
      System.err.println("Usage: RealValueFileEventStream eventfile [iterations cutoff]");
      System.exit(1);
    }
    int ai=0;
    String eventFile = args[ai++];
    EventStream es = new RealValueFileEventStream(eventFile);
    int iterations = 100;
    int cutoff = 5;
    if (ai < args.length) {
      iterations = Integer.parseInt(args[ai++]);
      cutoff = Integer.parseInt(args[ai++]);
    }
    GISModel model = GIS.trainModel(iterations,new OnePassRealValueDataIndexer(es,cutoff));
    new SuffixSensitiveGISModelWriter(model, new File(eventFile+".bin.gz")).persist();
  }
}

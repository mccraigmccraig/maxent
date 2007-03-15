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
    String[] context = new String[count];
    float[] values = new float[count];
    boolean hasRealValue = false;
    for (int ci = 0; ci < count; ci++) {
      context[ci] = st.nextToken();
      int ei = context[ci].lastIndexOf("=");
      if (ei > 0 && ei+1 < context[ci].length()) {
        values[ci] = Float.parseFloat(context[ci].substring(ei+1));
        hasRealValue = true;
      }
      else {
        values[ci] = 1;
      }
    }
    if (!hasRealValue) {
      values = null;
    }
    return (new Event(outcome, context, values));
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
    GISModel model = GIS.trainModel(es,iterations,cutoff);
    new SuffixSensitiveGISModelWriter(model, new File(eventFile+".bin.gz")).persist();
  }
}

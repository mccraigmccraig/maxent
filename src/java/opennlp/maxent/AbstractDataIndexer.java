/*
 * Created on Dec 12, 2003
 *
 */
package opennlp.maxent;

import gnu.trove.TObjectIntHashMap;
import gnu.trove.TObjectIntProcedure;

import java.util.Collections;
import java.util.List;

/**
 * @author Tom Morton
 *
 */
public abstract class AbstractDataIndexer implements DataIndexer {

  protected int[][] contexts;
  protected int[] outcomeList;
  protected int[] numTimesEventsSeen;
  protected String[] predLabels;
  protected String[] outcomeLabels;

  public int[][] getContexts() {
    return contexts;
  }

  public int[] getNumTimesEventsSeen() {
    return numTimesEventsSeen;
  }

  public int[] getOutcomeList() {
    return outcomeList;
  }

  public String[] getPredLabels() {
    return predLabels;
  }

  public String[] getOutcomeLabels() {
    return outcomeLabels;
  }
  
  

  /**
       * Sorts and uniques the array of comparable events.  This method
       * will alter the eventsToCompare array -- it does an in place
       * sort, followed by an in place edit to remove duplicates.
       *
       * @param eventsToCompare a <code>ComparableEvent[]</code> value
       * @since maxent 1.2.6
       */
  protected void sortAndMerge(List eventsToCompare) {
    Collections.sort(eventsToCompare);
    int numEvents = eventsToCompare.size();
    int numUniqueEvents = 1; // assertion: eventsToCompare.length >= 1

    if (numEvents <= 1) {
      return; // nothing to do; edge case (see assertion)
    }

    ComparableEvent ce = (ComparableEvent) eventsToCompare.get(0);
    for (int i = 1; i < numEvents; i++) {
      ComparableEvent ce2 = (ComparableEvent) eventsToCompare.get(i);

      if (ce.compareTo(ce2) == 0) {
        ce.seen++; // increment the seen count
        eventsToCompare.set(i, null); // kill the duplicate
      }
      else {
        ce = ce2; // a new champion emerges...
        numUniqueEvents++; // increment the # of unique events
      }
    }

    System.out.println("done. Reduced " + numEvents + " events to " + numUniqueEvents + ".");

    contexts = new int[numUniqueEvents][];
    outcomeList = new int[numUniqueEvents];
    numTimesEventsSeen = new int[numUniqueEvents];

    for (int i = 0, j = 0; i < numEvents; i++) {
      ComparableEvent evt = (ComparableEvent) eventsToCompare.get(i);
      if (null == evt) {
        continue; // this was a dupe, skip over it.
      }
      numTimesEventsSeen[j] = evt.seen;
      outcomeList[j] = evt.outcome;
      contexts[j] = evt.predIndexes;
      ++j;
    }
  }

  /**
       * Utility method for creating a String[] array from a map whose
       * keys are labels (Strings) to be stored in the array and whose
       * values are the indices (Integers) at which the corresponding
       * labels should be inserted.
       *
       * @param labelToIndexMap a <code>TObjectIntHashMap</code> value
       * @return a <code>String[]</code> value
       * @since maxent 1.2.6
       */
  protected static String[] toIndexedStringArray(TObjectIntHashMap labelToIndexMap) {
      final String[] array = new String[labelToIndexMap.size()];
      labelToIndexMap.forEachEntry(new TObjectIntProcedure() {
              public boolean execute(Object str, int index) {
                  array[index] = (String)str;
                  return true;
              }
          });
      return array;
  }
}

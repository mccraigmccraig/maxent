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

import gnu.trove.*;
import java.util.*;

/**
 * An indexer for maxent model data which handles cutoffs for uncommon
 * contextual predicates and provides a unique integer index for each of the
 * predicates.  The data structures built in the constructor of this class are
 * used by the GIS trainer.
 *
 * @author      Jason Baldridge
 * @version $Revision: 1.7 $, $Date: 2002/01/03 14:34:29 $
 */
public class DataIndexer {
    public int[][] contexts;
    public int[] outcomeList;
    public int[] numTimesEventsSeen;
    public String[] predLabels;
    public String[] outcomeLabels;

    /**
     * One argument constructor for DataIndexer which calls the two argument
     * constructor assuming no cutoff.
     *
     * @param events An Event[] which contains the a list of all the Events
     *               seen in the training data.
     */     
    public DataIndexer(EventStream eventStream) {
        this(eventStream, 0);
    }

    /**
     * Two argument constructor for DataIndexer.
     *
     * @param events An Event[] which contains the a list of all the Events
     *               seen in the training data.
     * @param cutoff The minimum number of times a predicate must have been
     *               observed in order to be included in the model.
     */
    public DataIndexer(EventStream eventStream, int cutoff) {
        TObjectIntHashMap predicateIndex;
        TLinkedList events;
        ComparableEvent[] eventsToCompare;

        predicateIndex = new TObjectIntHashMap();
        System.out.println("Indexing events using cutoff of " + cutoff + "\n");

        System.out.print("\tComputing event counts...  ");
        events = computeEventCounts(eventStream,predicateIndex,cutoff);
        System.out.println("done.");

        System.out.print("\tIndexing...  ");
        eventsToCompare = index(events,predicateIndex);
        // done with event list
        events = null;
        // done with predicates
        predicateIndex = null;

        System.out.println("done.");

        System.out.print("Sorting and merging events... ");
        sortAndMerge(eventsToCompare);
        System.out.println("Done indexing.");
    }

    /**
     * Sorts and uniques the array of comparable events.  This method
     * will alter the eventsToCompare array -- it does an in place
     * sort, followed by an in place edit to remove duplicates.
     *
     * @param eventsToCompare a <code>ComparableEvent[]</code> value
     * @since maxent 1.2.6
     */
    private void sortAndMerge(ComparableEvent[] eventsToCompare) {
        Arrays.sort(eventsToCompare);
        int numEvents = eventsToCompare.length;
        int numUniqueEvents = 1; // assertion: eventsToCompare.length >= 1

        if (eventsToCompare.length <= 1) {
            return;             // nothing to do; edge case (see assertion)
        }

        ComparableEvent ce = eventsToCompare[0];
        for (int i=1; i<numEvents; i++) {
            if (ce.compareTo(eventsToCompare[i]) == 0) {
                ce.seen++;      // increment the seen count
                eventsToCompare[i] = null; // kill the duplicate
            } else {
                ce = eventsToCompare[i]; // a new champion emerges...
                numUniqueEvents++; // increment the # of unique events
            }
        }

        System.out.println("done. Reduced " + eventsToCompare.length
                           + " events to " + numUniqueEvents + ".");

        contexts = new int[numUniqueEvents][];
        outcomeList = new int[numUniqueEvents];
        numTimesEventsSeen = new int[numUniqueEvents];

        for (int i = 0, j = 0; i<numEvents; i++) {
            ComparableEvent evt = eventsToCompare[i];
            if (null == evt) {
                continue;       // this was a dupe, skip over it.
            }
            numTimesEventsSeen[j] = evt.seen;
            outcomeList[j] = evt.outcome;
            contexts[j] = evt.predIndexes;
            ++j;
        }
    }

    
    /**
     * Reads events from <tt>eventStream</tt> into a linked list.  The
     * predicates associated with each event are counted and any which
     * occur at least <tt>cutoff</tt> times are added to the
     * <tt>predicatesInOut</tt> map along with a unique integer index.
     *
     * @param eventStream an <code>EventStream</code> value
     * @param predicatesInOut a <code>TObjectIntHashMap</code> value
     * @param cutoff an <code>int</code> value
     * @return a <code>TLinkedList</code> value
     */
    private TLinkedList computeEventCounts(EventStream eventStream,
                                           TObjectIntHashMap predicatesInOut,
                                           int cutoff) {
        TObjectIntHashMap counter = new TObjectIntHashMap();
        TLinkedList events = new TLinkedList();
        int predicateIndex = 0;

        while (eventStream.hasNext()) {
            Event ev = eventStream.nextEvent();
            events.addLast(ev);
            String[] ec = ev.getContext();
            for (int j=0; j<ec.length; j++) {
                if (! predicatesInOut.containsKey(ec[j])) {
                    int count = counter.get(ec[j]) + 1;
                    if (count >= cutoff) {
                        predicatesInOut.put(ec[j], predicateIndex++);
                        counter.remove(ec[j]);
                    } else {
                        counter.put(ec[j], count);
                    }
                }
            }
        }
        predicatesInOut.trimToSize();
        return events;
    }

    private ComparableEvent[] index(TLinkedList events,
                                    TObjectIntHashMap predicateIndex) {
        TObjectIntHashMap omap = new TObjectIntHashMap();

        int numEvents = events.size();
        int outcomeCount = 0;
        int predCount = 0;
        ComparableEvent[] eventsToCompare = new ComparableEvent[numEvents];
        TIntArrayList indexedContext = new TIntArrayList();

        for (int eventIndex=0; eventIndex<numEvents; eventIndex++) {
            Event ev = (Event)events.removeFirst();
            String[] econtext = ev.getContext();
	    
            int predID, ocID;
            String oc = ev.getOutcome();
	    
            if (omap.containsKey(oc)) {
                ocID = omap.get(oc);
            } else {
                ocID = outcomeCount++;
                omap.put(oc, ocID);
            }

            for (int i=0; i<econtext.length; i++) {
                String pred = econtext[i];
                if (predicateIndex.containsKey(pred)) {
                    indexedContext.add(predicateIndex.get(pred));
                }
            }

            // drop events with no active features
            if (indexedContext.size() > 0) {
                eventsToCompare[eventIndex] =
                    new ComparableEvent(ocID, indexedContext.toNativeArray());
            }
            // recycle the TIntArrayList
            indexedContext.resetQuick();
        }
        outcomeLabels = toIndexedStringArray(omap);
        predLabels = toIndexedStringArray(predicateIndex);
        return eventsToCompare;
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
    static String[] toIndexedStringArray(TObjectIntHashMap labelToIndexMap) {
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

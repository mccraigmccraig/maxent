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
 * @version $Revision: 1.5 $, $Date: 2001/12/27 19:20:26 $
 */
public class DataIndexer {
    public int[][] contexts;
    public int[] outcomeList;
    public int[] numTimesEventsSeen;
    public String[] predLabels;
    public String[] outcomeLabels;
    private static final IntegerPool intPool = new IntegerPool(50);

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
        Map count;
        TLinkedList events;

        System.out.println("Indexing events");

        System.out.print("\tComputing event counts...  ");
        count = new THashMap();
        events = computeEventCounts(eventStream,count);
        //for(int tid=0; tid<events.length; tid++) {
        System.out.println("done.");

        System.out.print("\tPerforming cutoff of " + cutoff + "...  ");
        applyCutoff(count, cutoff);
        System.out.println("done.");
	
        System.out.print("\tIndexing...  ");
        ComparableEvent[] eventsToCompare = index(events,count);
        // done with event list
        events = null;
        // done with predicate counts
        count = null;

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

    
    private TLinkedList computeEventCounts(EventStream eventStream,
					   Map count) {
        TLinkedList events = new TLinkedList();
        while (eventStream.hasNext()) {
            Event ev = eventStream.nextEvent();
            events.addLast(ev);
            String[] ec = ev.getContext();
            for (int j=0; j<ec.length; j++) {
                Counter counter = (Counter)count.get(ec[j]);
                if (counter!=null) {
                    counter.increment();
                } else {
                    count.put(ec[j], new Counter());
                }
            }
        }
        return events;
    }

    private void applyCutoff(Map count, int cutoff) {
        if (cutoff == 0) {
            return;             // nothing to do
        }
        
        for (Iterator cit=count.keySet().iterator(); cit.hasNext();) {
            String pred = (String)cit.next();
            if (! ((Counter)count.get(pred)).passesCutoff(cutoff)) {
                cit.remove();
            }
        }
    }

    private ComparableEvent[] index(TLinkedList events,
                                    Map count) {
        Map omap = new THashMap(), pmap = new THashMap();

        int numEvents = events.size();
        int outcomeCount = 0;
        int predCount = 0;
        ComparableEvent[] eventsToCompare = new ComparableEvent[numEvents];

        for (int eventIndex=0; eventIndex<numEvents; eventIndex++) {
            Event ev = (Event)events.removeFirst();
            String[] econtext = ev.getContext();
	    
            Integer predID, ocID;
            String oc = ev.getOutcome();
	    
            if (omap.containsKey(oc)) {
                ocID = (Integer)omap.get(oc);
            } else {
                ocID = intPool.get(outcomeCount++);
                omap.put(oc, ocID);
            }

            List indexedContext = new ArrayList();
            for (int i=0; i<econtext.length; i++) {
                String pred = econtext[i];
                if (count.containsKey(pred)) {
                    if (pmap.containsKey(pred)) {
                        predID = (Integer)pmap.get(pred);
                    } else {
                        predID = intPool.get(predCount++);
                        pmap.put(pred, predID);
                    }
                    indexedContext.add(predID);
                }
            }
            eventsToCompare[eventIndex] =
                new ComparableEvent(ocID.intValue(),
                                    toIntArray(indexedContext));
        }
        outcomeLabels = toIndexedStringArray(omap);
        predLabels = toIndexedStringArray(pmap);
        return eventsToCompare;
    }

    /**
     * Utility method for creating a String[] array from a map whose
     * keys are labels (Strings) to be stored in the array and whose
     * values are the indices (Integers) at which the corresponding
     * labels should be inserted.
     *
     * @param labelToIndexMap a <code>Map</code> value
     * @return a <code>String[]</code> value
     * @since maxent 1.2.6
     */
    static String[] toIndexedStringArray(Map labelToIndexMap) {
        String[] array = new String[labelToIndexMap.size()];
        for (Iterator i = labelToIndexMap.keySet().iterator(); i.hasNext();) {
            String label = (String)i.next();
            int index = ((Integer)labelToIndexMap.get(label)).intValue();
            array[index] = label;
        }
        return array;
    }

    /**
     * Utility method for turning a list of Integer objects into a
     * native array of primitive ints.
     *
     * @param integers a <code>List</code> value
     * @return an <code>int[]</code> value
     * @since maxent 1.2.6
     */
    static final int[] toIntArray(List integers) {
        int[] rv = new int[integers.size()];
        int i = 0;
        for (Iterator it = integers.iterator(); it.hasNext();) {
            rv[i++] = ((Integer)it.next()).intValue();
        }
        return rv;
    }
}

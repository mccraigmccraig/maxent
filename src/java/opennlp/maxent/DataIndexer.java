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

import java.util.*;


/**
 * An indexer for maxent model data which handles cutoffs for uncommon
 * contextual predicates and provides a unique integer index for each of the
 * predicates.  The data structures built in the constructor of this class are
 * used by the GIS trainer.
 *
 * @author      Jason Baldridge
 * @version $Revision: 1.1 $, $Date: 2001/10/23 14:06:53 $
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
	System.out.println("Indexing events");

	System.out.print("\tComputing event counts...  ");
	LinkedList events = new LinkedList();
	HashMap count = new HashMap();
	//for(int tid=0; tid<events.length; tid++) {
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
	System.out.println("done.");

	System.out.print("\tPerforming cutoff of " + cutoff + "...  ");
	HashSet useablePreds = new HashSet();
	String pred;
	for (Iterator cit=count.keySet().iterator(); cit.hasNext();) {
	    pred = (String)cit.next();
	    if (((Counter)count.get(pred)).passesCutoff(cutoff))
		useablePreds.add(pred);
	}
	System.out.println("done.");
	
	System.out.print("\tIndexing...  ");
	HashMap omap = new HashMap();
	HashMap pmap = new HashMap();

	int numEvents = events.size();
	int[] uncompressedOutcomeList = new int[numEvents];   
	LinkedList uncompressedContexts = new LinkedList();

	int outcomeCount = 0;
	int predCount = 0;
	for (int eventIndex=0; eventIndex<numEvents; eventIndex++) {
	    Event ev = (Event)events.removeFirst();
	    String[] econtext = ev.getContext();
	    
	    Integer predID, ocID;
	    String oc = ev.getOutcome();
	    
	    if (omap.containsKey(oc)) {
		ocID = (Integer)omap.get(oc);
	    } else {
		ocID = new Integer(outcomeCount++);
		omap.put(oc, ocID);
	    }

	    LinkedList indexedContext = new LinkedList();
	    for (int i=0; i<econtext.length; i++) {
		pred = econtext[i];
		if (useablePreds.contains(pred)) {
		    if (pmap.containsKey(pred)) {
			predID = (Integer)pmap.get(pred);
		    } else {
			predID = new Integer(predCount++);
			pmap.put(pred, predID);
		    }
		    indexedContext.add(predID);
		}
	    }
	    uncompressedContexts.addLast(indexedContext);
	    uncompressedOutcomeList[eventIndex] = ocID.intValue();
	}

	useablePreds = null;
	
	outcomeLabels = new String[omap.size()];
	for (Iterator i=omap.keySet().iterator(); i.hasNext();) {
	    String oc = (String)i.next();
	    outcomeLabels[((Integer)omap.get(oc)).intValue()] = oc;
	}
	omap = null;
	
	predLabels = new String[pmap.size()];
	for (Iterator i = pmap.keySet().iterator(); i.hasNext();) {
	    String n = (String)i.next();
	    predLabels[((Integer)pmap.get(n)).intValue()] = n;
	}
	pmap = null;
	
	ComparableEvent[] eventsToCompare = new ComparableEvent[numEvents];

	for (int i=0; i<numEvents; i++) {
	    LinkedList ecLL = (LinkedList)uncompressedContexts.removeFirst();
	    int[] ecInts = new int[ecLL.size()];
	    for (int j=0; j<ecInts.length; j++)
		ecInts[j] = ((Integer)ecLL.removeFirst()).intValue();
	    eventsToCompare[i] =
		new ComparableEvent(uncompressedOutcomeList[i], ecInts);
	}
	uncompressedContexts = null;
	uncompressedOutcomeList = null;
	
	System.out.println("done.");

	System.out.print("Sorting and merging events... ");
	Arrays.sort(eventsToCompare);

	ComparableEvent ce = eventsToCompare[0];
	LinkedList uniqueEvents = new LinkedList();
	LinkedList newGroup = new LinkedList();
	for (int i=0; i<numEvents; i++) {
	    if (ce.compareTo(eventsToCompare[i]) == 0) {
		  newGroup.addLast(eventsToCompare[i]);
	    } else {	    
		  ce = eventsToCompare[i];
		  uniqueEvents.addLast(newGroup);
		  newGroup = new LinkedList();
		  newGroup.addLast(eventsToCompare[i]);
	    }
	}
	uniqueEvents.addLast(newGroup);

	int numUniqueEvents = uniqueEvents.size();

	System.out.println("done. Reduced " + eventsToCompare.length
			   + " events to " + numUniqueEvents + ".");

	contexts = new int[numUniqueEvents][];
	outcomeList = new int[numUniqueEvents];
	numTimesEventsSeen = new int[numUniqueEvents];

	for (int i=0; i<numUniqueEvents; i++) {
	    LinkedList group = (LinkedList)uniqueEvents.removeFirst();
	    numTimesEventsSeen[i] = group.size();
	    ComparableEvent nextCE = (ComparableEvent)group.getFirst();
	    outcomeList[i] = nextCE.outcome;
	    contexts[i] = nextCE.predIndexes;
	}
	
	System.out.println("Done indexing.");
    }
    
}

///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2001 Chieu Hai Leong and Jason Baldridge
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//////////////////////////////////////////////////////////////////////////////   

import opennlp.maxent.*;
import java.io.*;
import java.util.*;

/**
 * Implements opennlp.maxent.EventCollector for the Weather sample.
 * Assumes a comma separated list containing all the features, 
 * with the last item being the outcome, e.g.:
 * <p>
 * feature_1, feature_2, ... feature_n, outcome
 *
 * @author  Chieu Hai Leong and Jason Baldridge
 * @version $Revision: 1.1 $, $Date: 2001/11/15 13:03:41 $
 */
public class MyEventCollector implements EventCollector {

    private ContextGenerator _cg = new MyContextGenerator();
    private List _eventList;

    public MyEventCollector (Reader datafr) {
	_eventList = new ArrayList();
	try {
	    BufferedReader br = new BufferedReader( datafr );
	    String s = br.readLine();
	    
	    while (s != null) {
		int lastComma = s.lastIndexOf(',');
		String oc = s.substring(lastComma+1);
		String[] context =
		    _cg.getContext(s.substring(0, lastComma));
		_eventList.add(new Event(oc, context));
		s = br.readLine();
	    }

	    br.close();

	} catch( Exception e ) {
	    e.printStackTrace();
	}

    }

    public Event[] getEvents () {
	return getEvents(false);
    }

    public Event[] getEvents (boolean evalMode) {
	Event[] events = new Event[_eventList.size()];
	_eventList.toArray(events);
	return events ;
    }

}

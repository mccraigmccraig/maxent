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

import opennlp.maxent.io.*;
import java.io.*;
import java.util.*;

/**
 * Trains or evaluates maxent components which have implemented the Evalable
 * interface.
 *
 * @author      Gann Bierner
 * @version     $Revision: 1.2 $, $Date: 2001/11/14 17:39:56 $
 */
public class TrainEval {
    
    public static void eval(MaxentModel model, Reader r, Evalable e) {
	eval(model, r, e, false);
    }

    public static void eval(MaxentModel model, Reader r,
			    Evalable e, boolean verbose) {

	float totPos=0, truePos=0, falsePos=0;
	Event[] events = (e.getEventCollector(r)).getEvents(true);
	//MaxentModel model = e.getModel(dir, name);
	String negOutcome = e.getNegativeOutcome();
	for(int i=0; i<events.length; i++) {
	    String guess =
		model.getBestOutcome(model.eval(events[i].getContext()));
	    String ans = events[i].getOutcome();
	    if(verbose)
		System.out.println(ans + " " + guess);
	    if(!ans.equals(negOutcome)) totPos++;
	    if(!guess.equals(negOutcome) && !guess.equals(ans))
		falsePos++;
	    else if(ans.equals(guess))
		truePos++;
	}
	
	System.out.println("Precision: " + truePos/(truePos+falsePos));
	System.out.println("Recall:    " + truePos/totPos);
	
    }

    public static MaxentModel train(EventStream events, int cutoff) {
	return GIS.trainModel(events, 100, cutoff);
    }

    public static void run(String[] args, Evalable e) throws IOException {
	String dir = "./";
	String stem = "maxent";
	int cutoff = 0; // default to no cutoff
	boolean train = false;
	boolean verbose = false;
	boolean local = false;
	gnu.getopt.Getopt g =
	    new gnu.getopt.Getopt("maxent", args, "d:s:c:tvl");
	int c;
	while ((c = g.getopt()) != -1) {
	    switch(c) {
	    case 'd':
		dir = g.getOptarg()+"/";
		break;
	    case 's':
		stem = g.getOptarg();
		break;
	    case 'c':
		cutoff = Integer.parseInt(g.getOptarg());
		break;
	    case 't':
		train = true;
		break;
	    case 'l':
		local = true;
		break;
	    case 'v':
		verbose = true;
		break;
	    }
	}
	
	FileReader datafr = new FileReader(args[g.getOptind()]);
	
	if(train) {
	    MaxentModel m =
		train(new EventCollectorAsStream(e.getEventCollector(datafr)),
		      cutoff);
	    new BinaryGISModelWriter((GISModel)m, new File(dir+stem)).persist();
	}
	else {
	    MaxentModel model =
		new BinaryGISModelReader(new File(dir+stem)).getModel();
	    if(local)
		e.localEval(model, datafr, e, verbose);
	    else
		eval(model, datafr, e, verbose);
	}
    }

}

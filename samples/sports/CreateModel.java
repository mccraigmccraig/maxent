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
import opennlp.maxent.io.*;
import java.io.*;

/**
 * Main class which calls the GIS procedure after building the EventStream
 * from the data.
 *
 * @author  Chieu Hai Leong and Jason Baldridge
 * @version $Revision: 1.1 $, $Date: 2001/11/15 13:03:41 $
 */
public class CreateModel {

    /**
     * Main method. Call as follows:
     * <p>
     * java CreateModel dataFile
     */
    public static void main (String[] args) {
	String dataFileName = new String(args[0]);
	String modelFileName =
	    dataFileName.substring(0,dataFileName.lastIndexOf('.'))
	    + "Model.txt";
	try {
	    FileReader datafr = new FileReader(new File(dataFileName));
	    EventCollector ec = new MyEventCollector(datafr);
	    EventStream es = new EventCollectorAsStream(ec);
	    GISModel model = GIS.trainModel(es);

	    File outputFile = new File(modelFileName);
	    GISModelWriter writer =
		new SuffixSensitiveGISModelWriter(model, outputFile);
	    writer.persist();
	} catch (Exception e) {
	    System.out.print("Unable to create model due to exception: ");
	    System.out.println(e);
	}
    }

}

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
package opennlp.maxent.io;

import cern.colt.map.*;
import java.io.*;
import java.util.zip.*;

/**
 * A reader for GIS models stored in the format used in v1.0 of Maxent. It
 * extends the PlainTextGISModelReader to read in the info and then overrides
 * the getParameters method so that it can appropriately read the binary file
 * which stores the parameters.
 *
 * @author      Jason Baldridge
 * @version     $Revision: 1.1 $, $Date: 2001/10/23 14:06:53 $
 */
public class OldFormatGISModelReader extends PlainTextGISModelReader {
    DataInputStream paramsInput;

    /**
     * Constructor which takes the name of the model without any suffixes,
     * such as ".mei.gz" or ".mep.gz".
     */
    public OldFormatGISModelReader(String modelname)
	throws IOException {
	super(new File(modelname+".mei.gz"));
	paramsInput = new DataInputStream(new GZIPInputStream(
		          new FileInputStream(modelname+".mep.gz")));
    }

    protected OpenIntDoubleHashMap[] getParameters (int[][] outcomePatterns)
	throws java.io.IOException {
	
	OpenIntDoubleHashMap[] params = new OpenIntDoubleHashMap[NUM_PREDS];
	  
	int pid=0;
	for (int i=0; i<outcomePatterns.length; i++) {
	    for (int j=0; j<outcomePatterns[i][0]; j++) {
		params[pid] = new OpenIntDoubleHashMap();
		for (int k=1; k<outcomePatterns[i].length; k++) {
		    double d = paramsInput.readDouble();
		    params[pid].put(outcomePatterns[i][k], d);
		}
		params[pid].trimToSize();
		pid++;
	    }
	}
	return params;
    }


    /**
     * Convert a model created with Maxent 1.0 to a format used with
     * Maxent 1.2.
     *
     * <p>Usage: java opennlp.maxent.io.OldFormatGISModelReader model_name_prefix (new_model_name)");
     *
     * <p>If the new_model_name is left unspecified, the new model will be saved
     * in gzipped, binary format as "<model_name_prefix>.bin.gz".
     */
    public static void main (String[] args) throws IOException {
	if (args.length < 1) {
	    System.out.println("Usage: java opennlp.maxent.io.OldFormatGISModelReader model_name_prefix (new_model_name)");
	    System.exit(0);
	}

	int nameIndex = 0;

	String infilePrefix = args[nameIndex];
	String outfile;

	if (args.length > nameIndex)
	    outfile = args[nameIndex+1];
	else
	    outfile = infilePrefix + ".bin.gz";


	GISModelReader reader = new OldFormatGISModelReader(infilePrefix);

	new SuffixSensitiveGISModelWriter(reader.getModel(),
					  new File(outfile)).persist();
				   
    }
    
}

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

/**
 * A Factory class which uses instances of GISTrainer to create and train
 * GISModels.
 *
 * @author  Jason Baldridge
 * @version $Revision: 1.3 $, $Date: 2001/11/16 10:37:43 $
 */
public class GIS {
    /**
     * Set this to false if you don't want messages about the progress of
     * model training displayed. Alternately, you can use the overloaded
     * version of trainModel() to conditionally enable progress messages.
     */
    public static boolean PRINT_MESSAGES = true;

    /**
     * Defines whether the created trainer will use smoothing while training
     * the model. This can improve model accuracy, though training will
     * potentially take longer and use more memory.  Model size will also be
     * larger. Set true if smoothing is desired, false if not.
     */
    public static boolean SMOOTHING = false;

    // If we are using smoothing, this is used as the "number" of
    // times we want the trainer to imagine that it saw a feature that it
    // actually didn't see.  Defaulted to 0.1.
    public static double SMOOTHING_OBSERVATION = 0.1;
    
    /**
     * Train a model using the GIS algorithm, assuming 100 iterations and no
     * cutoff.
     *
     * @param eventStream The EventStream holding the data on which this model
     *                    will be trained.
     * @return The newly trained model, which can be used immediately or saved
     *         to disk using an opennlp.maxent.io.GISModelWriter object.
     */
    public static GISModel trainModel(EventStream eventStream) {
        return trainModel(eventStream, 100, 0, PRINT_MESSAGES);
    }

    /**
     * Train a model using the GIS algorithm.
     *
     * @param eventStream The EventStream holding the data on which this model
     *                    will be trained.
     * @param iterations  The number of GIS iterations to perform.
     * @param cutoff      The number of times a feature must be seen in order
     *                    to be relevant for training.
     * @return The newly trained model, which can be used immediately or saved
     *         to disk using an opennlp.maxent.io.GISModelWriter object.
     */
    public static GISModel trainModel(EventStream eventStream,
                                      int iterations,
                                      int cutoff) {
        return trainModel(eventStream, iterations, cutoff, PRINT_MESSAGES);
    }
    
    /**
     * Train a model using the GIS algorithm.
     *
     * @param eventStream The EventStream holding the data on which this model
     *                    will be trained.
     * @param iterations  The number of GIS iterations to perform.
     * @param cutoff      The number of times a feature must be seen in order
     *                    to be relevant for training.
     * @param printMessagesWhileTraining write training status messages
     *                                   to STDOUT.
     * @return The newly trained model, which can be used immediately or saved
     *         to disk using an opennlp.maxent.io.GISModelWriter object.
     */
    public static GISModel trainModel(EventStream eventStream,
                                      int iterations,
                                      int cutoff,
                                      boolean printMessagesWhileTraining) {
        GISTrainer trainer = new GISTrainer(printMessagesWhileTraining);
	trainer.setSmoothing(SMOOTHING);
	trainer.setSmoothingObservation(SMOOTHING_OBSERVATION);
        return trainer.trainModel(eventStream, iterations, cutoff);
    }
}




package weka.classifiers.trees.occt.split.models;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;

/**
 * Class implementing a "no-weka.trees.classifiers.occt.split"-weka.trees.classifiers.occt.split.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.9 $
 */
public final class OCCTNoSplitModel extends OCCTSplitModel {

    /** for serialization */
    private static final long serialVersionUID = -1292620749331337546L;

    /**
     * Creates "no-weka.trees.classifiers.occt.split"-weka.trees.classifiers.occt.split
     */
    public OCCTNoSplitModel() {
        super();
    }

    /**
     * Creates a "no-weka.trees.classifiers.occt.split"-weka.trees.classifiers.occt.split for a given set of instances.
     *
     * @exception Exception if weka.trees.classifiers.occt.split can't be built successfully
     */
    public final void buildClassifier(Instances instances) throws Exception {
    }

    /**
     * Always returns 0 because only there is only one subset.
     */
    public final int whichSubset(Instance instance){
        return 0;
    }

    /**
     * Always returns null because there is only one subset.
     */
    public final double [] weights(Instance instance){
        return null;
    }

    /**
     * Does nothing because no condition has to be satisfied.
     */
    public final String leftSide(Instances instances){
        return "";
    }

    /**
     * Does nothing because no condition has to be satisfied.
     */
    public final String rightSide(int index, Instances instances){
        return "";
    }

    /**
     * Returns a string containing java source code equivalent to the test
     * made at this node. The instance being tested is called "i".
     *
     * @param index index of the nominal value tested
     * @param data the data containing instance structure info
     * @return a value of type 'String'
     */
    public final String sourceExpression(int index, Instances data) {
        return "true";  // or should this be false??
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.0 $");
    }
}

package weka.classifiers.trees.occt.split.models;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * Created by sepetnit on 07/02/15.
 *
 */
public abstract class OCCTSplitModel extends ClassifierSplitModel {

    private static final long serialVersionUID = -5212316095762853537L;

    public OCCTSplitModel() {
    }

    public String rightSide(String value, Instances data) {
        return "";
    }

    public Attribute getChosenAttribute() {
        return null;
    }

    public double score() {
        return 0;
    }
}

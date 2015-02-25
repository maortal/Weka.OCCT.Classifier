package weka.classifiers.trees.occt.split.pruning;

import weka.classifiers.trees.occt.split.general.OCCTSplitModelFactory;
import weka.classifiers.trees.occt.split.models.OCCTSingleAttributeSplitModel;
import weka.classifiers.trees.occt.tree.OCCTSplitModelSelection;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.Serializable;
import java.util.List;

/**
 * Created by sepetnit on 14/02/15.
 *
 */
public abstract class OCCTGeneralPruningMethod implements Serializable {

    private static final long serialVersionUID = 2223139345097265363L;

    /** All the training data */
    protected Instances m_allData;
    /** Attributes of the table B */
    protected List<Attribute> m_attributesOfB;
    /** The model selection method which is used by the pruner in order to calculate scores **/
    protected OCCTSplitModelSelection m_internalModelSelection;
    /** A general threshold for performing pruning. Not always required **/
    protected double m_pruningThreshold;

    public OCCTGeneralPruningMethod(Instances instances,
                                    List<Attribute> attributesOfB,
                                    Class<? extends OCCTSingleAttributeSplitModel> splitMethodCls,
                                    double pruningThreshold) {
        this.m_allData = instances;
        this.m_attributesOfB = attributesOfB;
        this.m_internalModelSelection = new OCCTSplitModelSelection(
                OCCTSplitModelFactory.extractModelName(splitMethodCls),
                this.m_allData,
                this.m_attributesOfB);
        this.m_pruningThreshold = pruningThreshold;
    }

    public abstract boolean shouldPrune(Instances data) throws Exception;

    /**
     * Sets reference to training data to null.
     */
    public void cleanup() {
        this.m_allData = null;
    }

}

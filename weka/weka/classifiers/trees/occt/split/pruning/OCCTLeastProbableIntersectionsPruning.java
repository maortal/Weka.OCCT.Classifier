package weka.classifiers.trees.occt.split.pruning;

import weka.classifiers.trees.occt.split.models.OCCTLeastProbableIntersectionsSplitModel;
import weka.classifiers.trees.occt.split.models.OCCTSplitModel;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.List;

/**
 * Created by sepetnit on 14/02/15.
 *
 */
public class OCCTLeastProbableIntersectionsPruning extends OCCTGeneralPruningMethod {

    private static final long serialVersionUID = 5291663316793118748L;

    public OCCTLeastProbableIntersectionsPruning(Instances instances,
                                                 List<Attribute> attributesOfB,
                                                 double pruningThreshold) {
        super(instances, attributesOfB,
                OCCTLeastProbableIntersectionsSplitModel.class, pruningThreshold);
    }

    /**
     * In the least probable intersections method, Z is calculated for each possible split.
     * If for all possible splitting attributes Z is smaller than a predefined threshold,
     * it means that all of the possible splits are likely to be formed by a random split.
     * Thus, we will not gain much information from any of the possible splits, and the branch will
     * be pruned.
     *
     * Since the chosen model for LPI has the highest score over all the models, it is sufficient
     * to check that the score of this model is lower than the threshold
     */
    @Override
    public boolean shouldPrune(Instances data) throws Exception {
        // Select the best model using the internal split model instance
        OCCTSplitModel bestModel = this.m_internalModelSelection.selectModel(data);
        // In case of no split we of course want to prune
        return bestModel.score() <= this.m_pruningThreshold;
    }
}

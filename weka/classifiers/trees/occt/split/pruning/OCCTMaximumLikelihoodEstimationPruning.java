package weka.classifiers.trees.occt.split.pruning;

import weka.classifiers.trees.occt.split.models.OCCTMaximumLikelihoodEstimationSplitModel;
import weka.classifiers.trees.occt.split.models.OCCTSplitModel;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.List;

/**
 * Created by sepetnit on 14/02/15.
 *
 */
public class OCCTMaximumLikelihoodEstimationPruning extends OCCTGeneralPruningMethod {

    private static final long serialVersionUID = 4235510347757174457L;

    private OCCTMaximumLikelihoodEstimationSplitModel m_currentMLEScoreEstimator;

    public OCCTMaximumLikelihoodEstimationPruning(Instances instances,
                                                  List<Attribute> attributesOfB,
                                                  double pruningThreshold) {
        super(instances, attributesOfB,
                OCCTMaximumLikelihoodEstimationSplitModel.class, pruningThreshold);
        // We don't need the splitting attribute (the first parameter) as well as possible
        // attributes for splitting
        this.m_currentMLEScoreEstimator =
                new OCCTMaximumLikelihoodEstimationSplitModel(null, null, this.m_attributesOfB);
    }

    /**
     * In the maximum likelihood method, an MLE score is computed for each of the possible splits.
     * If none of the candidate attributes achieve an MLE score which is higher than the current
     * node's MLE score, the branch is pruned, and the current node becomes a leaf.
     *
     * @param data The instances on which pre-pruning is applied
     *
     * @return Whether we should prune
     *
     * @throws Exception if something goes wrong
     */
    public boolean shouldPrune(Instances data) throws Exception {
        double currentScore = this.m_currentMLEScoreEstimator.calculateMLEScore(data);
        // Select the best model using the internal split model instance
        OCCTSplitModel bestModel = this.m_internalModelSelection.selectModel(data);
        // In case of no split we of course want to prune
        return bestModel.numSubsets() == 0 || bestModel.score() <= currentScore;
    }
}

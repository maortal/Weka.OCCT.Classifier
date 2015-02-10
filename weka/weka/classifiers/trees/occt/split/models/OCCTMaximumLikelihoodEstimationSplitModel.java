package weka.classifiers.trees.occt.split.models;

import weka.classifiers.trees.occt.split.auxiliary.OCCTSplitModelComparators;
import weka.classifiers.trees.occt.split.auxiliary.ProbModelsHandler;
import weka.classifiers.trees.occt.split.iterators.GeneralInstancesIterator;
import weka.classifiers.trees.occt.split.iterators.SingleInstancesPairIterator;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.List;

/**
 * Created by sepetnit on 16/01/15.
 *
 */
public class OCCTMaximumLikelihoodEstimationSplitModel extends OCCTSingleAttributeSplitModel  {
    private static final long serialVersionUID = -4241500188359190093L;

    private ProbModelsHandler probModelsBuilder;

    static {
        m_scoresComparator =
                OCCTSplitModelComparators.HIGHEST_SCORE_CHOOSER;
    }

    public OCCTMaximumLikelihoodEstimationSplitModel(Attribute splittingAttribute,
                                                     List<Attribute> possibleAttributes,
                                                     List<Attribute> attributesOfB) {
        super(splittingAttribute, possibleAttributes, attributesOfB);
        this.probModelsBuilder = new ProbModelsHandler(this.m_attributesOfB);
    }

    private double calculateLValueForInstances(Instances instances) throws Exception {
        double toReturn = 0;
        Enumeration instancesEnum = instances.enumerateInstances();
        while (instancesEnum.hasMoreElements()) {
            Instance currentInstance = (Instance)instancesEnum.nextElement();
            toReturn += this.probModelsBuilder.calculateLValueForSingleInstance(instances,
                    currentInstance);
        }
        return toReturn;
    }

    @Override
    protected double calculateSplitScore(Instances i1, Instances i2) throws Exception {
        // Let's assert that the second instance is null
        // (since we calculate score for a single instance only in MLE)
        assert (i2 == null);
        // First, build the probabilistic models
        this.probModelsBuilder.buildModels(i1);
        // Once the set of models has been induced, the probability of each record given these
        // models is calculated
        return this.calculateLValueForInstances(i1);
    }

    @Override
    protected GeneralInstancesIterator getInstancesSetsIterator(
            Instances trainInstances, Instances[] splittedTrainInstances) {
        return new SingleInstancesPairIterator(trainInstances, splittedTrainInstances);
    }
}
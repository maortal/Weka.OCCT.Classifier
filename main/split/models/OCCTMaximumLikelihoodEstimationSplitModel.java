package split.models;

import split.auxiliary.OCCTSplitModelComparators;
import split.iterators.GeneralInstancesIterator;
import split.iterators.SingleInstancesPairIterator;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by sepetnit on 16/01/15.
 *
 */
public class OCCTMaximumLikelihoodEstimationSplitModel extends OCCTSingleAttributeSplitModel {

    static {
        OCCTMaximumLikelihoodEstimationSplitModel.m_scoresComparator =
                OCCTSplitModelComparators.HIGHEST_SCORE_CHOOSER;
    }

    private int[] attributesOfBIndexes;
    // Models for each of the attributes of B, given other attributes
    private Map<Attribute, Classifier> probabilisticModels;

    public OCCTMaximumLikelihoodEstimationSplitModel(Attribute splittingAttribute,
                                                     List<Attribute> possibleAttributes,
                                                     List<Attribute> attributesOfB) {
        super(splittingAttribute, possibleAttributes, attributesOfB);
        // Find all the indexes of B's attributes (perform this a single time)
        this.attributesOfBIndexes = new int[this.m_attributesOfB.size()];
        int index = 0;
        for (Attribute attribute : this.m_attributesOfB) {
            this.attributesOfBIndexes[index++] = attribute.index();
        }
        this.probabilisticModels = new HashMap<Attribute, Classifier>(this.m_attributesOfB.size());
    }

    private Remove getRemoveFilter() {
        // A filter to remove unnecessary columns
        // A new filter is required for each attribute since the class index is specified for
        // the filter using the dataset and cannot be changed by the API of the filtered
        // classifier
        Remove rm = new Remove();
        // Only the specified attributes are kept
        rm.setInvertSelection(true);
        // Only B attributes should remain
        rm.setAttributeIndicesArray(this.attributesOfBIndexes);
        return rm;
    }

    private J48 getActualClassifier() {
        // TODO: Set minimum number of objects in a leaf? (setMinNumObj)
        J48 currentActualClassifier = new J48();
        currentActualClassifier.setUseLaplace(true);
        currentActualClassifier.setUnpruned(true);
        return currentActualClassifier;
    }

    private void buildModels(Instances instances) {
        // First, let's remove the unnecessary attributes
        this.probabilisticModels.clear();
        for (Attribute attr : this.m_attributesOfB) {
            instances.setClass(attr);
            FilteredClassifier filteredClassifier = new FilteredClassifier();
            filteredClassifier.setFilter(this.getRemoveFilter());
            filteredClassifier.setClassifier(this.getActualClassifier());
            // Train and make predictions
            try {
                // Here, the Remove filter learns the class index
                filteredClassifier.buildClassifier(instances);
            }
            catch (Exception e) {
                // TODO: What should be done in case of an error here?
                e.printStackTrace();
            }
            this.probabilisticModels.put(attr, filteredClassifier);
        }
    }

    private double calculateLValueForSingleInstance(Instances allInstances,
                                                    Instance currentInstance) throws Exception {
        double toReturn = 0;
        for (Map.Entry<Attribute, Classifier> entry : this.probabilisticModels.entrySet()) {
            Attribute currentAttribute = entry.getKey();
            Classifier currentClassifier = entry.getValue();
            allInstances.setClass(currentAttribute);
            double[] dist = currentClassifier.distributionForInstance(currentInstance);
            int valueIndex = currentAttribute.indexOfValue(
                    currentInstance.stringValue(currentAttribute));
            toReturn += Math.log10(dist[valueIndex]);
        }
        return toReturn;
    }

    private double calculateLValueForInstances(Instances instances) throws Exception {
        double toReturn = 0;
        Enumeration instancesEnum = instances.enumerateInstances();
        while (instancesEnum.hasMoreElements()) {
            Instance currentInstance = (Instance)instancesEnum.nextElement();
            toReturn += this.calculateLValueForSingleInstance(instances, currentInstance);
        }
        return toReturn;
    }

    @Override
    protected double calculateSplitScore(Instances i1, Instances i2) throws Exception {
        // Let's assert that the second instance is null
        // (since we calculate score for a single instance only in MLE)
        assert (i2 == null);
        // First, build the probabilistic models
        this.buildModels(i1);
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
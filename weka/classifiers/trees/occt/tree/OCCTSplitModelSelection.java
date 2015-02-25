package weka.classifiers.trees.occt.tree;

import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.occt.split.general.OCCTSplitModelFactory;
import weka.classifiers.trees.occt.split.models.OCCTNoSplitModel;
import weka.classifiers.trees.occt.split.models.OCCTSingleAttributeSplitModel;
import weka.classifiers.trees.occt.split.models.OCCTSplitModel;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.RevisionUtils;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * Created by sepetnit on 30/12/14.
 *
 */
public class OCCTSplitModelSelection extends ModelSelection {

    /** for serialization */
    private static final long serialVersionUID = 3372204862440822039L;

    private String m_splitCriterionType;

    /** All the training data */
    private Instances m_allData;

    /** The attributes which can be used for the split **/
    private List<Attribute> m_possibleAttributes;
    private List<Attribute> m_attributesOfB;

    // Copy constructor
    public OCCTSplitModelSelection(OCCTSplitModelSelection toCopy, Attribute exceptionAttribute) {
        this.m_splitCriterionType = toCopy.m_splitCriterionType;
        this.m_allData = toCopy.m_allData;
        this.m_possibleAttributes = new LinkedList<Attribute>(toCopy.m_possibleAttributes);
        this.m_possibleAttributes.remove(exceptionAttribute);
        this.m_attributesOfB = toCopy.m_attributesOfB;
    }

    public OCCTSplitModelSelection(OCCTSplitModelSelection toCopy) {
        this.m_splitCriterionType = toCopy.m_splitCriterionType;
        this.m_allData = toCopy.m_allData;
        this.m_possibleAttributes = new LinkedList<Attribute>(toCopy.m_possibleAttributes);
        this.m_attributesOfB = toCopy.m_attributesOfB;
    }

    /**
     * Initializes the selection method with the given parameters.
     *
     * @param allData FULL training dataset (may be relevant for selection of split points).
     */
    public OCCTSplitModelSelection(String splitCriterionType, Instances allData,
                                   List<Attribute> attributesOfB,
                                   Attribute... except)
            throws IllegalArgumentException {
        this.m_splitCriterionType = splitCriterionType;
        if (!OCCTSplitModelFactory.isValidSplitModelType(this.m_splitCriterionType)) {
            System.out.println("illegal");
            throw new IllegalArgumentException(splitCriterionType);
        }
        this.m_allData = allData;
        this.m_possibleAttributes = new LinkedList<Attribute>();
        Enumeration attributes = allData.enumerateAttributes();
        List<Attribute> exceptionAttributes = Arrays.asList(except);
        while (attributes.hasMoreElements()) {
            Attribute toAdd = (Attribute)attributes.nextElement();
            if (!exceptionAttributes.contains(toAdd)) {
                this.m_possibleAttributes.add(toAdd);
            }
        }
        this.m_attributesOfB = attributesOfB;
    }

    public OCCTSplitModelSelection(String splitCriterionType, Instances allData,
                                   Attribute... attributesOfB) {
        this(splitCriterionType, allData, Arrays.asList(attributesOfB));
    }

    /**
     * Sets reference to training data to null.
     */
    public void cleanup() {
        this.m_allData = null;
    }

    /**
     * This function chooses the best attribute to weka.trees.classifiers.occt.split on.
     *
     * @param possibleSplitModels An list of models for each possible attribute
     *
     * @return The chosen split model
     */
    private OCCTSplitModel selectBestModel(
            List<OCCTSingleAttributeSplitModel> possibleSplitModels) {
        // The score of the best model chosen so far
        Double bestScore = null;
        // Let's look for the min score
        Set<OCCTSingleAttributeSplitModel> allBestModels = new
                HashSet<OCCTSingleAttributeSplitModel>();
        // This comparator is used in order to compare between different models (built for each
        // attribute)
        Comparator<Double> splitModelComparator =
                OCCTSplitModelFactory.getSplitModelComparator(this.m_splitCriterionType);

        // Record all the scores of the best value
        for (OCCTSingleAttributeSplitModel currentModel: possibleSplitModels) {
            if (currentModel.checkModel()) {
                if ((bestScore == null) ||
                        (splitModelComparator.compare(currentModel.score(), bestScore) >= 0)) {
                    allBestModels.clear();
                    allBestModels.add(currentModel);
                    bestScore = currentModel.score();
                }
            }
        }
        // In case no possible model was found
        if (allBestModels.size() == 0) {
            return new OCCTNoSplitModel();
        }
        // Currently return the first index of the minimal value
        // TODO: Maybe think about some other way of tie breaking
        System.out.println("Best score for attribute " +
                allBestModels.iterator().next().leftSide(this.m_allData) + " " + bestScore);
        return this.updateLastRunAndReturnBestModel(allBestModels);
    }

    private OCCTSplitModel updateLastRunAndReturnBestModel(
            Set<OCCTSingleAttributeSplitModel> allBestModels) {
        return allBestModels.iterator().next();
    }

    /**
     * Selects a split for the given dataset.
     */
    public final OCCTSplitModel selectModel(Instances data) throws Exception {
        // TODO: Check if all Instances belong to one class or if not enough Instances to split
        // TODO: Check if all attributes are nominal and have a lot of values.

        // Used to split by each attribute and choose the best one ...
        List<OCCTSingleAttributeSplitModel> possibleSplitModels =
                new LinkedList<OCCTSingleAttributeSplitModel>();
        // For each attribute.
        Enumeration attributes = data.enumerateAttributes();
        while (attributes.hasMoreElements()) {
            Attribute nextAttribute = (Attribute)attributes.nextElement();
            if (this.m_possibleAttributes.contains(nextAttribute) &&
                    (!this.m_attributesOfB.contains(nextAttribute)) &&
                    (nextAttribute.index() != data.classIndex())) {
                // Get model for current attribute.
                // TODO: How to decide which type of weka.trees.classifiers.occt.split it will be???
                OCCTSingleAttributeSplitModel currentSplitModel =
                        OCCTSplitModelFactory.getSplitModel(
                                this.m_splitCriterionType,
                                nextAttribute, this.m_possibleAttributes, this.m_attributesOfB);
                currentSplitModel.buildClassifier(data);

                // TODO: Check if useful weka.trees.classifiers.occt.split for current attribute exists and check for
                // TODO: enumerated attributes with a lot of values.
                if (currentSplitModel.checkModel()) {
                    possibleSplitModels.add(currentSplitModel);
                }
            }
        }

        // Check if any useful split was found.
        if (possibleSplitModels.size() == 0) {
            return new OCCTNoSplitModel();
        }

        // TODO: Add all Instances with unknown values for the corresponding attribute to the
        // TODO: distribution for the model, so that the complete distribution is stored with
        // TODO: the model.

        return this.selectBestModel(possibleSplitModels);
    }

    /**
     * Selects the relevant weka.trees.classifiers.occt.split for the given dataset
     */
    public final ClassifierSplitModel selectModel(Instances train, Instances test) throws Exception {
        return this.selectModel(train);
    }

    public List<Attribute> getAttributesOfB() {
        return this.m_attributesOfB;
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

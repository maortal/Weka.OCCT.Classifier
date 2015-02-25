package weka.classifiers.trees.occt.split.auxiliary;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by sepetnit on 17/02/15.
 *
 */
public class OCCTFeatureSelector implements Serializable {

    private static final long serialVersionUID = -2596337180687583510L;

    private Attribute m_selectedSplittingAttribute;
    private List<Attribute> m_attributesOfB;
    private boolean m_isBuilt;
    private List<Attribute> m_selectedFeatures;

    private static int[] B_INDEXES;

    static {
        OCCTFeatureSelector.B_INDEXES = null;
    }

    public OCCTFeatureSelector(Attribute selectedSplittingAttribute,
                               List<Attribute> attributesOfB) {
        this.m_selectedSplittingAttribute = selectedSplittingAttribute;
        this.m_attributesOfB = attributesOfB;
    }

    /**
     * This constructor is used in order to immediately select the features
     * (without calling a separate function)
     *
     * @param selectedSplittingAttribute The splitting attribute by which the current dataset is
     *                                   splitted
     * @param attributesOfB The attributes of T_B
     * @param instances The leaf dataset (instances on which the feature selection process will be
     *                  ran)
     * @throws Exception In case the feature selection process failed for some reason
     */
    public OCCTFeatureSelector(Attribute selectedSplittingAttribute,
                               List<Attribute> attributesOfB,
                               Instances instances) throws Exception {
        this.m_selectedSplittingAttribute = selectedSplittingAttribute;
        this.m_attributesOfB = attributesOfB;
        this.selectFeatures(instances);
    }

    private int[] getIndexesOfAttributesOfB() {
        if (OCCTFeatureSelector.B_INDEXES == null) {
            // Find all the indexes of B's attributes (perform this a single time)
            OCCTFeatureSelector.B_INDEXES = this.getIndexesFromAttributesList(this.m_attributesOfB);
        }
        return OCCTFeatureSelector.B_INDEXES;
    }

    private int[] getIndexesFromAttributesList(List<Attribute> attributes) {
        int index = 0;
        int[] toReturn = new int[attributes.size()];
        for (Attribute attribute : attributes) {
            toReturn[index++] = attribute.index();
        }
        return toReturn;
    }
    private int[] joinArrays(int[]... arrays) {
        // calculate size of target array
        int size = 0;
        int index = 0;
        for (int[] array : arrays) {
            size += array.length;
        }
        // create the result array of appropriate size
        int[] toReturn = new int[size];
        // add arrays
        for (int[] array : arrays) {
            for (int elem : array) {
                toReturn[index++] = elem;
            }
        }
        return toReturn;
    }

    private Instances removeAttributesOfA(Instances instances, Attribute ... except)
            throws Exception {
        List<Attribute> exceptList = Arrays.asList(except);
        // A filter to remove unnecessary columns
        Remove rm = new Remove();
        // Only the specified attributes are kept
        rm.setInvertSelection(true);
        // Only B attributes should remain (+ the exception list)
        int[] joinedIndexes = this.joinArrays(this.getIndexesOfAttributesOfB(),
                this.getIndexesFromAttributesList(exceptList));
        Arrays.sort(joinedIndexes);
        rm.setAttributeIndicesArray(joinedIndexes);
        rm.setInputFormat(instances);
        return Filter.useFilter(instances, rm);
    }

    private List<Attribute> selectFeaturesEx(Instances instances) throws Exception {
        instances.setClass(this.m_selectedSplittingAttribute);
        // Remove all except the splitting attribute (and attributes of B)
        Instances withOnlyNecessaryAttributes =
                this.removeAttributesOfA(instances, this.m_selectedSplittingAttribute);
        System.out.println("Selecting " + withOnlyNecessaryAttributes.classAttribute());
        AttributeSelection filter = new AttributeSelection();
        ASEvaluation evaluator = new CfsSubsetEval();
        ASSearch search = new BestFirst();
        filter.setEvaluator(evaluator);
        filter.setSearch(search);
        filter.setInputFormat(withOnlyNecessaryAttributes);
        Instances withSelectedAttributes = Filter.useFilter(withOnlyNecessaryAttributes, filter);
        List<Attribute> toReturn = new LinkedList<Attribute>();
        Enumeration attributesEnum = withSelectedAttributes.enumerateAttributes();
        while (attributesEnum.hasMoreElements()) {
            Attribute currentAttribute = (Attribute)attributesEnum.nextElement();
            if (!currentAttribute.equals(this.m_selectedSplittingAttribute)) {
                // Note that we take the real attribute from instance since the index of
                // currentAttribute is wrong
                toReturn.add(instances.attribute(currentAttribute.name()));
            }
        }
        return toReturn;
    }

    /**
     * The feature selection process is executed on the leaf dataset in order to choose the
     * attributes that will be finally represented
     *
     * @param instances The instances to select the features from
     * @return A list of the selected attributes
     *
     * @throws Exception In case something failed during the feature selection process
     */
    public List<Attribute> selectFeatures(Instances instances) throws Exception {
        Attribute previousClass = null;
        try {
            previousClass = instances.classAttribute();
        } catch (UnassignedClassException e) {
            // Do nothing
        }
        this.m_selectedFeatures = this.selectFeaturesEx(instances);
        this.m_isBuilt = true;
        if (previousClass != null) {
            instances.setClass(previousClass);
        }
        return this.m_selectedFeatures;
    }

    /**
     * The function returns the currently selected features which are internally stored
     *
     * @return A list of the selected features
     * @throws IllegalStateException In case the feature selector wasn't built
     */
    public List<Attribute> getSelectedFeatures() throws IllegalStateException {
        if (!this.m_isBuilt) {
            throw new IllegalStateException("Feature-Selector wasn't built");
        }
        return this.m_selectedFeatures;
    }
}

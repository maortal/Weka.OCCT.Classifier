package weka.classifiers.trees.occt.tree;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.trees.occt.split.auxiliary.ProbModelsHandler;
import weka.core.Attribute;
import weka.core.Instance;
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
 * Created by sepetnit on 01/02/15.
 *
 */
public class OCCTLeafNode implements Serializable {

    private static final long serialVersionUID = 42477850897166532L;

    private List<Attribute> selectedAttributes;

    private boolean m_performFeatureSelection;
    private Attribute m_selectedSplittingAttribute;
    private ProbModelsHandler m_probModels;

    private List<Attribute> m_attributesOfB;
    private static int[] B_INDEXES;

    /** How many instances we have under the leaf **/
    private int m_trainInstancesCount;

    static {
        OCCTLeafNode.B_INDEXES = null;
    }

    public OCCTLeafNode(List<Attribute> attributesOfB,
                        boolean performFeatureSelection) {
        this.m_attributesOfB = attributesOfB;
        this.m_performFeatureSelection = performFeatureSelection;
        this.m_probModels = null;
    }

    // By default: perform the feature extraction
    public OCCTLeafNode(List<Attribute> attributesOfB) {
        this(attributesOfB, true);
    }

    public OCCTLeafNode(List<Attribute> attributesOfB,
                        Attribute selectedSplittingAttribute) {
        this(attributesOfB, true);
        this.m_selectedSplittingAttribute = selectedSplittingAttribute;
    }

    public void buildLeaf(Instances instances) throws Exception {
        this.m_trainInstancesCount = instances.numInstances();
        // TODO: selected attribute can be null (if the root is a leaf)
        if (this.m_performFeatureSelection && this.m_selectedSplittingAttribute != null) {
            this.selectedAttributes = this.selectFeatures(instances);
        } else {
            this.selectedAttributes = this.m_attributesOfB;
        }
        this.m_probModels = new ProbModelsHandler(this.selectedAttributes);
        this.m_probModels.buildModels(instances);
    }

    @Override
    public String toString() {
        StringBuilder names = new StringBuilder();
        names.append("Leaf: {");
        boolean shouldAddComma = false;
        for (Attribute attribute : this.selectedAttributes) {
            if (!attribute.equals(this.m_selectedSplittingAttribute)) {
                if (shouldAddComma) {
                    names.append(", ");
                }
                shouldAddComma = true;
                names.append(attribute.name());
            }
        }
        names.append(" (");
        names.append(this.m_trainInstancesCount);
        names.append(")}");
        return names.toString();
    }

    private int[] getIndexesFromAttributesList(List<Attribute> attributes) {
        int index = 0;
        int[] toReturn = new int[attributes.size()];
        for (Attribute attribute : attributes) {
            toReturn[index++] = attribute.index();
        }
        return toReturn;
    }

    private int[] getIndexesOfAttributesOfB() {
        if (OCCTLeafNode.B_INDEXES == null) {
            // Find all the indexes of B's attributes (perform this a single time)
            OCCTLeafNode.B_INDEXES = this.getIndexesFromAttributesList(this.m_attributesOfB);
        }
        return OCCTLeafNode.B_INDEXES;
    }

    private Instances removeAttributesOfA(Instances instances) throws Exception {
        return removeAttributesOfA(instances, new LinkedList<Attribute>());
    }

    private Instances removeAttributesOfA(Instances instances,
                                          Attribute except) throws Exception {
        List<Attribute> exceptList = new LinkedList<Attribute>();
        exceptList.add(except);
        return removeAttributesOfA(instances, exceptList);
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

    private Instances removeAttributesOfA(Instances instances,
                                          List<Attribute> except) throws Exception {
        // A filter to remove unnecessary columns
        Remove rm = new Remove();
        // Only the specified attributes are kept
        rm.setInvertSelection(true);
        // Only B attributes should remain (+ the exception list)
        int[] joinedIndexes = this.joinArrays(this.getIndexesOfAttributesOfB(),
                this.getIndexesFromAttributesList(except));
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

    private List<Attribute> selectFeatures(Instances instances) throws Exception {
        Attribute previousClass = null;
        try {
            previousClass = instances.classAttribute();
        } catch (UnassignedClassException e) {
            // Do nothing
        }
        List<Attribute> toReturn = this.selectFeaturesEx(instances);
        if (previousClass != null) {
            instances.setClass(previousClass);
        }
        return toReturn;
    }

    /**
     * Calculates the likelihood for a match
     *
     * @param instance The instance to calculate the probability for
     *
     * @return The calculated value
     *
     * @throws Exception If something bad occurred
     */
    public double classifyInstance(Instance instance) throws Exception {
        return this.m_probModels.calculateLValueForSingleInstance(instance);
    }
}

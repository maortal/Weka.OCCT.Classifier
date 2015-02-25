package weka.classifiers.trees.occt.tree;

import weka.classifiers.trees.occt.split.auxiliary.OCCTProbModelsHandler;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.List;

/**
 * Created by sepetnit on 01/02/15.
 *
 */
public class OCCTLeafNode implements Serializable {

    private static final long serialVersionUID = 42477850897166532L;


    private OCCTProbModelsHandler m_probModels;

    private List<Attribute> m_attributesOfB;
    private List<Attribute> m_selectedAttributesOfB;

    /** How many instances we have under the leaf **/
    private int m_trainInstancesCount;

    public OCCTLeafNode(List<Attribute> attributesOfB,
                        List<Attribute> selectedAttributesOfB) {
        this.m_attributesOfB = attributesOfB;
        this.m_selectedAttributesOfB = selectedAttributesOfB;
        this.m_probModels = null;
    }

    // In case no selected features were given - use all the attributes of B
    public OCCTLeafNode(List<Attribute> attributesOfB) {
        this(attributesOfB, null);
    }

    public void buildLeaf(Instances instances) throws Exception {
        this.m_trainInstancesCount = instances.numInstances();
        // In case no selected features - use all the attributes of B
        if (this.m_selectedAttributesOfB == null) {
            this.m_selectedAttributesOfB = this.m_attributesOfB;
        }
        this.m_probModels = new OCCTProbModelsHandler(this.m_selectedAttributesOfB);
        this.m_probModels.buildModels(instances);
    }

    @Override
    public String toString() {
        StringBuilder names = new StringBuilder();
        names.append("Leaf: {");
        boolean shouldAddComma = false;
        for (Attribute attribute : this.m_selectedAttributesOfB) {
            if (shouldAddComma) {
                names.append(", ");
            }
            shouldAddComma = true;
            names.append(attribute.name());
        }
        names.append(" (");
        names.append(this.m_trainInstancesCount);
        names.append(")}");
        return names.toString();
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

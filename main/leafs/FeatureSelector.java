package leafs;

import utils.SimpleConfiguration;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

/**
 * Created by sepetnit on 28/12/14.
 *
 * This class is responsible of the feature selection process which is performed prior to building
 * probabilistic models which represent the leafs of the tree.OCCT
 */
public class FeatureSelector {
    private AttributeSelection filter;
    private CfsSubsetEval evaluator;
    private BestFirst search;

    public FeatureSelector() {
        this.filter = new AttributeSelection();
        this.evaluator = new CfsSubsetEval();
        this.search = new BestFirst();

        filter.setEvaluator(evaluator);
        filter.setSearch(search);
    }

    /**
     * Removes all the instances whose names are in the given set from the instances
     *
     * @param instances The instances to remove attributes from
     * @param attributesNamesToRemove The attributes to remove from the instances
     * @param exceptionAttribute An attribute to skip from attributesToRemove (it is not deleted)
     */
    private void removeAttributesFromInstances(Instances instances,
                                               String[] attributesNamesToRemove,
                                               String exceptionAttribute) {
        for (String attrNameToRemove: attributesNamesToRemove) {
            if (exceptionAttribute == null || !attrNameToRemove.equals(exceptionAttribute)) {
                Attribute attributeToRemove = instances.attribute(attrNameToRemove);
                if (attributeToRemove != null) {
                    instances.deleteAttributeAt(attributeToRemove.index());
                }
            }
        }
    }

    /**
     * Removes all the instances whose names are in the given set from the instances
     *
     * @param instances The instances to remove attributes from
     * @param attributesToRemove The attributes to remove from the instances
     */
    private void removeAttributesFromInstances(Instances instances, String[] attributesToRemove) {
        this.removeAttributesFromInstances(instances, attributesToRemove, null);
    }

    /**
     * The function extracts the names of all the attributes associated with the given instances
     * except the given attributes list
     *
     * @param instances The instances set to operate on
     * @param except The attributes to except (not returned in the result list)
     *
     * @return The extracted set of attributes (names only)
     */
    private List<String> extractAttributesFromInstancesWithException(Instances instances,
                                                                     List<String> except) {
        List<String> toReturn = new ArrayList<String>();
        Enumeration attributesEnum = instances.enumerateAttributes();
        while (attributesEnum.hasMoreElements()) {
            Attribute currentAttribute = (Attribute) attributesEnum.nextElement();
            if ((except == null) || !except.contains(currentAttribute.name())) {
                toReturn.add(currentAttribute.name());
            }
        }
        return toReturn;
    }

    /**
     * The function extracts the names of all the attributes associated with the given instances
     * except the given attribute
     *
     * @param instances The instances set to operate on
     * @param except An attribute to except
     *
     * @return The extracted set of attributes (names only)
     */
    private List<String> extractAttributesFromInstancesWithException(Instances instances,
                                                                     String except) {
        List<String> exceptionList = null;
        if (except != null) {
            exceptionList = new ArrayList<String>(1);
            exceptionList.add(except);
        }
        return this.extractAttributesFromInstancesWithException(instances, exceptionList);
    }

    /**
     * The function extracts the names of all the attributes associated with the given instances
     *
     * @param instances The instances set to operate on
     *
     * @return The extracted set of attributes (names only)
     */
    private List<String> extractAttributesFromInstancesWithException(Instances instances) {
        return this.extractAttributesFromInstancesWithException(instances, (List<String>)null);
    }

    /**
     * The function runs the Correlation-based Feature Subset Selection algorithm [31] feature
     * selection algorithm. This feature selection algorithm searches for a subset of features that
     * have high correlation with the class attribute and which have low correlation with one
     * another.
     *
     * The feature selection process is applied on a leaf data set and the class attribute is chosen
     * to be the last selected splitting attribute from A.
     *
     * At the end of the feature selection process, we are left with a set of features from B that
     * are highly correlated with ai, but are uncorrelated with one another.
     *
     * @param instances The instances set on which the feature selection process is ran
     * @param splittingAttribute The last chosen splitting attribute from A
     *
     * @return The selected list of chosen attributes from B, for which probabilistic models should
     *         be built
     *
     * @throws java.lang.NullPointerException In case the given splitting attribute in null
     */
    public List<String> runFeatureSelection(Instances instances, Attribute splittingAttribute)
            throws NullPointerException {
        // A safety ...
        if (splittingAttribute == null) {
            throw new NullPointerException("The given splitting attribute is null.");
        }

        // TODO: Is train the true one?
        Instances instancesCopy = new Instances(instances);
        instancesCopy.setClassIndex(splittingAttribute.index());

        // Remove A's attributes
        // TODO: Make it generic ... don't use configuration ...
        String[] attributesToRemove = SimpleConfiguration.valueOf("contextAttributes").split(",");
        this.removeAttributesFromInstances(instancesCopy,
                attributesToRemove,
                splittingAttribute.name());
        // Here the instances are ready for the feature selection process
        try {
            // Let's perform the feature selection
            this.filter.setInputFormat(instancesCopy);
            Instances filtered = Filter.useFilter(instancesCopy, this.filter);
            return this.extractAttributesFromInstancesWithException(filtered,
                    splittingAttribute.name());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return null;
    }
}

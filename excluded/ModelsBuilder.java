import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;


import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by sepetnit on 28/12/14.
 *
 * Given an Instances set and an attribute, this class is responsible of building a set of
 * probabilistic models for each attribute from the table B
 */

public class ModelsBuilder {
    private Instances instances;
    private Instances instancesWithOnlyBAttrs;
    private Set<Attribute> bAttributes;

    private boolean featureSelectionApplied;

    //Contains all the classifiers build by buildModels()
    List<Classifier> allClassifiers;
    // A map from the classifier to the relevant class attribute index
    Map<Classifier, Integer> classifierToClass;

    /**
     * The constructor of the class
     *
     * @param instances   The set of instances (A\Ad) X B
     * @param bAttributes The attributes of the table T-B
     */
    public ModelsBuilder(Instances instances, Set<Attribute> bAttributes) {
        this.instances = instances;
        this.bAttributes = bAttributes;
        // Prepare Instances set with only instances from B (used for creating the models)
        this.instancesWithOnlyBAttrs = new Instances(instances);
        for (Attribute attr : this.bAttributes) {
            instancesWithOnlyBAttrs.deleteAttributeAt(attr.index());
        }
        // Still no models were built
        this.featureSelectionApplied = false;
    }

    private Set<Attribute> getAttributesForModel(boolean shouldApplyFeatureSelection) {
        // TODO ...
        return null;
    }

    /**
     * This function actually builds the set of models. - given some splitting attribute and also
     * whether to apply a feature selection process on B's attribute, before building the set of
     * models
     *
     * @param shouldApplyFeatureSelection Whether feature selection should be applied for choosing
     *                                    attributes from B, before building the models
     */
    public void buildModels(Attribute splittingAttribute, boolean shouldApplyFeatureSelection) {
        // A own.general classifier used to build the current model
        Classifier currentModel;
        Set<Attribute> modelAttributes = this.getAttributesForModel(shouldApplyFeatureSelection);
        for (Attribute currentAttribute : modelAttributes) {
            if (currentAttribute.numValues() > 0) {
                if (currentAttribute.numValues() > 1) {
                    currentModel = new J48();
                    ((J48) currentModel).setUseLaplace(true);
                    // TODO: Is this required?
                    if (this.instances.numInstances() > 100000) {
                        ((J48) currentModel).setMinNumObj(this.instances.numInstances());
                    }
                    this.instancesWithOnlyBAttrs.setClass(currentAttribute);
                    try {
                        currentModel.buildClassifier(this.instancesWithOnlyBAttrs);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    // currentAttribute.numValues() == 1
                } else {
                    currentModel = new UnaryClassifier();
                    // TODO: Set class attribute???
                    Enumeration currentAttributeValues = currentAttribute.enumerateValues();
                    String singleValue = (String) currentAttributeValues.nextElement();
                    ((UnaryClassifier) currentModel).setUnaryValue(singleValue);
                }
                this.allClassifiers.add(currentModel);
                this.classifierToClass.put(currentModel, currentAttribute.index());
            }

        }
    }
}

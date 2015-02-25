package weka.classifiers.trees.occt.split.auxiliary;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.filters.unsupervised.attribute.Remove;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by sepetnit on 31/01/15.
 *
 */
public class OCCTProbModelsHandler implements Serializable {

    /** for serialization */
    static final long serialVersionUID = -4813820170237218194L;

    OCCTProbModelBasicCreator m_probModelInitializer;
    private List<Attribute> m_attributesToBuildFrom;
    private int[] m_attributesToBuildFromIndexes;
    // Indicated whether the probabilistic models were created
    private boolean m_built;
    // Models for each of the attributes of B, given other attributes
    private Map<Attribute, Classifier> m_probabilisticModels;

    private void initializeAttributesIndexes() {
        int index = 0;
        // Find all the indexes of B's attributes (perform this a single time)
        this.m_attributesToBuildFromIndexes = new int[this.m_attributesToBuildFrom.size()];
        for (Attribute attribute : this.m_attributesToBuildFrom) {
            System.out.println("Will build from " + attribute.index());
            this.m_attributesToBuildFromIndexes[index++] = attribute.index();
        }
    }

    public OCCTProbModelsHandler(List<Attribute> attributesToBuildFrom,
                                 OCCTProbModelBasicCreator probModelInitializer) {
        this.m_probModelInitializer = probModelInitializer;
        this.m_attributesToBuildFrom = attributesToBuildFrom;
        this.m_probabilisticModels = new HashMap<Attribute,
                Classifier>(this.m_attributesToBuildFrom.size());
        this.initializeAttributesIndexes();
        this.m_built = false;
    }


    public OCCTProbModelsHandler(List<Attribute> attributesToBuildFrom) {
        this(attributesToBuildFrom, new OCCTSerializableBasicModelsCreator());
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
        rm.setAttributeIndicesArray(this.m_attributesToBuildFromIndexes);
        return rm;
    }

    private void buildModelsEx(Instances instances) throws Exception {
        // First, let's remove the unnecessary attributes
        this.m_built = false;
        this.m_probabilisticModels.clear();
        for (Attribute currentAttribute : this.m_attributesToBuildFrom) {
            instances.setClass(currentAttribute);
            FilteredClassifier filteredClassifier = new FilteredClassifier();
            filteredClassifier.setFilter(this.getRemoveFilter());
            filteredClassifier.setClassifier(
                    this.m_probModelInitializer.getClassifierForProbModel());
            // Here, the Remove filter learns the class index
            filteredClassifier.buildClassifier(instances);
            this.m_probabilisticModels.put(currentAttribute, filteredClassifier);
        }
        this.m_built = true;
    }

    public void buildModels(Instances instances) throws Exception {
        Attribute previousClass = null;
        try {
            previousClass = instances.classAttribute();
        } catch (UnassignedClassException e) {
            // Do nothing
        }
        this.buildModelsEx(instances);
        if (previousClass != null) {
            instances.setClass(previousClass);
        }
    }

    public double calculateLValueForSingleInstance(Instance currentInstance) throws Exception {
        return this.calculateLValueForSingleInstance(currentInstance.dataset(), currentInstance);
    }

    private double calculateLValueForSingleInstanceEx(Instances allInstances,
                                                      Instance currentInstance) throws Exception {
        double toReturn = 0;
        // Build the models if the user forgot to call the buildModels() method
        if (!this.m_built) {
            this.buildModels(allInstances);
        }
        for (Map.Entry<Attribute, Classifier> entry : this.m_probabilisticModels.entrySet()) {
            Attribute currentAttribute = entry.getKey();
            Classifier currentClassifier = entry.getValue();
            allInstances.setClass(currentAttribute);
            double[] dist = currentClassifier.distributionForInstance(currentInstance);
            System.out.println("******** dist is " + Arrays.toString(dist) + "******** ");
            System.out.println(Arrays.toString(dist));
            System.out.println(currentAttribute);
            System.out.println(currentInstance);
            System.out.println(currentAttribute.index());
            System.out.println(currentInstance.stringValue(currentAttribute));
            int valueIndex = currentAttribute.indexOfValue(
                    currentInstance.stringValue(currentAttribute));
            toReturn += Math.log10(dist[valueIndex]);
        }
        return toReturn;
    }

    public double calculateLValueForSingleInstance(Instances allInstances,
                                                   Instance currentInstance) throws Exception {
        Attribute previousClass = null;
        try {
            previousClass = allInstances.classAttribute();
        } catch (UnassignedClassException e) {
            // Do nothing
        }
        double toReturn = this.calculateLValueForSingleInstanceEx(allInstances, currentInstance);
        if (previousClass != null) {
            allInstances.setClass(previousClass);
        }
        return toReturn;
    }

}

class OCCTSerializableBasicModelsCreator implements OCCTProbModelBasicCreator, Serializable {

    private static final long serialVersionUID = -4425482033289259153L;

    public Classifier getClassifierForProbModel() {
        J48 currentActualClassifier = new J48();
        currentActualClassifier.setUseLaplace(true);
        currentActualClassifier.setUnpruned(true);
        return currentActualClassifier;
    }
}

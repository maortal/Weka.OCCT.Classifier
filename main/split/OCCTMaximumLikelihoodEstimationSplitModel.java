package split.models;

import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;
import split.auxiliary.OCCTSplitModelComparators;
import split.iterators.GeneralInstancesIterator;
import split.iterators.SingleInstancesPairIterator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;

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

    private void buildModels(Instances instances) {
        try {
        // First, let's remove the unnecessary attributes
        this.probabilisticModels.clear();
        Remove rm = new Remove();
        rm.setInvertSelection(true);
        rm.setAttributeIndicesArray(this.attributesOfBIndexes);
        for (Attribute attr : this.m_attributesOfB) {
            System.out.println("Building model for attr " + attr.name() + " index " + attr.index());
            FilteredClassifier filteredClassifier = new FilteredClassifier();
            J48 currentActualClassifier = new J48();
            currentActualClassifier.setUseLaplace(true);
            currentActualClassifier.setUnpruned(true);
            filteredClassifier.setFilter(rm);
            filteredClassifier.setClassifier(currentActualClassifier);
            // Train and make predictions
            try {
                // TODO: Set minimum number of objects in a leaf? (setMinNumObj)
                instances.setClass(attr);
                filteredClassifier.buildClassifier(instances);
                //System.out.println(filteredClassifier.graph());

                System.out.println("Start of print");
                //System.out.println(instances.toString());
                System.out.println("End of print");

                System.out.println(filteredClassifier.toString());
                Enumeration instancesEnum = instances.enumerateInstances();
                while (instancesEnum.hasMoreElements()) {
                    Instance currentInstance = (Instance) instancesEnum.nextElement();
                    if (currentInstance.stringValue(attr).equals("Hamburg")) {
                        System.out.println("Classifying instance " + currentInstance.toString());
                        System.out.println("Value " + Arrays.toString(filteredClassifier.distributionForInstance(currentInstance)));
                        System.out.println("Value " + Arrays.toString(filteredClassifier.distributionForInstance(currentInstance)));
                    }
                }
            }
            catch (Exception e) {
                e.printStackTrace();
            }
            System.out.println("put classifier " + attr.name() + " fil " + filteredClassifier.hashCode());
            this.probabilisticModels.put(attr, filteredClassifier);
        }
            System.out.println("A");
        System.out.println(this.probabilisticModels.toString());
        System.out.println("Done");
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private double calculateLValueForSingleInstance(Instances allInstances,
                                                    Instance currentInstance) throws Exception {
        double toReturn = 0;
        for (Map.Entry<Attribute, Classifier> entry : this.probabilisticModels.entrySet()) {
            Attribute currentAttribute = entry.getKey();
            Classifier currentClassifier = entry.getValue();
            System.out.println("Got classifier" + currentClassifier.hashCode());
            System.out.println("Testing model for attribute " + currentAttribute.name());
            allInstances.setClass(currentAttribute);
            System.out.println("Class index of instances is " + allInstances.classIndex());
            System.out.println("Class index of instance is " + currentInstance.classIndex());
            System.out.println("Classifier is " + currentClassifier.toString());
            System.out.println("Instance is " + currentInstance.toString());

            FilteredClassifier c = (FilteredClassifier)currentClassifier;
            Instances i = Filter.useFilter(allInstances, c.getFilter());
            System.out.println("Filtered is i" );
            System.out.println(i.toString());
            System.out.println("class " + i.classIndex());



            double[] dist = ((FilteredClassifier)currentClassifier).distributionForInstance(currentInstance);
            System.out.println("Classificationn is " + currentClassifier.classifyInstance(currentInstance));
            if (currentInstance.stringValue(currentAttribute).equals("Hamburg")) {
                System.out.println("Classes " + currentInstance.numClasses());
                System.out.println("Dist is " + Arrays.toString(dist));
                System.out.println(dist.length);
                dist = currentClassifier.distributionForInstance(currentInstance);
                System.out.println("Dist is " + Arrays.toString(dist));
            }
            System.out.println("Values count : " + currentAttribute.numValues());
            Enumeration e = currentAttribute.enumerateValues();
            while (e.hasMoreElements()) {
                String t = (String)e.nextElement();
                System.out.println("Value is " + t);
            }
            System.out.println("Current Value is " + currentInstance.stringValue(currentAttribute));
            int valueIndex = currentAttribute.indexOfValue(
                    currentInstance.stringValue(currentAttribute));
            toReturn += Math.log(dist[valueIndex]);
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
        System.out.println(this.m_splittingAttribute.name());
        // Let's assert that the second instance is null
        // (since we calculate score for a single instance only in MLE)
        assert (i2 != null);
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
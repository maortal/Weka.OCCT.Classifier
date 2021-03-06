package weka.classifiers.trees.occt.split.models;

import weka.classifiers.trees.occt.split.iterators.GeneralInstancesIterator;
import weka.classifiers.trees.occt.split.iterators.PairedInstancesIterator;
import weka.classifiers.trees.occt.utils.OCCTNotImplementedException;
import weka.classifiers.trees.occt.utils.OCCTPair;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;

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
public abstract class OCCTSingleAttributeSplitModel extends OCCTSplitModel {

    /** for serialization */
    private static final long serialVersionUID = 3064079330067903161L;

    protected Attribute m_splittingAttribute;
    private int m_complexityIndex;
    private double m_splittingScore;

    protected List<Attribute> m_possibleAttributes;
    protected List<Attribute> m_attributesOfB;

    public static Comparator<Double> SCORES_COMPARATOR;

    public OCCTSingleAttributeSplitModel(Attribute splittingAttribute,
                                         List<Attribute> possibleAttributes,
                                         List<Attribute> attributesOfB) {
        super();
        this.m_splittingAttribute = splittingAttribute;
        this.m_attributesOfB = attributesOfB;
        this.m_possibleAttributes = possibleAttributes;
    }

    /**
     * The function builds a string representation of the values of an instances, based on the
     * values of the possible attributes for that model and except the splitting attribute.
     *
     * @param instance The instance whose string representation should be returned
     * @param includeSplittingAttribute Whether the splitting attribute should be includes in the
     *                                  the created string
     *
     * @return The created string representation
     */
    protected String buildAttrValuesString(Instance instance,
                                           boolean includeSplittingAttribute) {
        StringBuilder toReturn = new StringBuilder();
        Enumeration attributes = instance.enumerateAttributes();
        while (attributes.hasMoreElements()) {
            Attribute currentAttr = (Attribute) attributes.nextElement();
            if (currentAttr.equals(this.m_splittingAttribute)) {
                if (includeSplittingAttribute) {
                    toReturn.append(instance.stringValue(currentAttr));
                }
            } else if (this.m_possibleAttributes.contains(currentAttr)) {
                toReturn.append(instance.stringValue(currentAttr));
            }
        }
        return toReturn.toString();
    }

    protected String buildAttrValuesString(Instance instance) {
        return this.buildAttrValuesString(instance, false);
    }

    protected Set<String> calculateIntersection(Instances i1, Instances i2) {
        // Used to avoid duplicates
        Set<String> intersection = new HashSet<String>();
        // Calculate the distance between each pair of instances
        Enumeration inst1Enum = i1.enumerateInstances();
        while (inst1Enum.hasMoreElements()) {
            Instance currentI1 = (Instance) inst1Enum.nextElement();
            Enumeration inst2Enum = i2.enumerateInstances();
            while (inst2Enum.hasMoreElements()) {
                Instance currentI2 = (Instance) inst2Enum.nextElement();
                // Build values strings
                String currentI1String = this.buildAttrValuesString(currentI1);
                String currentI2String = this.buildAttrValuesString(currentI2);
                // Compare the strings and update the intersection
                if (currentI1String.equals(currentI2String)) {
                    intersection.add(currentI1String);
                }
            }
        }
        return intersection;
    }

    protected int calculateIntersectionSize(Instances i1, Instances i2) {
        return this.calculateIntersection(i1, i2).size();
    }

    /**
     * Splits a dataset according to the values of an attribute
     *
     * @param data the data which is to be weka.trees.classifiers.occt.split
     * @param attr the attribute to be used for splitting
     *
     * @return the sets of instances produced by the weka.trees.classifiers.occt.split
     */
    protected static Instances[] splitInstancesByAttribute(Instances data, Attribute attr) {
        Instances[] splitted = new Instances[attr.numValues()];
        // Will contain only non-empty instances
        List<Instances> nonEmpty = new LinkedList<Instances>();

        // Create a single array of instances for each value of the attribute attr
        for (int j = 0; j < attr.numValues(); j++) {
            splitted[j] = new Instances(data, data.numInstances());
        }
        // Go over all instances and divide them to the buckets
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitted[(int) inst.value(attr)].add(inst);
        }
        // Finally, make each bucket to be of size which is equal to its real capacity
        for (Instances instances : splitted) {
            instances.compactify();
            if (instances.numInstances() > 0) {
                nonEmpty.add(instances);
            }
        }
        return nonEmpty.toArray(new Instances[nonEmpty.size()]);
    }

    protected double calculateSplitScore(OCCTPair<Instances, Instances> instancesSetsPair)
            throws Exception {
        return this.calculateSplitScore(instancesSetsPair.getFirst(),
                instancesSetsPair.getSecond());
    }

    /**
     * Given two different sets of instances, this function is responsible of calculating the
     * score (a number between 0 and 1) which denotes the similarity between the sets
     *
     * @param i1 The first set of instances
     * @param i2 The second set of instances
     *
     * The function assumes that i1 and i2 have the same header
     *
     * @return The calculated similarity score (a number between 0 and 1)
     */
    protected abstract double calculateSplitScore(Instances i1, Instances i2) throws Exception;

    protected GeneralInstancesIterator getInstancesSetsIterator(
            Instances trainInstances, Instances[] splittedTrainInstances) {
        return new PairedInstancesIterator(trainInstances, splittedTrainInstances);
    }

    /**
     * Returns the score of the model for a nominal attribute, calculated for specific data
     *
     * @param trainInstances The instances for which the split score should be calculated
     *
     * @return The calculated score
     *
     * @exception Exception if something goes wrong
     */
    protected double handleEnumeratedAttribute(Instances trainInstances) throws Exception {
        System.out.println("Handling splitting " + this.m_splittingAttribute.name());
        double toReturn = 0;
        // Split the instances according to the attribute (single set for each value)
        Instances[] splitted = OCCTSingleAttributeSplitModel.splitInstancesByAttribute(
                trainInstances, this.m_splittingAttribute);
        if (splitted.length == 1) {
            // TODO: Throw some better exception ...
            // The model as invalid!
            System.out.println("Invalid model ...");
            throw new NullPointerException();
        } else {
            GeneralInstancesIterator instancesSetsIter =
                    this.getInstancesSetsIterator(trainInstances, splitted);
            while (instancesSetsIter.hasNext()) {
                OCCTPair<OCCTPair<Instances, Instances>, Double> instancesAndWeights =
                        instancesSetsIter.next();
                OCCTPair<Instances, Instances> instancesSetsToCompare = instancesAndWeights.getFirst();
                double weight = instancesAndWeights.getSecond();
                // Calculate the relative part of this binary weka.trees.classifiers.occt.split
                toReturn += weight * this.calculateSplitScore(instancesSetsToCompare);
            }
        }
        System.out.println("Overall split score for " + this.m_splittingAttribute.name() + " is " + toReturn);
        return toReturn;
    }

    /**
     * Handles numeric attribute.
     *
     * @exception Exception if something goes wrong
     */
    private double handleNumericAttribute(Instances trainInstances) throws Exception {
        throw new OCCTNotImplementedException();
    }


    @Override
    public void buildClassifier(Instances instances) throws Exception {

        try {
            // Different treatment for nominal and numeric attributes.
            if (this.m_splittingAttribute.isNominal()) {
                this.m_complexityIndex = this.m_splittingAttribute.numValues();
                this.m_splittingScore = this.handleEnumeratedAttribute(instances);
                System.out.println("Splitting score " + this.m_splittingScore);
                // TODO: m_minNoObj?
                if (instances.numInstances() > 0) {
                    this.m_numSubsets = m_complexityIndex;
                }
            } else {
                this.m_complexityIndex = 2;
                // TODO ...
                instances.sort(this.m_splittingAttribute);
                this.m_splittingScore = this.handleNumericAttribute(instances);
            }
        } catch (NullPointerException e) {
            this.m_numSubsets = 0;
        }
    }

    public int getSplittingAttributeIndex() {
        return this.m_splittingAttribute.index();
    }

    @Override
    public double score() {
        return this.m_splittingScore;
    }

    @Override
    /**
     * Prints left side of condition..
     *
     * @param data training set.
     */
    public String leftSide(Instances data) {
        return this.m_splittingAttribute.name();
    }

    @Override
    public String rightSide(String value, Instances data) {
        return this.rightSide(this.m_splittingAttribute.indexOfValue(value), data);
    }

    /**
     * Prints the condition satisfied by instances in a subset.
     *
     * @param index of subset
     * @param data training set.
     */
    @Override
    public String rightSide(int index, Instances data) {
        StringBuilder toReturn = new StringBuilder();
        if (this.m_splittingAttribute.isNominal()) {
            toReturn.append(" = ");
            toReturn.append(this.m_splittingAttribute.value(index));
            toReturn.append("");
        } else if (this.m_splittingAttribute.isNumeric()) {
            //throw new NotImplementedException();
        }
        return toReturn.toString();
    }

    /**
     * @return The attribute which is used by this model
     */
    public Attribute getChosenAttribute() {
        return this.m_splittingAttribute;
    }


    @Override
    public String sourceExpression(int index, Instances data) {
        return null;
    }

    @Override
    public double[] weights(Instance instance) {
        return null;
    }

    @Override
    public int whichSubset(Instance instance) throws Exception {
        // In case the splitting attribute has no value on this instances - return -1 and handle
        // by the {@see weights method}
        if (instance.isMissing(this.m_splittingAttribute)) {
            return -1;
        } else {
            // Treat differently for nominal and numeric values
            if (this.m_splittingAttribute.isNominal()) {
                return (int) instance.value(this.m_splittingAttribute);
            // Currently, no implementation for numeric attributes
            } else {
                throw new OCCTNotImplementedException();
            }
        }
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.0 $");
    }
}

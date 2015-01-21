package split;

import sun.awt.image.ImageWatched;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.core.*;

import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by sepetnit on 30/12/14.
 *
 */
public abstract class OCCTSingleAttributeSplitModel extends ClassifierSplitModel {

    /** for serialization */
    private static final long serialVersionUID = 3064079330067903161L;

    protected Attribute m_splittingAttribute;
    private int m_complexityIndex;
    private double m_splittingScore;

    //protected Set<Attribute> m_attributesOfB;
    protected List<Attribute> m_possibleAttributes;

    public static Comparator<Double> m_scoresComparator;

    public OCCTSingleAttributeSplitModel(Attribute splittingAttribute,
                                         List<Attribute> possibleAttributes) {
        this.m_splittingAttribute = splittingAttribute;
        //this.m_attributesOfB = attributesOfB;
        this.m_possibleAttributes = possibleAttributes;
    }

    public int getSplittingAttributeIndex() {
        return this.m_splittingAttribute.index();
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
            if (this.m_splittingAttribute.equals(currentAttr)) {
                if (includeSplittingAttribute) {
                    toReturn.append(instance.toString(currentAttr));
                }
            } else if (this.m_possibleAttributes.contains(currentAttr)) {
                toReturn.append(instance.toString(currentAttr));
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
            //System.out.println();
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
     * @param data the data which is to be split
     * @param attr the attribute to be used for splitting
     *
     * @return the sets of instances produced by the split
     */
    private static Instances[] splitInstancesByAttribute(Instances data, Attribute attr) {
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
    protected abstract double calculateSplitScore(Instances i1, Instances i2);

    /**
     * Creates split on enumerated attribute.
     *
     * @exception Exception if something goes wrong
     */
    private double handleEnumeratedAttribute(Instances trainInstances) throws Exception {
        System.out.println("Handling splitting " + this.m_splittingAttribute.name());
        double toReturn = 0;
        // Split the instances according to the attribute (single set for each value)
        Instances[] splitted = OCCTSingleAttributeSplitModel.splitInstancesByAttribute(
                trainInstances, this.m_splittingAttribute);
        if (splitted.length == 1) {
            // The model as invalid!
            throw new NullPointerException();
        }
        if (splitted.length == 2) {
            toReturn = this.calculateSplitScore(splitted[0], splitted[1]);
        } else {
            // Go over the splitted data and aggregate the scores
            for (int i = 0; i < splitted.length; ++i) {
                if (splitted[i].numInstances() > 0) {
                    // i1 contains all instances with ai == d and i2 contains all instances with ai != d
                    Instances i1 = new Instances(splitted[i]);
                    Instances i2 = new Instances(trainInstances, trainInstances.numInstances() -
                            i1.numInstances());
                    for (int j = 0; j < splitted.length; ++j) {
                        if (j != i) {
                            Enumeration currentSplittedEnum = splitted[j].enumerateInstances();
                            while (currentSplittedEnum.hasMoreElements()) {
                                Instance current = (Instance) currentSplittedEnum.nextElement();
                                i2.add(current);
                            }
                        }
                    }
                    // The weight of this pair is proportional to the size of the subsets to
                    // ensure that the resulted splitting value of an attribute will be influenced mainly by
                    // records having more dominant values and not by esoteric ones
                    double weight = (splitted[i].numInstances() /
                            (double) trainInstances.numInstances());
                    // Now calculate the relative part of this binary split
                    // System.out.println("Weight: " + weight);
                    toReturn += weight * this.calculateSplitScore(i1, i2);
                }
            }
        }
        System.out.println("Overall split score for " + this.m_splittingAttribute.name() + " is " + toReturn);
        return toReturn;
    }

    /**
     * Creates split on enumerated attribute.
     *
     * @exception Exception if something goes wrong
     */
    private double handleNumericAttribute(Instances trainInstances) throws Exception {
        throw new NotImplementedException();
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
    /**
     * Prints the condition satisfied by instances in a subset.
     *
     * @param index of subset
     * @param data training set.
     */
    public String rightSide(int index, Instances data) {
        StringBuilder text = new StringBuilder();
        if (this.m_splittingAttribute.isNominal()) {
            text.append(" = ");
            text.append(this.m_splittingAttribute.value(index));
        } else if (this.m_splittingAttribute.isNumeric()) {
            //throw new NotImplementedException();
        }
        return text.toString();
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
                throw new NotImplementedException();
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

package weka.classifiers.trees.occt.tree;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by sepetnit on 15/02/15.
 *
 */
public class OCCTCardinalityHandler implements Serializable {

    private static final long serialVersionUID = 1971843965121510087L;

    /** The computed cardinality **/
    protected Map<String, Double> m_cardinality;
    /** The train dataset **/
    protected Instances m_train;
    /** The first index of the B's attributes **/
    protected int m_firstIndexOfB;
    /** Don't include these attributes in the calculation **/
    protected List<Attribute> m_exceptionAttributes;

    public OCCTCardinalityHandler(boolean shouldBuildNow, Instances instances, int firstIndexOfB,
                                  Attribute... except) {
        this.m_train = instances;
        this.m_firstIndexOfB = firstIndexOfB;
        this.m_exceptionAttributes = Arrays.asList(except);
        this.m_cardinality = null;
        if (shouldBuildNow) {
            this.buildCardinality(instances, firstIndexOfB);
        }
    }

    public OCCTCardinalityHandler(Instances instances, int firstIndexOfB,
                                  boolean shouldBuildNow) {
        this(shouldBuildNow, instances, firstIndexOfB);
    }

    public void setExceptionAttributesOfB(Attribute... except) {
        this.m_exceptionAttributes = Arrays.asList(except);
    }

    private String getBRecordAsString(Instance instance, int firstIndexOfB) {
        StringBuilder second = new StringBuilder();
        for (int i = firstIndexOfB; i < instance.numAttributes(); ++i) {
            if (!this.m_exceptionAttributes.contains(instance.attribute(i))) {
                second.append(instance.stringValue(i));
            }
        }
        return second.toString();
    }

    private void finalizeCardinality() {
        if (this.m_cardinality != null) {
            Map<String, Double> updatedCardinality = new HashMap<String, Double>();
            for (Map.Entry<String, Double> entry : this.m_cardinality.entrySet()) {
                updatedCardinality.put(entry.getKey(),
                        entry.getValue() / this.m_train.numInstances());
            }
            this.m_cardinality = updatedCardinality;
        }
    }

    public void buildCardinality() {
        this.buildCardinality(this.m_train, this.m_firstIndexOfB);
    }

    private void checkInput(Instances instances, int firstIndexOfB)
            throws IllegalArgumentException{
        if (instances == null) {
            throw new IllegalArgumentException("Can't build cardinality without instances");
        }
        if (firstIndexOfB == -1) {
            throw new IllegalArgumentException("First index of B must be != -1");
        }
    }

    public void buildCardinality(Instances instances, int firstIndexOfB)
            throws IllegalArgumentException {
        this.checkInput(instances, firstIndexOfB);
        this.m_cardinality = new HashMap<String, Double>();
        Enumeration instancesEnum = instances.enumerateInstances();
        while (instancesEnum.hasMoreElements()) {
            Instance currentInstance = (Instance)instancesEnum.nextElement();
            String bRecordAsString = this.getBRecordAsString(currentInstance, firstIndexOfB);
            // We can't use getOrDefault (since it exists only in Java >= 1.8)
            double currentCount = 0.0;
            if (this.m_cardinality.containsKey(bRecordAsString)) {
                currentCount = this.m_cardinality.get(bRecordAsString);
            }
            this.m_cardinality.put(bRecordAsString, currentCount + 1);
        }
        this.finalizeCardinality();
    }

    public double getCardinalityValue(Instance instance) {
        if (this.m_cardinality == null) {
            System.out.println("no");
            return 0;
        }
        String bRecordAsString = this.getBRecordAsString(instance, this.m_firstIndexOfB);
        System.out.println("Str " + bRecordAsString);
        if (!this.m_cardinality.containsKey(bRecordAsString)) {
            System.out.println("noooo");
            return 0;
        }
        return this.m_cardinality.get(bRecordAsString);
    }

    public void cleanup() {
        this.m_train = null;
    }

    public void fullCleanup() {
        this.cleanup();
        this.m_cardinality = null;
        this.m_firstIndexOfB = -1;
    }
}

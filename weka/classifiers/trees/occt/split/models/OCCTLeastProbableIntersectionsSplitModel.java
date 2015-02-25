package weka.classifiers.trees.occt.split.models;

import weka.classifiers.trees.occt.split.auxiliary.OCCTSplitModelComparators;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by sepetnit on 12/20/2014.
 *

 */
public class OCCTLeastProbableIntersectionsSplitModel extends OCCTSingleAttributeSplitModel {

    static {
        OCCTLeastProbableIntersectionsSplitModel.SCORES_COMPARATOR =
                OCCTSplitModelComparators.HIGHEST_SCORE_CHOOSER;
    }

    public OCCTLeastProbableIntersectionsSplitModel(Attribute attr,
                                                    List<Attribute> possibleAttributes,
                                                    List<Attribute> attributesOfB) {
        super(attr, possibleAttributes, attributesOfB);
    }

    private int countAppearances(String instanceAttrsStr, Instances instances) {
        int toReturn = 0;
        // Calculate the distance between each pair of instances
        Enumeration instEnum = instances.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance currentInstance = (Instance) instEnum.nextElement();
            //System.out.println("is " + instanceAttrsStr);
            if (instanceAttrsStr.equals(this.buildAttrValuesString(currentInstance, true))) {
                ++toReturn;
            }
        }
        System.out.println("a");
        return toReturn;
    }

    /**
     * The function counts the number of appearances of some instance in a given set of instances
     *
     * @param instance The instance to look for
     * @param instances The set of instances to look in
     *
     * @return The number of appearances of i in instances
     */
    private int countAppearances(Instance instance, Instances instances) {
        // Include the current splitting attribute
        return this.countAppearances(this.buildAttrValuesString(instance, true), instances);
    }

    /**
     * Calculates single probability of an instance being a part of the intersection of
     * a random split of instances to sets of the same sizes like the given sets of instances
     *
     * @param instance An instance whose probability should be calculated
     * @param allInstancesSets All the sets of instances to calculate according too
     *
     * @return The calculated probability (a number in [0, 1])
     */
    private double calculateSingleProbability(Instance instance,
                                              Instances[] allInstancesSets) {
        int oi = 0;
        int totalInstancesCount = 0;
        // First, let's calculate the total number of instances
        for (Instances instances: allInstancesSets) {
            //System.out.println(instances);
            oi += this.countAppearances(instance, instances);
            //System.out.println("Current oi is " + oi);
            totalInstancesCount += instances.numInstances();
        }
        // The final Pi value
        double toReturn = 1.0;
        // Now, calculate the value of Pi
        for (Instances instances: allInstancesSets) {
            System.out.println(allInstancesSets.length);
            //System.out.println("num: " + instances.numInstances() + " total " + totalInstancesCount + " oi " + oi);
            toReturn -= Math.pow(instances.numInstances() / (double)totalInstancesCount, oi);
        }
        return toReturn;
    }

    /**
     * The function calculates lambda value which is the sum of single probabilities, each one
     * denotes the probability of a record to belong to all the sets of instances in a random
     * split of the unified set to sets of same sizes. For further explanation look at the
     * description of {@see calculateSingleProbability}
     *
     * @param allInstancesSets The sets of instances (divided)
     *
     * @return The calculated value of lambda
     */
    private double calculateLambda(Instances[] allInstancesSets) {
        double lambdaValue = 0;
        // Used in order to calculate p-values for each distinct record only
        Set<String> distinctStringRepresentations = new HashSet<String>();
        for (Instances instances: allInstancesSets) {
            Enumeration instancesEnum = instances.enumerateInstances();
            while (instancesEnum.hasMoreElements()) {
                Instance currentInstance = (Instance)instancesEnum.nextElement();
                //String currentInstanceStr = currentInstance.toString(currentInstance.numAttributes() - 2) +
                //        currentInstance.toString(currentInstance.numAttributes() - 1);
                // Use all attributes, including the splitting attribute
                String currentInstanceStr = this.buildAttrValuesString(currentInstance, true);
                if (!distinctStringRepresentations.contains(currentInstanceStr)) {
                    double value = this.calculateSingleProbability(currentInstance, allInstancesSets);
                    System.out.println(currentInstanceStr + " is " + value);
                    lambdaValue += value;
                    distinctStringRepresentations.add(currentInstanceStr);
                }
            }
            // If you want to run only on instances where vi = di (the first table),
            // break here.
            // break;
        }
        //System.out.println("And total is " + lambdaValue);
        return lambdaValue;
    }

    protected double calculateSplitScore(Instances i1, Instances i2) {
        double stdevToReturn = 0;
        double sumOfPi = this.calculateLambda(new Instances[] {i1, i2});
        System.out.println("lambda is " + sumOfPi);
        if (sumOfPi >= 0) {
            int j = this.calculateIntersectionSize(i1, i2);
            System.out.println("J is " + j);
            if (j > 0) {
                stdevToReturn = (j - sumOfPi) / Math.sqrt(sumOfPi);
            } else {
                System.out.println("Negative ...");
                stdevToReturn = sumOfPi - j;
            }

        }
        System.out.println("Stdev is " + Math.abs(stdevToReturn));
        return Math.abs(stdevToReturn);
    }

}

package split;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by sepetnit on 12/20/2014.
 *

 */
public class OCCTLeastProbableIntersectionsSplitModel extends OCCTSingleAttributeSplitModel {

    static {
        OCCTLeastProbableIntersectionsSplitModel.m_scoresComparator =  new Comparator<Double>() {
            public int compare(Double score1, Double score2) {
                if (Utils.gr(score1, score2)) {
                    return 1;
                } else if (Utils.gr(score1, score2)) {
                    return -1;
                }
                return 0;
            }
        };
    }

    public OCCTLeastProbableIntersectionsSplitModel(Attribute attr,
                                                    List<Attribute> possibleAttributes) {
        super(attr, possibleAttributes);
    }

    private int countAppearances(String instanceAttrsStr, Instances instances) {
        int toReturn = 0;
        // Calculate the distance between each pair of instances
        Enumeration instEnum = instances.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance currentInstance = (Instance) instEnum.nextElement();
            if (instanceAttrsStr.equals(this.buildAttrValuesString(currentInstance, true))) {
                ++toReturn;
            }
        }
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
        // The final Pi value
        double toReturn = 1;
        // First, let's calculate the total number of instances
        for (Instances instances: allInstancesSets) {
            oi += this.countAppearances(instance, instances);
            totalInstancesCount += instances.numInstances();
        }
        // Now, calculate the value of Pi
        for (Instances instances: allInstancesSets) {
            System.out.println("num: " + instances.numInstances() + " total " + totalInstancesCount + " oi " + oi);
            toReturn -= Math.pow(instances.numInstances() / (double)totalInstancesCount, oi);
        }
        System.out.println("toreturn " + toReturn);
        /*if (instance.toString().equals("Afternoon,Friday,Berlin,Berlin,private")) {
            System.out.println("Oi is " + oi);
            System.out.println("P is " + toReturn);
        }*/
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
                System.out.println(currentInstanceStr + " " +
                        distinctStringRepresentations.contains(currentInstanceStr));
                //System.out.println(currentInstance.toString());
                // Calculate if required
                if (!distinctStringRepresentations.contains(currentInstanceStr)) {
                    lambdaValue +=
                            this.calculateSingleProbability(currentInstance, allInstancesSets);
                    distinctStringRepresentations.add(currentInstanceStr);
                    System.out.println(Arrays.toString(distinctStringRepresentations.toArray()));
                }
            }
        }
        return lambdaValue;
    }

    protected double calculateSplitScore(Instances i1, Instances i2) {
        double STDevToReturn = 0;
        double sumOfPi = this.calculateLambda(new Instances[] {i1, i2});
        System.out.println("lambda is " + sumOfPi);
        if (sumOfPi >= 0) {
            int j = this.calculateIntersectionSize(i1, i2);
            System.out.println("J is " + j);
            if (j > 0) {
                STDevToReturn = (j - sumOfPi) / Math.sqrt(sumOfPi);
            } else {
                STDevToReturn = Math.abs(sumOfPi - j);
            }
        }
        return STDevToReturn;
    }

}

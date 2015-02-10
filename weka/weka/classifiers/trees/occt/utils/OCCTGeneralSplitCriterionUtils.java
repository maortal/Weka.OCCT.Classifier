package weka.classifiers.trees.occt.utils;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Created by sepetnit on 27/12/14.
 *
 */

public class OCCTGeneralSplitCriterionUtils {

    /**
     * The function counts the number of appearances of some instance in a given set of instances
     *
     * @param i The instance to look for
     * @param instances The set of instances to look in
     *
     * @return The number of appearances of i in instances
     */
    public static int countAppearances(Instance i, Instances instances) {
        int toReturn = 0;
        String iStr = i.toString();
        // Calculate the distance between each pair of instances
        Enumeration instEnum = instances.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance currentInstance = (Instance) instEnum.nextElement();
            toReturn += (currentInstance.toString().equals(iStr)? 1 : 0);
        }
        return toReturn;
    }
}

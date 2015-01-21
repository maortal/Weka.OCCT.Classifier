package split;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.*;

/**
 * Created by sepetnit on 12/20/2014.
 *
 * The Jaccard similarity coefficient, a measure that is commonly used in clustering, measures the
 * similarity between clusters.
 *
 * The similarity between two subsets is computed for each possible splitting attribute as the ratio
 * between the number of records belonging to both sets and the total number of records
 *
 * Records are considered in the intersection only if they are completely identical (all attributes
 * share the same values):
 */
public class OCCTCoarseGrainedJaccardSplitModel extends OCCTSingleAttributeSplitModel {

    static {
        OCCTCoarseGrainedJaccardSplitModel.m_scoresComparator =  new Comparator<Double>() {
            public int compare(Double score1, Double score2) {
                if (Utils.sm(score1, score2)) {
                    return 1;
                } else if (Utils.gr(score1, score2)) {
                    return -1;
                }
                return 0;
            }
        };
    }

    public OCCTCoarseGrainedJaccardSplitModel(Attribute attr, List<Attribute> possibleAttributes) {
        super(attr, possibleAttributes);
    }

    private Set<String> getDistinctInstancesRepresentations(Instances instances) {
        Set<String> union = new HashSet<String>();
        // Calculate the distance between each pair of instances
        Enumeration instancesEnum = instances.enumerateInstances();
        while (instancesEnum.hasMoreElements()) {
            Instance current = (Instance) instancesEnum.nextElement();
            union.add(this.buildAttrValuesString(current));
        }
        return union;
    }

    private int getUnionSize(Instances i1, Instances i2) {
        Set<String> union = new HashSet<String>();
        union.addAll(this.getDistinctInstancesRepresentations(i1));
        union.addAll(this.getDistinctInstancesRepresentations(i2));
        return union.size();
    }

    protected final double calculateSplitScore(Instances i1, Instances i2) {
        // Handle empty case
        if (i1.numInstances() == 0 || i2.numInstances() == 0) {
            return 0;
        }
        int intersectionSize = this.calculateIntersectionSize(i1, i2);
        // For efficiency: Calculate union only if intersection is not empty
        if (intersectionSize > 0) {
            return intersectionSize / (double)(this.getUnionSize(i1, i2));
        }
        return 0;
    }
}

package weka.classifiers.trees.occt.split.models;

import weka.classifiers.trees.occt.split.auxiliary.OCCTSplitModelComparators;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

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

    private static final long serialVersionUID = 5232633291775764495L;

    static {
        m_scoresComparator =
                OCCTSplitModelComparators.LOWEST_SCORE_CHOOSER;
    }

    public OCCTCoarseGrainedJaccardSplitModel(Attribute attr,
                                              List<Attribute> possibleAttributes,
                                              List<Attribute> attributesOfB) {
        super(attr, possibleAttributes, attributesOfB);
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

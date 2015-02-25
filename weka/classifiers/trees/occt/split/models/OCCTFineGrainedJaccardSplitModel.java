package weka.classifiers.trees.occt.split.models;

import weka.classifiers.trees.occt.split.auxiliary.OCCTSplitModelComparators;
import weka.classifiers.trees.occt.utils.Pair;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Created by sepetnit on 17/01/15.
 *
 * The Fine-Grained Jaccard similarity coefficient is capable of identifying partial record matches,
 * as opposed to the Coarse-Grained method, which identifies exact matches only. It not only
 * considers records that are exactly identical, but also checks to what extent each possible pair
 * of records is similar
 */
public class OCCTFineGrainedJaccardSplitModel extends OCCTSingleAttributeSplitModel {

    private static final long serialVersionUID = -1537257016438337444L;

    static {
        OCCTFineGrainedJaccardSplitModel.SCORES_COMPARATOR =
                OCCTSplitModelComparators.LOWEST_SCORE_CHOOSER;
    }

    // Logger will not be serialized!
    private transient Logger logger;

    public OCCTFineGrainedJaccardSplitModel(Attribute splittingAttribute,
                                            List<Attribute> possibleAttributes,
                                            List<Attribute> attributesOfB) {
        super(splittingAttribute, possibleAttributes, attributesOfB);
        // For debugging issues initialize this logger
        this.logger = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    }

    /**
     * This function calculates the intersection between two instances. Which means, all the
     * attributes that have same values and does not contain null
     *
     * @param i1 The first instance
     * @param i2 The second instance
     *
     * @return The set of attributes that contain the same values in both of the instances
     */
    private List<Attribute> calculateIntersection(Instance i1, Instance i2) {
        List<Attribute> intersection = new LinkedList<Attribute>();
        for (Attribute currentAttr : this.m_possibleAttributes) {
            if (!currentAttr.equals(this.m_splittingAttribute)) {
                // It is sufficient to check only if the attribute is missing in the first instance
                if (!i1.isMissing(currentAttr) &&
                        (i1.stringValue(currentAttr).equals(i2.stringValue(currentAttr)))) {
                    intersection.add(currentAttr);
                }
            }
        }
        return intersection;
    }

    /**
     * Calculates the size of the intersection between two instances.
     *
     * {@see calculateIntersection()}
     *
     * @param i1 The first instance
     * @param i2 The second instance
     *
     * @return The size of the intersection
     */
    private int calculateIntersectionSize(Instance i1, Instance i2) {
        return this.calculateIntersection(i1, i2).size();
    }


    /**
     * In this case, union of two records returns all the attributes whose value in either of the
     * records is not missing
     *
     * @param i1 The first instance
     * @param i2 The second instance
     *
     * @return A list of all the attributes which have non-missing values in either of the records
     */
    protected List<Attribute> calculateUnion(Instance i1, Instance i2) {
        List<Attribute> union = new LinkedList<Attribute>();
        for (Attribute currentAttr : this.m_possibleAttributes) {
            if (!currentAttr.equals(this.m_splittingAttribute)) {
                // Assure that the examined attribute did not contain null in either of the records
                if (!i1.isMissing(currentAttr) && !i2.isMissing(currentAttr)) {
                    union.add(currentAttr);
                }
            }
        }
        return union;
    }

    private int calculateUnionSize(Instance i1, Instance i2) {
        return this.calculateUnion(i1, i2).size();
    }


    // TODO: Make this configurable from the GUI
    private static int NUM_OF_CLUSTERS = 100;

    /**
     * Creates new Instances instances with both the sets of instances inside
     */
    private Instances joinInstances(Instances i1, Instances i2) {
        Instances toReturn = new Instances(i1);
        Enumeration i2Enum = i2.enumerateInstances();
        while (i2Enum.hasMoreElements()) {
            Instance currentI2Instance = (Instance)i2Enum.nextElement();
            toReturn.add(currentI2Instance);
        }
        return toReturn;
    }

    /**
     * This class is used in order to calculate the weka.trees.classifiers.occt.split score between the clusters.
     * After the joined instances from T_AB(d1)a and T_AB(d2)a are inserted as an input of the
     * clustering algorithm, we go over them and build an array which contains an instance of this
     * class for each cluster, such that the first element of the pair contains all elements from
     * T_AB(d1)a that were clustered to this cluster and the second element of the pair contains
     * all elements of T_AB(d2)a that were clustered to this cluster
     */
    private class DataForSingleCluster extends Pair<List<Instance>, List<Instance>> {

        private int clusterId;

        public DataForSingleCluster(int clusterId) {
            super(new ArrayList<Instance>(), new ArrayList<Instance>());
            this.clusterId = clusterId;
        }

        public void addInstanceToD1(Instance toAdd) {
            List<Instance> list = super.getFirst();
            list.add(toAdd);
        }

        public void addInstanceToD2(Instance toAdd) {
            List<Instance> list = super.getSecond();
            list.add(toAdd);
        }

        public int getClusterId() {
            return this.clusterId;
        }
    }

    /**
     * Takes the given instances and updates the final data structure according to their clustering
     *
     * @param instances The instances to cluster
     * @param firstOrSecondTable Whether the instances belong to T_AB(d1)a or T_AB(d2)a:
     *                           'true' value means they belong to T_AB(d1)a and 'false'
     *                           means they belong to T_AB(d2)a
     * @param clusterer The clustered (pre-built) which is used in order to cluster the instances
     * @param toUpdate The final data structure to update with the clustered instances
     *
     * @throws Exception In case the clusterer failed for some reason ...
     */
    private void clusterSingleInstances(Instances instances, boolean firstOrSecondTable,
                                        Clusterer clusterer, DataForSingleCluster[] toUpdate)
            throws Exception {
        Enumeration instancesEnum = instances.enumerateInstances();
        while (instancesEnum.hasMoreElements()) {
            Instance currentInstance = (Instance)instancesEnum.nextElement();
            int clusterID = clusterer.clusterInstance(currentInstance);
            DataForSingleCluster currentDataForSingleCluster = toUpdate[clusterID];
            if (firstOrSecondTable) {
                currentDataForSingleCluster.addInstanceToD1(currentInstance);
            } else {
                currentDataForSingleCluster.addInstanceToD2(currentInstance);
            }
        }
    }

    private DataForSingleCluster[] clusterInstances(Instances i1, Instances i2) {
        // This is the array which should be returned (contains single element for each cluster)
        DataForSingleCluster[] toReturn =
                new DataForSingleCluster[OCCTFineGrainedJaccardSplitModel.NUM_OF_CLUSTERS];
        // Make a single instances object in order to run the clustering algorithm with it
        Instances joinedInstances = this.joinInstances(i1, i2);
        // TODO: Maybe allow various clustering algorithms
        SimpleKMeans kMeansClusterer = new SimpleKMeans();
        try {
            // Build the clusterer using the joined instances
            kMeansClusterer.setNumClusters(OCCTFineGrainedJaccardSplitModel.NUM_OF_CLUSTERS);
            kMeansClusterer.buildClusterer(joinedInstances);
            // Initialize the array to return
            for (int i = 0; i < toReturn.length; ++i) {
                toReturn[i] = new DataForSingleCluster(i);
            }
            // Now, go over all the instances and separate them according their clustering result
            this.clusterSingleInstances(i1, true, kMeansClusterer, toReturn);
            this.clusterSingleInstances(i2, false, kMeansClusterer, toReturn);
        } catch (Exception e) {
            // TODO: What should we do in this case?
            e.printStackTrace();
            System.exit(-1);
        }
        return toReturn;
    }

    private double calculateSplitScoreForInstancesPair(Instance i1, Instance i2) {
        int intersectionSize = this.calculateIntersectionSize(i1, i2);
        // No need to calculate union if the size of the intersection is 0
        if (intersectionSize > 0) {
            int unionSize = this.calculateUnionSize(i1, i2);
            return (intersectionSize / (double) unionSize);
        }
        return 0;
    }

    /**
     * This function calculates the weka.trees.classifiers.occt.split score without clustering given two sets of instances
     *
     * @param i1 The first set of instances
     * @param i2 The second set of instances
     *
     * @return A pair which contains the sum of score for each pair of instances participated in the
     *         comparing
     *
     * @throws Exception
     */
    private Pair<Double, Integer> calculateSplitScoreInternal(Instances i1, Instances i2)
            throws Exception {
        // Handle empty case
        double sumOfScores = 0;
        int comparisonsCount = 0;
        if ((i1.numInstances()) > 0 && (i2.numInstances() > 0)) {
            // Calculate the sum of similarities of all possible pairs of records,
            Enumeration inst1Enum = i1.enumerateInstances();
            while (inst1Enum.hasMoreElements()) {
                Instance currentI1 = (Instance) inst1Enum.nextElement();
                Enumeration inst2Enum = i2.enumerateInstances();
                while (inst2Enum.hasMoreElements()) {
                    Instance currentI2 = (Instance) inst2Enum.nextElement();
                    sumOfScores += this.calculateSplitScoreForInstancesPair(currentI1, currentI2);
                    ++comparisonsCount;
                }
            }
        }
        // Return the result - don't calculate here the final score
        return new Pair<Double, Integer>(sumOfScores, comparisonsCount);
    }

    private Pair<Double, Integer> calculateSplitScoreInternal(List<Instance> i1, List<Instance> i2)
            throws Exception {
        // Handle empty case
        double sumOfScores = 0;
        int comparisonsCount = 0;
        if ((i1.size()) > 0 && (i2.size() > 0)) {
            // Calculate the sum of similarities of all possible pairs of records,
            for (Instance currentI1 : i1) {
                for (Instance currentI2 : i2) {
                    sumOfScores += this.calculateSplitScoreForInstancesPair(currentI1, currentI2);
                    ++comparisonsCount;
                }
            }
        }
        // Return the result - don't calculate here the final score
        return new Pair<Double, Integer>(sumOfScores, comparisonsCount);
    }


    protected double calculateSplitScoreUsingClustering(Instances i1, Instances i2)
            throws Exception {
        double totalSumOfScores = 0;
        int totalNumOfComparions = 0;
        // First, perform the clustering
        DataForSingleCluster[] clusteredData = this.clusterInstances(i1, i2);
        // Aggregate the results of the weka.trees.classifiers.occt.split scores calculation for each of the clusters
        for (DataForSingleCluster current : clusteredData) {
            Pair<Double, Integer> internalSplitScoreElements =
                    this.calculateSplitScoreInternal(current.getFirst(), current.getSecond());
            totalSumOfScores += internalSplitScoreElements.getFirst();
            totalNumOfComparions += internalSplitScoreElements.getSecond();
        }
        logger.fine("TOTAL " + totalSumOfScores / (double) totalNumOfComparions);
        return totalSumOfScores / (double)totalNumOfComparions;
    }

    protected double calculateSplitScoreWithoutClustering(Instances i1, Instances i2)
            throws Exception {
        Pair<Double, Integer> internalSplitScoreElements =
                this.calculateSplitScoreInternal(i1, i2);
        double sumOfScores = internalSplitScoreElements.getFirst();
        int comparisonsCount = internalSplitScoreElements.getSecond();
        logger.fine("TOTAL " + sumOfScores / (double) comparisonsCount);
        return sumOfScores / (double)comparisonsCount;
    }

    @Override
    protected double calculateSplitScore(Instances i1, Instances i2) throws Exception {
        // We want to use clustering if the number of instances is too large. Otherwise, if we
        // have less instances than the defined number of clusters, there is no reason to perform
        // clustering - it will be faster to weka.trees.classifiers.occt.split the instances using the standard equation of
        // Fine-Grained Jaccard
        if (i1.numInstances() + i2.numInstances() >=
                OCCTFineGrainedJaccardSplitModel.NUM_OF_CLUSTERS) {
            return this.calculateSplitScoreUsingClustering(i1, i2);
        }
        return this.calculateSplitScoreWithoutClustering(i1, i2);
    }
}

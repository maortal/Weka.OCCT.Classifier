package tree;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import utils.MyStringBuffer;
import utils.Pair;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.core.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 /**
 * Class for handling a tree structure used for
 * classification.
 *
 * @author sepetnit
 * @version $Revision: 1 $
 */
public class OCCTInternalClassifierNode implements Drawable, Serializable,
        CapabilitiesHandler, RevisionHandler {

    /** for serialization */
    static final long serialVersionUID = -973249211542528293L;

    /** References to sons. */
    protected OCCTInternalClassifierNode[] m_sons;

    /** References to parent. */
    protected OCCTInternalClassifierNode m_parent;

    /** True if node is empty. */
    protected boolean m_isEmpty;

    /** True if node is leaf. */
    protected boolean m_isLeaf;

    /** The training instances. */
    protected Instances m_train;

    /** The model selection method. */
    protected OCCTSplitModelSelection m_modelSelectionMethod;

    /** Local model at node. */
    protected ClassifierSplitModel m_localModel;

    /**
     * Constructor
     *
     */
    public OCCTInternalClassifierNode(OCCTSplitModelSelection modelSelectionMethod) {
        this.m_sons = null;
        this.m_parent = null;
        this.m_train = null;
        this.m_isLeaf = false;

        this.m_modelSelectionMethod = modelSelectionMethod;
    }


    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();

        // TODO ...
        result.enable(Capabilities.Capability.NO_CLASS);

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);


        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public int graphType() {
        return 0;
    }

    @Override
    public String graph() throws Exception {
        return null;
    }

    @Override
    public String getRevision() {
        return null;
    }

    /**
     * Cleanup in order to save memory.
     *
     * @param justHeaderInfo The only instances data we want to save are the headers
     */
    public final void cleanup(Instances justHeaderInfo) {
        this.m_train = justHeaderInfo;
        if (!this.m_isLeaf) {
            for (OCCTInternalClassifierNode son : this.m_sons) {
                son.cleanup(justHeaderInfo);
            }
        }
    }

    /**
     * Method for building a pruneable classifier tree.
     *
     * @param instances the data to build the tree from
     * @throws Exception if tree can't be built successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
        // can classifier tree handle the data?
        getCapabilities().testWithFail(instances);
        this.buildTree(instances, false);
        // TODO: Should cleanup?
        cleanup(new Instances(instances, 0));
    }

    /**
     * This method just exists to make program easier to read.
     *
     * @return The splitting model associated with this node
     */
    private ClassifierSplitModel localModel() {

        return this.m_localModel;
    }

    /**
     * This Method just exists to make program easier to read.
     *
     * @return The son node at the given index
     */
    private OCCTInternalClassifierNode son(int index) {

        return m_sons[index];
    }

    /**
     * Help method for computing class probabilities of a given instance.
     *
     * @param classIndex the class index
     * @param instance the instance to compute the probabilities for
     * @param weight the weight to use
     * @return the calculated probability
     *
     * @throws Exception if something goes wrong
     */
    private double getProbabilityForClass(int classIndex, Instance instance, double weight)
            throws Exception {
        // If it is leaf - just calculate the probability accordingly ...
        if (this.m_isLeaf) {
            return weight * this.localModel().classProb(classIndex, instance, -1);
        // Otherwise, go downwards according to the attributes of A ...
        } else {
            int subsetIndex = this.localModel().whichSubset(instance);
            // TODO: Index no found ...
            if (subsetIndex == -1) {
                throw new NotImplementedException();
            } else {
                // In case there are no instances under this son
                if (son(subsetIndex).m_isEmpty) {
                    throw new NotImplementedException();
                } else {
                    return son(subsetIndex).getProbabilityForClass(classIndex, instance, weight);
                }
            }
        }
    }

    /**
     * Returns class probabilities for a weighted instance.
     *
     * @param instance the instance to get the distribution for
     * @return the distribution
     * @throws Exception if something goes wrong
     */
    public final double[] distributionForInstance(Instance instance) throws Exception {
        double [] doubles = new double[instance.numClasses()];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = this.getProbabilityForClass(i, instance, 1);
        }
        return doubles;
    }

    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     *
     * @return the classification
     *
     * @throws Exception if something goes wrong
     */
    public double classifyInstance(Instance instance) throws Exception {
        double maxProb = -1;

        double currentProb;
        int maxIndex = 0;

        // There are two opposite classes: Match/Not match
        for (int j = 0; j < instance.numClasses(); j++) {
            currentProb = this.getProbabilityForClass(j, instance, 1);
            if (Utils.gr(currentProb,maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
        }
        return (double)maxIndex;
    }



    /**
     * Returns a newly created tree.
     *
     * @param data the training data
     *
     * @return the generated tree
     *
     * @throws Exception if something goes wrong
     */
    protected OCCTInternalClassifierNode getNewTree(Instances data) throws Exception {
        // Call copy constructor (since different attributes may be chosen for each child ...)
        OCCTSplitModelSelection newModelSelection =
                new OCCTSplitModelSelection(this.m_modelSelectionMethod,
                        this.m_modelSelectionMethod.getChosenAttribute());
        OCCTInternalClassifierNode newTree = new OCCTInternalClassifierNode(newModelSelection);
        newTree.buildTree(data, false);
        return newTree;
    }

    /**
     * Builds the tree structure.
     *
     * @param instances the data for which the tree structure is to be generated.
     * @param keepData is training data to be kept?
     *
     * @throws Exception if something goes wrong
     */
    public void buildTree(Instances instances, boolean keepData) throws Exception {
        if (keepData) {
            this.m_train = instances;
        }
        // Currently that is not a leaf and there are no sons
        this.m_isLeaf = false;
        this.m_sons = null;
        // Choose the local model on this node (according to the training instances set)
        this.m_localModel = this.m_modelSelectionMethod.selectModel(instances);
        // Continue to perform a split only if more that one subset was extracted
        if (this.m_localModel.numSubsets() > 1) {
            // Perform the actual split
            Instances[] localSplittedTrain = this.m_localModel.split(instances);
                    //((OCCTSingleAttributeSplitModel)this.m_localModel).splitInstances(instances);
            // Initialize sons and continue splitting
            this.m_sons = new OCCTInternalClassifierNode[this.m_localModel.numSubsets()];
            for (int i = 0; i < this.m_sons.length; ++i) {
                this.m_sons[i] = this.getNewTree(localSplittedTrain[i]);
                localSplittedTrain[i] = null;
            }
        } else{
            this.m_isLeaf = true;
            // TODO?
            if (Utils.eq(instances.sumOfWeights(), 0)) {
                this.m_isEmpty = true;
            }
        }
    }

    /**
     * Returns the number of leaves under the current node
     *
     * @return the calculated number of leaves
     */
    public int getLeavesCount() {
        int leavesCount = 0;
        if (this.m_isLeaf) {
            return 1;
        }
        for (OCCTInternalClassifierNode son: this.m_sons) {
            leavesCount += son.getLeavesCount();
        }
        return leavesCount;
    }

    /**
     * Returns number of nodes in tree structure.
     *
     * @return the number of nodes
     */
    public int getNodesCount() {
        int nodesCount = 1;
        if (!this.m_isLeaf) {
            for (OCCTInternalClassifierNode son : this.m_sons) {
                nodesCount += son.getNodesCount();
            }
        }
        return nodesCount;
    }

    private List<Pair<String, OCCTInternalClassifierNode> > getSortedSons() {
        List<Pair<String, OCCTInternalClassifierNode> > toReturn =
                new ArrayList<Pair<String, OCCTInternalClassifierNode>>(this.m_sons.length);
        // Fill the list
        for (int i = 0; i < this.m_sons.length; ++i) {
            toReturn.add(new Pair<String, OCCTInternalClassifierNode>(
                    this.m_localModel.rightSide(i, this.m_train), this.m_sons[i]));
        }
        // Sort the list
        toReturn.sort(new Comparator<Pair<String, OCCTInternalClassifierNode>>() {
            @Override
            public int compare(Pair<String, OCCTInternalClassifierNode> firstPair,
                               Pair<String, OCCTInternalClassifierNode> secondPair) {
                return firstPair.getFirst().compareTo(secondPair.getFirst());
            }
        });
        return toReturn;
    }

    /**
     * A help method for printing node's structure.
     *
     * @param depth the current depth
     * @param text for outputting the structure
     *
     * @throws Exception if something goes wrong
     */
    private void dumpTree(int depth, MyStringBuffer text) throws Exception {
        // Go over all the sons of the node
        List<Pair<String, OCCTInternalClassifierNode> > sons = this.getSortedSons();
        for (Pair<String, OCCTInternalClassifierNode> current : sons) {
            String rightSide = current.getFirst();
            OCCTInternalClassifierNode currentSon = current.getSecond();
            if (!currentSon.m_isEmpty) {
                text.append("\n");
                // Print the relevant number of delimiters (one below the required depth)
                for (int j = 0; j < depth; j++) {
                    text.append("|   ");
                }
                text.append(this.m_localModel.leftSide(this.m_train));
                text.append(rightSide);
                if (currentSon.m_isLeaf) {
                    text.append(": <Leaf>");
                    //text.append(this.m_localModel.dumpLabel(i,m_train));
                } else {
                    currentSon.dumpTree(depth + 1, text);
                }
            }
        }
    }

    /**
     * Prints tree structure.
     *
     * @return the tree structure
     */
    @Override
    public String toString() {
        try {
            MyStringBuffer text = new MyStringBuffer();
            if (this.m_isLeaf) {
                text.append(": <Leaf>");
                // TODO ...
                //text.append(this.m_localModel.dumpLabel(0, m_train));
            } else {
                this.dumpTree(0, text);
            }
            text.append("\n\nNumber of Leaves : \t", this.getLeavesCount(), "\n");
            text.append("\nSize of the tree : \t", this.getNodesCount(), "\n");
            return text.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Can't print classification tree.";
        }
    }
}

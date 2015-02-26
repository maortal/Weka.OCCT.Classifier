package weka.classifiers.trees.occt.tree;

import weka.classifiers.trees.occt.split.auxiliary.OCCTFeatureSelector;
import weka.classifiers.trees.occt.split.models.OCCTSplitModel;
import weka.classifiers.trees.occt.split.pruning.OCCTGeneralPruningMethod;
import weka.classifiers.trees.occt.utils.OCCTStringBuffer;
import weka.classifiers.trees.occt.utils.OCCTPair;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * Class for handling a tree structure used for classification.
 *
 * @author sepetnit
 * @version $Revision: 1 $
 */
public class OCCTInternalClassifierNode implements Drawable, Serializable,
        CapabilitiesHandler, RevisionHandler {

    /**
     * for serialization
     */
    static final long serialVersionUID = -973249211542528293L;

    /**
     * References to sons.
     */
    protected Map<String, OCCTInternalClassifierNode> m_sons;

    /**
     * References to parent.
     */
    protected OCCTInternalClassifierNode m_parent;

    /**
     * True if node is empty.
     */
    protected boolean m_isEmpty;

    /**
     * True if node is leaf.
     */
    protected boolean m_isLeaf;

    /**
     * The id for the node.
     */
    protected int m_id;

    /**
     * The training instances.
     */
    protected Instances m_train;

    /**
     * The model selection method.
     */
    protected OCCTSplitModelSelection m_modelSelectionMethod;

    /**
     * The chosen pruning method, used for pre-pruning of redundant branches
     */
    protected OCCTGeneralPruningMethod m_pruningMethod;

    /**
     * Local model at node.
     */
    protected OCCTSplitModel m_localModel;

    /**
     * Feature selector to execute on the leaf dataset in order to choose the attributes that will
     * be represented.
     */
    protected OCCTFeatureSelector m_featureSelector;


    protected OCCTLeafNode m_leafModel;

    /**
     * Constructor
     */
    public OCCTInternalClassifierNode(OCCTSplitModelSelection modelSelectionMethod,
                                      OCCTGeneralPruningMethod pruningMethod) {
        this.m_sons = null;
        this.m_parent = null;
        this.m_train = null;
        this.m_isLeaf = false;
        this.m_localModel = null;
        this.m_modelSelectionMethod = modelSelectionMethod;
        this.m_pruningMethod = pruningMethod;
        this.m_featureSelector = null;
        // This is not a leaf until specified
        this.m_leafModel = null;
    }

    public OCCTInternalClassifierNode(OCCTSplitModelSelection modelSelectionMethod,
                                  OCCTInternalClassifierNode parent,
                                  OCCTGeneralPruningMethod pruningMethod) {
        this(modelSelectionMethod, pruningMethod);
        this.m_parent = parent;
    }

    /**
     * This constructor assumes parent is null and also pruning method is null
     */
    public OCCTInternalClassifierNode(OCCTSplitModelSelection modelSelectionMethod) {
        this(modelSelectionMethod, null, null);
    }

    /**
     * A constructor of the class, without any pruning method
     *
     * In case no pruning was defined, call the constructor with a null pruning method
     *
     * @param modelSelectionMethod The selection method to chose the best split model for each node
     * @param parent The parent node
     */
    public OCCTInternalClassifierNode(OCCTSplitModelSelection modelSelectionMethod,
                                      OCCTInternalClassifierNode parent) {
        this(modelSelectionMethod, parent, null);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();

        // TODO ...
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NO_CLASS);

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);


        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public int graphType() {
        return Drawable.TREE;
    }

    @Override
    /**
     * Returns graph describing the tree.
     *
     * @throws Exception if something goes wrong
     * @return the tree as graph
     */
    public String graph() throws Exception {
        assignIDs(-1);
        OCCTStringBuffer text = new OCCTStringBuffer();
        text.append("digraph OCCTTree {\n");
        graphTree(text);
        return text.toString() + "}\n";
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
            for (OCCTInternalClassifierNode son : this.m_sons.values()) {
                son.cleanup(justHeaderInfo);
            }
        }
    }

    /**
     * Method for building a prunable classifier tree.
     *
     * @param instances the data to build the tree from
     * @throws Exception if weka.trees.classifiers.occt.split.tree can't be built successfully
     */
    public void buildClassifier(Instances instances) throws Exception {
        // can the classifier handle the data?
        getCapabilities().testWithFail(instances);
        this.buildTree(instances, false);
        // TODO: Should cleanup?
        cleanup(new Instances(instances, 0));
    }

    /**
     * @return The Attribute of the splitting model associated with this node or null if there is
     * still no model
     */
    public Attribute getChosenAttribute() {
        if (this.m_localModel != null) {
            return this.m_localModel.getChosenAttribute();
        }
        return null;
    }

    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     * @return the classification
     * @throws Exception if something goes wrong
     */
    public double classifyInstance(Instance instance) throws Exception {
        /*
        if (this.m_isEmpty) {
            return 0.5;
        }*/
        if (this.m_isLeaf) {
            return this.m_leafModel.classifyInstance(instance);
        }
        Attribute chosen = this.getChosenAttribute();
        OCCTInternalClassifierNode relevantSon = this.m_sons.get(instance.stringValue(chosen));
        return relevantSon.classifyInstance(instance);
    }

    /**
     * Returns a newly created weka.trees.classifiers.occt.split.tree.
     *
     * @param data the training data
     * @return the generated weka.trees.classifiers.occt.split.tree
     * @throws Exception if something goes wrong
     */
    protected OCCTInternalClassifierNode getNewTree(Instances data) throws Exception {
        // Call copy constructor (since different attributes may be chosen for each child ...)
        OCCTSplitModelSelection newModelSelection =
                new OCCTSplitModelSelection(this.m_modelSelectionMethod,
                        this.m_localModel.getChosenAttribute());
        // The pruning method remains the same
        OCCTInternalClassifierNode newTree =
                new OCCTInternalClassifierNode(newModelSelection, this, this.m_pruningMethod);
        newTree.buildTree(data, false);
        return newTree;
    }

    private void buildLeaf(Instances instances, List<Attribute> selectedAttributesOfB)
            throws Exception {
        this.m_isLeaf = true;
        // TODO? Do we really need this?
        if (instances.numInstances() == 0) {
            this.m_isEmpty = true;
        } else {
            // We reach a leaf - thus, models for that leaf should be created
            this.m_leafModel =
                    new OCCTLeafNode(this.m_modelSelectionMethod.getAttributesOfB(),
                                     selectedAttributesOfB);
            this.m_leafModel.buildLeaf(instances);
        }
    }

    private List<Attribute> getCurrentSelectedFeatures() {
        if (this.m_parent == null || this.m_parent.m_featureSelector == null) {
            return null;
        }
        try {
            return this.m_parent.m_featureSelector.getSelectedFeatures();
        } catch (IllegalStateException e) {
            return null;
        }
    }

    /**
     * Builds the tree structure.
     *
     * @param instances the data for which the tree structure is to be generated.
     * @param keepData  is training data to be kept?
     * @throws Exception if something goes wrong
     */
    public void buildTree(Instances instances, boolean keepData) throws Exception {
        if (keepData) {
            this.m_train = instances;
        }
        // Currently that is not a leaf and there are no sons
        this.m_isLeaf = false;
        this.m_sons = null;
        // First, let's check if we should perform pruning here - without continuing to branch
        if (this.m_pruningMethod != null) {
            if (this.m_pruningMethod.shouldPrune(instances)) {
                this.buildLeaf(instances, this.getCurrentSelectedFeatures());
                return;
            }
        }
        // In any other case, create the leafs
        this.m_localModel = this.m_modelSelectionMethod.selectModel(instances);
        // Continue to perform splits only if more that one subset was extracted, otherwise, this
        // is a leaf
        if (this.m_localModel.numSubsets() > 1) {
            Attribute chosenAttribute = this.m_localModel.getChosenAttribute();
            // Select the features for the leaf dataset
            this.m_featureSelector =
                    new OCCTFeatureSelector(chosenAttribute,
                                            this.m_modelSelectionMethod.getAttributesOfB(),
                                            instances);
            // Perform the actual split
            Instances[] localSplittedTrain = this.m_localModel.split(instances);
            //((OCCTSingleAttributeSplitModel)this.m_localModel).splitInstances(instances);
            // Initialize sons and continue splitting
            this.m_sons = new HashMap<String, OCCTInternalClassifierNode>(
                    this.m_localModel.numSubsets());
            for (int i = 0; i < localSplittedTrain.length; ++i) {
                this.m_sons.put(chosenAttribute.value(i), this.getNewTree(localSplittedTrain[i]));
                localSplittedTrain[i] = null;
            }
            Enumeration possibleValues = chosenAttribute.enumerateValues();
            while (possibleValues.hasMoreElements()) {
                String currentValue = (String)possibleValues.nextElement();
                if (!this.m_sons.containsKey(currentValue)) {
                    this.m_sons.put(currentValue, this.getNewTree(new Instances(instances, 0)));
                }
            }
        } else {
            this.buildLeaf(instances, this.getCurrentSelectedFeatures());
        }
    }

    /**
     * Returns the number of leaves under the current node
     *
     * @return the calculated number of leaves
     */
    public int getLeavesCount() {
        int leavesCount = 0;
        if (this.m_isLeaf && !this.m_isEmpty) {
            return 1;
        } else if (this.m_isEmpty) {
            return 0;
        }
        for (OCCTInternalClassifierNode son : this.m_sons.values()) {
            leavesCount += son.getLeavesCount();
        }
        return leavesCount;
    }

    /**
     * Returns number of nodes in tree structure (leafs are not included)
     *
     * @return the number of nodes
     */
    public int getNodesCount() {
        int nodesCount = this.m_isEmpty ? 0 : 1;
        if (!this.m_isLeaf) {
            for (OCCTInternalClassifierNode son : this.m_sons.values()) {
                nodesCount += son.getNodesCount();
            }
        }
        return nodesCount;
    }

    private List<OCCTPair<String, OCCTInternalClassifierNode>> getSortedSons() {
        List<OCCTPair<String, OCCTInternalClassifierNode>> toReturn =
                new ArrayList<OCCTPair<String, OCCTInternalClassifierNode>>(this.m_sons.size());
        // Fill the list
        for (Map.Entry<String, OCCTInternalClassifierNode> currentSon : this.m_sons.entrySet()) {
            toReturn.add(new OCCTPair<String, OCCTInternalClassifierNode>(
                    this.m_localModel.rightSide(currentSon.getKey(), this.m_train),
                    currentSon.getValue()));
        }
        // Sort the list
        toReturn.sort(new Comparator<OCCTPair<String, OCCTInternalClassifierNode>>() {
            @Override
            public int compare(OCCTPair<String, OCCTInternalClassifierNode> firstPair,
                               OCCTPair<String, OCCTInternalClassifierNode> secondPair) {
                return firstPair.getFirst().compareTo(secondPair.getFirst());
            }
        });
        return toReturn;
    }

    /**
     * A help method for printing node's structure.
     *
     * @param depth the current depth
     * @param text  for outputting the structure
     * @throws Exception if something goes wrong
     */
    private void dumpTree(int depth, OCCTStringBuffer text) throws Exception {
        if (this.m_isLeaf) {
            text.append("\n");
            // Print the relevant number of delimiters (one below the required depth)
            for (int j = 0; j < depth; j++) {
                text.append("|   ");
            }
            text.append(this.m_leafModel.toString());
        } else {
            // Go over all the sons of the node
            List<OCCTPair<String, OCCTInternalClassifierNode>> sons = this.getSortedSons();
            for (OCCTPair<String, OCCTInternalClassifierNode> current : sons) {
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
                    text.append(": ");
                    text.append(Utils.doubleToString(this.m_localModel.score(), 3));
                    currentSon.dumpTree(depth + 1, text);
                }
            }
        }
    }

    /**
     * Returns a string representation of the tree
     *
     * @return the tree structure
     */
    @Override
    public String toString() {
        try {
            OCCTStringBuffer text = new OCCTStringBuffer();
            this.dumpTree(0, text);
            text.append("\n\nTree size (total number of nodes):\t", this.getNodesCount(), "\n");
            text.append("\nLeaves (number of predictor nodes)\t", this.getLeavesCount(), "\n");
            return text.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "Can't print classification tree.";
        }
    }

    /**
     * Assigns a unique id to every node in the tree.
     *
     * @param lastID the last ID that was assign
     * @return the new current ID
     */
    public int assignIDs(int lastID) {

        int currLastID = lastID + 1;

        m_id = currLastID;
        if (m_sons != null) {
            for (OCCTInternalClassifierNode current : m_sons.values())
                currLastID = current.assignIDs(currLastID);
        }
        return currLastID;
    }


    /**
     * Help method for printing tree structure as a graph.
     *
     * @param text for outputting the tree
     * @throws Exception if something goes wrong
     */
    private void graphTree(OCCTStringBuffer text) throws Exception {
        if (this.m_isLeaf)
            text.append("N" + m_id + " [label=\""
                    + this.m_leafModel.toString() + "\", shape=box]\n");
        else {
            text.append("N" + m_id + " [label=\""
                    + this.m_localModel.leftSide(this.m_train) + "\"]\n");
            List<OCCTPair<String, OCCTInternalClassifierNode>> sons = this.getSortedSons();
            for (OCCTPair<String, OCCTInternalClassifierNode> current : sons) {
                //TODO: remove repeated edges
                String labelForEdge = current.getFirst();
                OCCTInternalClassifierNode currentSon = current.getSecond();
                if (!currentSon.m_isEmpty) {
                    text.append("N" + m_id + " -> N" + currentSon.m_id
                            + " [label=\"" + labelForEdge + "\"]\n");
                    currentSon.graphTree(text);
                }
            }
        }
    }
}
package weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.classifiers.trees.occt.tree.OCCTInternalClassifierNode;
import weka.classifiers.trees.occt.tree.OCCTSplitModelSelection;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.SingleIndex;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

public class OCCT extends Classifier implements OptionHandler, TechnicalInformationHandler {

	/** for serialization */
	private static final long serialVersionUID = 3850592762853236707L;

	public static String FAKE_MATCH_FIELD_ATTRIBUTE_NAME = "IsMatch";

	/** The decision tree */
	private OCCTInternalClassifierNode m_root;

	// Whether a class attribute should be added to the instances prior to building a classifier
	private boolean shouldAddClassAttribute;

	/** split criteria: Coarse-Grained Jaccard */
	public static final int SPLIT_CGJ = 0;
	/** split criteria: Fine-Grained Jaccard */
	public static final int SPLIT_FGJ = 1;
	/** split criteria: Least-Probable Intersections */
	public static final int SPLIT_LPI = 2;
	/** split criteria: Maximum-Likelihood Estimation */
	public static final int SPLIT_MLE = 3;

	/** split criteria */
	public static final Tag[] TAGS_SPLIT_CRITERIA = {
			new Tag(SPLIT_CGJ, "cgj", "Coarse-Grained Jaccard"),
			new Tag(SPLIT_FGJ, "fgj", "Fine-Grained Jaccard"),
			new Tag(SPLIT_LPI, "lpi", "Least-Probable Intersections"),
			new Tag(SPLIT_MLE, "mle", "Maximum-Likelihood Estimation")
	};

	/** pruning method: No pruning */
	public static final int PRUNING_NO_PRUNING = 0;
	/** pruning method: Least-Probable Intersections */
	public static final int PRUNING_LPI = 1;
	/** pruning method: Maximum-Likelihood Estimation */
	public static final int PRUNING_MLE = 2;

	/** split criteria */
	public static final Tag[] TAGS_PRUNING_METHOD = {
			new Tag(PRUNING_NO_PRUNING, "noprune", "No pruning"),
			new Tag(PRUNING_LPI, "lpi", "Least-Probable Intersections"),
			new Tag(PRUNING_MLE, "mle", "Maximum-Likelihood Estimation"),
	};

	/** The required split criteria (one from the 4 possible) **/
	private int m_SplitCriteria = SPLIT_MLE;
	/** The chosen pruning method (one from the 2 possible) **/
	private int m_PruningMethod = PRUNING_NO_PRUNING;
	/** The first index of the sensitive attributes table (T_B). */
	private SingleIndex m_FirstAttributeIndexOfB = new SingleIndex("last");

	public OCCT(boolean shouldAddClassAttribute) {
		super();
		this.shouldAddClassAttribute = shouldAddClassAttribute;
	}

	public OCCT() {
		// By default, a fake class attribute is added
		this(true);
	}

	/**
	 * Get the first index of the sensitive attributes (table T_B)
	 *
	 * @return the index of the attribute
	 */
	public String getFirstAttributeIndexOfB() {

		return m_FirstAttributeIndexOfB.getSingleIndex();
	}

	/**
	 * Sets first index of the sensitive attributes (table T_B)
	 *
	 * @param attIndex the index of the attribute
	 */
	public void setFirstAttributeIndexOfB(String attIndex) {

		m_FirstAttributeIndexOfB.setSingleIndex(attIndex);
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String setFirstAttributeIndexOfBTipText() {
		return "Sets the first index of the sensitive attributes (it is assumed that the input\n"+
				"consists of T_A x T_B";
	}

	/**
	 * Gets the split criteria to use.
	 *
	 * @return the split criteria.
	 */
	public SelectedTag getSplitCriteria() {
		return new SelectedTag(m_SplitCriteria, TAGS_SPLIT_CRITERIA);
	}

	/**
	 * Sets the split criteria
	 */
	public void setSplitCriteria(SelectedTag value) {
		if (value.getTags() == TAGS_SPLIT_CRITERIA) {
			m_SplitCriteria = value.getSelectedTag().getID();
		}
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String splitCriteriaTipText() {
		return "Set the required split criteria used by the algorithm.";
	}

	/**
	 * Gets the pruning method to use.
	 *
	 * @return pruning method.
	 */
	public SelectedTag getPruningMethod() {
		return new SelectedTag(m_PruningMethod, TAGS_PRUNING_METHOD);
	}

	/**
	 * Sets the pruning method
	 */
	public void setPruningMethod(SelectedTag value) {
		if (value.getTags() == TAGS_PRUNING_METHOD) {
			m_PruningMethod = value.getSelectedTag().getID();
		}
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String pruningMethodTipText() {
		return "The pruning method to use in order to decide which branches should be trimmed.";
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * Valid options are:
	 *
	 *  TODO: Complete this ...
	 *  First attribute of T_B: It is assumed that the input dataset is a cartesian product which
	 *  consists of T_A x T_B. Thus, in order to know which part of the dataset is T_B, only the
	 *  first attribute of T_B is required
     *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();

		newVector.addElement(new Option(
				"\tSets the first attribute index of the B table (default last).",
				"F", 1, "-F <first B's attribute>"));

		newVector.addElement(new Option(
				"\tThe split criteria.\n"+
						"\t(default: " + new SelectedTag(SPLIT_MLE, TAGS_SPLIT_CRITERIA) + ")",
				"S", 1, "-S " + Tag.toOptionList(TAGS_SPLIT_CRITERIA)));

		newVector.addElement(new Option(
				"\tThe pruning method.\n"+
						"\t(default: " + new SelectedTag(PRUNING_NO_PRUNING, TAGS_PRUNING_METHOD) + ")",
				"P", 1, "-P " + Tag.toOptionList(TAGS_PRUNING_METHOD)));

		return newVector.elements();
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();
		Collections.addAll(options, super.getOptions());
		options.add("-F");
		options.add("" + getFirstAttributeIndexOfB());

		options.add("-S");
		options.add("" + getSplitCriteria());

		options.add("-P");
		options.add("" + getPruningMethod());

		return options.toArray(new String[options.size()]);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO: Set here the options
		super.setOptions(options);
		// -F
		String firstAttributeIndexOfBString = Utils.getOption('F', options);
		if (firstAttributeIndexOfBString.length() != 0) {
			this.setFirstAttributeIndexOfB(firstAttributeIndexOfBString);
		} else {
			this.setFirstAttributeIndexOfB("last");
		}
		// -S
		String splitCriteriaString = Utils.getOption('S', options);
		if (splitCriteriaString.length() != 0) {
			this.setSplitCriteria(new SelectedTag(splitCriteriaString, TAGS_SPLIT_CRITERIA));
		} else {
			this.setSplitCriteria(new SelectedTag(SPLIT_MLE, TAGS_SPLIT_CRITERIA));
		}
		// -P
		String pruningMethodString = Utils.getOption('P', options);
		if (pruningMethodString.length() != 0) {
			this.setPruningMethod(new SelectedTag(pruningMethodString, TAGS_PRUNING_METHOD));
		} else {
			this.setPruningMethod(new SelectedTag(PRUNING_NO_PRUNING, TAGS_PRUNING_METHOD));
		}
		// Other?
		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns a string describing the classifier
	 *
	 * @return a description suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for implementing the One-Class Clustering Tree for\n" +
				"One-to-Many Data Linkage.\n\n"
				+ "For more information see: \n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
		result.setValue(TechnicalInformation.Field.AUTHOR, "Ma'ayan Gafny and Asaf Shabtai and " +
				"Lior Rokach and Yuval Elovici");
		result.setValue(TechnicalInformation.Field.YEAR, "2014");
		result.setValue(TechnicalInformation.Field.TITLE, "OCCT: A One-Class Clustering Tree " +
				"for Implementing One-to-Many Data Linkage");
		result.setValue(TechnicalInformation.Field.JOURNAL, "IEEE Transactions on Knowledge and " +
				"Data Engineering (TKDE)");
		result.setValue(TechnicalInformation.Field.VOLUME, "26");
		result.setValue(TechnicalInformation.Field.NUMBER, "3");
		result.setValue(TechnicalInformation.Field.PAGES, "682-697");
		result.setValue(TechnicalInformation.Field.ISSN, "1041-4347");
		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = new OCCTInternalClassifierNode(null).getCapabilities();
		result.setOwner(this);
		return result;
	}

	private Instances addClassAttribute(Instances instances) throws Exception {
		// Add only if doesn't exist
		if (instances.attribute(OCCT.FAKE_MATCH_FIELD_ATTRIBUTE_NAME) == null) {
			Add add = new Add();
			add.setAttributeIndex("last");
			add.setNominalLabels("Match,Unmatch");
			add.setAttributeName(OCCT.FAKE_MATCH_FIELD_ATTRIBUTE_NAME);
			add.setInputFormat(instances);
			instances = Filter.useFilter(instances, add);
		}
		// In any case, the class attribute is the fake one
		instances.setClass(instances.attribute(OCCT.FAKE_MATCH_FIELD_ATTRIBUTE_NAME));
		return instances;
	}

	/**
	 * The function checks the input attribute m_FirstAttributeIndexOfB against the input
	 * instances
	 *
	 * @return The extracted attributes of the table T_B
	 */
	private List<Attribute> checkAndGetAttributesOfB(Instances instances) throws Exception {
		// Initialize the result list
		List<Attribute> attributesOfB = new LinkedList<Attribute>();
		// The last possible index
		int lastIndex = instances.numAttributes() - 1;
		// In case there is a fake class attribute - we shouldn't include it
		if (this.shouldAddClassAttribute) {
			lastIndex--;
		}
		try {
			int index = Integer.parseInt(this.m_FirstAttributeIndexOfB.getSingleIndex());
			// Check too low index
			if (index == 0) {
				throw new IllegalArgumentException("There are no attributes for table A.\n"+
					"Please choose first attribute index for table B, that is greater than 0.");
			}
			// Check too big index
			if (index > lastIndex + 1) {
				throw new IllegalArgumentException("Illegal index was chosen for first attribute\n"+
						"of table B - there are only " + (lastIndex + 1) + " attributes.\n" +
						"Thus, the index must be between 1 and " + (lastIndex + 1) + "\n");
			}
			// Call again - set the real value and raise exception on errors
			// Update the last possible index of attribute (and this actually sets the index)
			this.m_FirstAttributeIndexOfB.setUpper(lastIndex);
			for (int i = this.m_FirstAttributeIndexOfB.getIndex(); i <= lastIndex; ++i) {
				attributesOfB.add(instances.attribute(i));
			}
		} catch (NumberFormatException e) {
			// If any exception occurred, use the generic function
			this.m_FirstAttributeIndexOfB.setUpper(lastIndex);
		}
		return attributesOfB;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances the data to train the classifier with
	 * @throws Exception if classifier can't be built successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		Instances instancesWithClass;
		// can classifier handle the data?
		this.getCapabilities().testWithFail(instances);
		// In case there is no class attribute - let's add it manually before starting working with
		// the instances
		if (this.shouldAddClassAttribute) {
			instancesWithClass = this.addClassAttribute(instances);
		} else {
			instancesWithClass = instances;
		}
		// Let's assure the attribute index is valid
		List<Attribute> attributesOfB = this.checkAndGetAttributesOfB(instances);

		String splitSelectionMethodName = getSplitCriteria().getSelectedTag().getReadable();

		OCCTSplitModelSelection modelSelection =
				new OCCTSplitModelSelection(splitSelectionMethodName, instancesWithClass,
						attributesOfB,
						instancesWithClass.classIndex() != -1?
								instancesWithClass.classAttribute() : null);
		this.m_root = new OCCTInternalClassifierNode(modelSelection);
		// Now, build the tree using the instances
		this.m_root.buildClassifier(instancesWithClass);
		// Finally, perform a cleanup to save memory
		modelSelection.cleanup();
	}

	/**
	 * Classifies an instance.
	 *
	 * @param instance the instance to classify
	 * @return the classification for the instance
	 * @throws Exception if instance can't be classified successfully
	 */
	public final double classifyInstance(Instance instance) throws Exception {
		System.out.println("Called classifier ... with " + instance.classIndex());
		System.out.println("Called classifier ... with " + instance);
		double value = this.m_root.classifyInstance(instance);
		System.out.println("Value is " + value);
		return value > 0.5? 1: 0;
	}

	/**
	 * Returns class probabilities for an instance.
	 *
	 * @param instance the instance to calculate the class probabilities for
	 * @return the class probabilities
	 * @throws Exception if distribution can't be computed successfully
	 */
	public final double[] distributionForInstance(Instance instance) throws Exception {
		double dist[] = new double[2];
		double firstValue = this.classifyInstance(instance);
		dist[0] = firstValue;
		dist[1] = 1 - dist[0];
		return dist;
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	@Override
	public String toString() {
		if (this.m_root == null) {
			return "No classifier built";
		}
		return "OCCT (One-Class Clustering Tree)\n------------------\n" + this.m_root.toString();
	}

	/**
	 * Returns the revision string.
	 *
	 * @return		the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1.9 $");
	}
}

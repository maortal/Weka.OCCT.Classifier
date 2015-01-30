package tree;

import java.util.*;

import weka.classifiers.Classifier;
import weka.core.*;

public class OCCT extends Classifier implements OptionHandler, TechnicalInformationHandler {

	/** for serialization */
	private static final long serialVersionUID = 3850592762853236707L;

	/** The decision tree */
	private OCCTInternalClassifierNode m_root;


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

	/**
	 * Returns an instance of a TechnicalInformation object, containing
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 *
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
		result.setValue(TechnicalInformation.Field.AUTHOR, "Ma’ayan Gafny and Asaf Shabtai and " +
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
	 * Generates the classifier.
	 *
	 * @param instances the data to train the classifier with
	 * @throws Exception if classifier can't be built successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// can classifier handle the data?
		this.getCapabilities().testWithFail(instances);

		// TODO: Should be parameter ...
		List<Attribute> attributesOfB = Arrays.asList(
				instances.attribute("City"),
				instances.attribute("CustomerTypeDesc"));

		OCCTSplitModelSelection modelSelection =
				//new OCCTSplitModelSelection("CoarseGrainedJaccard", instances, attributesOfB);
				new OCCTSplitModelSelection("FineGrainedJaccard", instances, attributesOfB);
				//new OCCTSplitModelSelection("LeastProbableIntersections", instances, attributesOfB);
				//new OCCTSplitModelSelection("MaximumLikelihoodEstimation", instances, attributesOfB);
		this.m_root = new OCCTInternalClassifierNode(modelSelection);
		// Now, build the tree.OCCT tree using the instances
		this.m_root.buildClassifier(instances);
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
	public double classifyInstance(Instance instance) throws Exception {
		return this.m_root.classifyInstance(instance);
	}

	/**
	 * Returns class probabilities for an instance.
	 *
	 * @param instance the instance to calculate the class probabilities for
	 * @return the class probabilities
	 * @throws Exception if distribution can't be computed successfully
	 */
	public final double[] distributionForInstance(Instance instance) throws Exception {
		return this.m_root.distributionForInstance(instance);
	}

	/**
	 * Returns an enumeration describing the available options.
	 *	 
	 * Valid options are:
	 * <p>
	 * 
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
	    Vector<Option> newVector = new Vector<Option>();
	    
	    // TODO: Add here the relevant options
	    newVector.addAll(Collections.list(super.listOptions()));

	    return newVector.elements();
	}
	
	/**
	 * Parses a given list of options.
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO: Set here the options
	    super.setOptions(options);

	    Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		
	    Vector<String> options = new Vector<String>();
	    Collections.addAll(options, super.getOptions());

	    return (String[])options.toArray();

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

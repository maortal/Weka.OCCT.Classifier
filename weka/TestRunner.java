import weka.classifiers.trees.OCCT;
import weka.core.Instances;
import weka.core.SelectedTag;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class TestRunner {
	public TestRunner() {
		System.out.println("Initializing TestRunner");
	}

	/**
	 * Main method for testing this class
	 *
	 * @param argv the commandline options
	 */
	public static void main(String [] argv) throws IOException {
		// load data
		BufferedReader reader = new BufferedReader(
				new FileReader(new File(argv[0], "database_misuse_example_1.arff")));
		Instances data = new Instances(reader);
		reader.close();

		// Setting class attribute
		/// data.setClassIndex(data.numAttributes() - 1);

		// Train OCCT (don't add any fake class attribute)
		OCCT occt = new OCCT(false);
		occt.setSplitCriteria(new SelectedTag(OCCT.SPLIT_MLE, OCCT.TAGS_SPLIT_CRITERIA));
		// Make the attributes of B be: "City", "CustomerTypeDesc" (+1 since indexes are 1-based)
		occt.setFirstAttributeIndexOfB((data.attribute("City").index() + 1) + "");

		try {
			occt.buildClassifier(data);
			System.out.println(occt.toString());

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
}

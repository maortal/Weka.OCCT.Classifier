package weka.classifiers.trees.occt.split.general;

import weka.classifiers.trees.occt.split.pruning.OCCTGeneralPruningMethod;
import weka.classifiers.trees.occt.split.pruning.OCCTLeastProbableIntersectionsPruning;
import weka.classifiers.trees.occt.split.pruning.OCCTMaximumLikelihoodEstimationPruning;
import weka.core.Attribute;
import weka.core.Instances;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by sepetnit on 14/02/15.
 *
 */
public class OCCTPruningMethodFactory {

    private static Map<String, Class<? extends OCCTGeneralPruningMethod>> VALID_PRUNING_METHODS =
            new HashMap<String, Class<? extends OCCTGeneralPruningMethod>>();

    private static String[] SUBSTRINGS_TO_REMOVE = new String[]{"occt", "pruning", "-", " "};

    public static String extractPruningMethodName(
            Class<? extends OCCTGeneralPruningMethod> inputClass) {
        return OCCTPruningMethodFactory.extractPruningMethodName(inputClass.getSimpleName());
    }

    public static String extractPruningMethodName(String base) {
        String simpleName = base.toLowerCase();
        for (String toRemove : OCCTPruningMethodFactory.SUBSTRINGS_TO_REMOVE) {
            simpleName = simpleName.replace(toRemove, "");
        }
        return simpleName;
    }

    private static void addClass(Class<? extends OCCTGeneralPruningMethod> toAdd) {
        String key = OCCTPruningMethodFactory.extractPruningMethodName(toAdd.getSimpleName());
        OCCTPruningMethodFactory.VALID_PRUNING_METHODS.put(key, toAdd);
    }

    static {
        // Add an empty pruning method (no pruning)
        // TODO: Maybe unify this string with strings in OCCT ...
        OCCTPruningMethodFactory.VALID_PRUNING_METHODS.put("nopruning", null);
        OCCTPruningMethodFactory.addClass(OCCTLeastProbableIntersectionsPruning.class);
        OCCTPruningMethodFactory.addClass(OCCTMaximumLikelihoodEstimationPruning.class);
    }

    private static Class<? extends OCCTGeneralPruningMethod> getClass(String pruningMethod) {
        String nameToLookFor = OCCTPruningMethodFactory.extractPruningMethodName(pruningMethod);
        return OCCTPruningMethodFactory.VALID_PRUNING_METHODS.get(nameToLookFor);
    }

    public static boolean isValidPruningMethodType(String criterion) {
        String nameToLookFor = OCCTPruningMethodFactory.extractPruningMethodName(criterion);
        System.out.println("looking for " + nameToLookFor);
        return OCCTPruningMethodFactory.VALID_PRUNING_METHODS.containsKey(nameToLookFor);
    }

    /**
     * The main method of the factory - instantiates a pruner according to the required method name
     *
     * @param pruningMethod The name of the pruning method
     * @param allData The full data
     * @param attributesOfB The attributes of Table B
     *
     * @return The created and instantiated pruning method
     */
    public static OCCTGeneralPruningMethod getPruner(String pruningMethod,
                                                     double pruningThreshold,
                                                     Instances allData,
                                                     List<Attribute> attributesOfB) {
        Class<? extends OCCTGeneralPruningMethod> pruningMethodClass =
                OCCTPruningMethodFactory.getClass(pruningMethod);
        if (pruningMethodClass != null) {
            try {
                // Get the relevant constructor
                Constructor<?> constructor = pruningMethodClass.getConstructor(
                        Instances.class, List.class, double.class);
                // Create the pruner
                return (OCCTGeneralPruningMethod)
                        constructor.newInstance(allData, attributesOfB, pruningThreshold);
            } catch (Exception e) {
                throw new IllegalArgumentException("No such pruner: " + pruningMethod);
            }
        }
        return null;
    }
}
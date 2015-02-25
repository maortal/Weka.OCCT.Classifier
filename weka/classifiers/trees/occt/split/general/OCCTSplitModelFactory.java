package weka.classifiers.trees.occt.split.general;

import weka.classifiers.trees.occt.split.models.*;
import weka.core.Attribute;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.*;

/**
 * Created by sepetnit on 02/01/15.
 *
 * Responsible of returning an instance of a weka.trees.classifiers.occt.split criterion according to the requirement
 *
 */
public class OCCTSplitModelFactory {

    private static Map<String, Class<? extends OCCTSingleAttributeSplitModel>> VALID_SPLIT_MODELS =
            new HashMap<String, Class<? extends OCCTSingleAttributeSplitModel>>();

    private static String[] SUBSTRINGS_TO_REMOVE = new String[] {"occt", "splitmodel", "-", " "};

    public static String extractModelName(
            Class<? extends OCCTSingleAttributeSplitModel> inputClass) {
        return OCCTSplitModelFactory.extractModelName(inputClass.getSimpleName());
    }

    public static String extractModelName(String base) {
        String simpleName = base.toLowerCase();
        for (String toRemove : OCCTSplitModelFactory.SUBSTRINGS_TO_REMOVE) {
            simpleName = simpleName.replace(toRemove, "");
        }
        return simpleName;
    }

    private static void addClass(Class<? extends OCCTSingleAttributeSplitModel> toAdd) {
        String key = OCCTSplitModelFactory.extractModelName(toAdd.getSimpleName());
        OCCTSplitModelFactory.VALID_SPLIT_MODELS.put(key, toAdd);
    }

    static {
        OCCTSplitModelFactory.addClass(OCCTCoarseGrainedJaccardSplitModel.class);
        OCCTSplitModelFactory.addClass(OCCTFineGrainedJaccardSplitModel.class);
        OCCTSplitModelFactory.addClass(OCCTLeastProbableIntersectionsSplitModel.class);
        OCCTSplitModelFactory.addClass(OCCTMaximumLikelihoodEstimationSplitModel.class);
    }

    private static Class<? extends OCCTSingleAttributeSplitModel> getClass(String criterion) {
        String nameToLookFor = OCCTSplitModelFactory.extractModelName(criterion);
        return OCCTSplitModelFactory.VALID_SPLIT_MODELS.get(nameToLookFor);
    }

    public static boolean isValidSplitModelType(String criterion) {
        String nameToLookFor = OCCTSplitModelFactory.extractModelName(criterion);
        return OCCTSplitModelFactory.VALID_SPLIT_MODELS.containsKey(nameToLookFor);
    }

    //use getShape method to get object of type shape
    public static OCCTSingleAttributeSplitModel getSplitModel(String criterion,
                                                              Attribute attr,
                                                              List<Attribute> possibleAttributes,
                                                              List<Attribute> attributesOfB) {
        Class<? extends  OCCTSingleAttributeSplitModel> splitModelClass =
                OCCTSplitModelFactory.getClass(criterion);
        if (splitModelClass != null) {
            Constructor<?>[] constructors = splitModelClass.getConstructors();
            for (Constructor<?> constructor : constructors) {
                try {
                    return (OCCTSingleAttributeSplitModel) constructor.newInstance(attr,
                            possibleAttributes, attributesOfB);
                // TODO! TODO! TODO! TODO! TODO!
                } catch (InvocationTargetException e) {
                    e.printStackTrace();
                } catch (InstantiationException e) {
                    e.printStackTrace();
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }
        return null;
    }

    /**
     * Returns the comparator of split scores which is implemented for the input split method
     *
     * @param criterion The name of the split method to retrieve the comparator for
     *
     * @note This method disables the 'unchecked' inspection since the SCORES_COMPARATOR should be
     *       always a Comparator of doubles
     */
    @SuppressWarnings("unchecked")
    public static Comparator<Double> getSplitModelComparator(String criterion) {
        Comparator<Double> ret = null;
        Class<? extends OCCTSingleAttributeSplitModel> splitModelClass =
                OCCTSplitModelFactory.getClass(criterion);
        if (splitModelClass != null) {
            try {
                ret = (Comparator<Double>)splitModelClass.getField("SCORES_COMPARATOR").get(null);
            // TODO! TODO! TODO! TODO!
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (NoSuchFieldException e) {
                e.printStackTrace();
            }
        }
        return ret;
    }
}

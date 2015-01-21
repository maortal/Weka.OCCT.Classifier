package split;

import weka.core.Attribute;

import java.util.*;

/**
 * Created by sepetnit on 02/01/15.
 *
 * Responsible of returning an instance of a split criterion according to the requirement
 *
 */
public class OCCTSplitModelFactory {

    private static Set<Class<? extends OCCTSingleAttributeSplitModel>> validSplitModels  =
            new HashSet<Class<? extends OCCTSingleAttributeSplitModel>>();

    static {
        validSplitModels.add(OCCTCoarseGrainedJaccardSplitModel.class);
        validSplitModels.add(OCCTLeastProbableIntersectionsSplitModel.class);
    }

    /*
    private static Set<Attribute> attributesOfB = new HashSet<Attribute>();
    public static void setAttributesOfB(Set<Attribute> attributes) {
        OCCTSplitModelFactory.attributesOfB = attributes;
    }
    */

    public static boolean isValidSplitModelType(String criterion) {
        for (Class<? extends OCCTSingleAttributeSplitModel> possibleModelType :
                OCCTSplitModelFactory.validSplitModels) {
            String possibleModelName = possibleModelType.getSimpleName();
            String possibleModelNameLower = possibleModelName.toLowerCase();
            String finalPossibleModelName = possibleModelNameLower.replace("OCCT", "");

            if (finalPossibleModelName.contains(criterion.toLowerCase()))  {
                return true;
            }
        }
        return false;
    }

    //use getShape method to get object of type shape
    public static OCCTSingleAttributeSplitModel getSplitModel(String criterion, Attribute attr,
                                                              List<Attribute> possibleAttributes) {
        if (criterion == null){
            return null;
        }
        if(criterion.equalsIgnoreCase("CoarseGrainedJaccard")) {
            return new OCCTCoarseGrainedJaccardSplitModel(attr, possibleAttributes);
        } else if(criterion.equalsIgnoreCase("LeastProbableIntersections")) {
            return new OCCTLeastProbableIntersectionsSplitModel(attr, possibleAttributes);
        }
        return null;
    }

    public static Comparator<Double> getSplitModelComparator(String criterion) {
        if (criterion == null){
            return null;
        }
        if (criterion.equalsIgnoreCase("CoarseGrainedJaccard")) {
            return OCCTCoarseGrainedJaccardSplitModel.m_scoresComparator;
        } else if(criterion.equalsIgnoreCase("LeastProbableIntersections")) {
            return OCCTLeastProbableIntersectionsSplitModel.m_scoresComparator;
        }
        return null;
    }
}

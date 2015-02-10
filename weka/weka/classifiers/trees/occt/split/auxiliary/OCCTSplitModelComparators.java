package weka.classifiers.trees.occt.split.auxiliary;

import weka.core.Utils;

import java.util.Comparator;

/**
 * Created by sepetnit on 16/01/15.
 *
 * Here general comparators to use by Split Models are defined.
 *
 * Of course, each Split Model can pre-define its own comparator, instead of using comparators
 * defined here
 */
public final class OCCTSplitModelComparators {

    /**
     * This comparator chooses the lowest score among the two provided scores
     */
    public static Comparator<Double> LOWEST_SCORE_CHOOSER =  new Comparator<Double>() {
        public int compare(Double score1, Double score2) {
            if (Utils.sm(score1, score2)) {
                return 1;
            } else if (Utils.sm(score2, score1)) {
                return -1;
            }
            return 0;
        }
    };

    /**
     * This comparator chooses the highest score among the two provided scores
     */
    public static Comparator<Double> HIGHEST_SCORE_CHOOSER =  new Comparator<Double>() {
        public int compare(Double score1, Double score2) {
            if (Utils.gr(score1, score2)) {
                return 1;
            } else if (Utils.gr(score2, score1)) {
                return -1;
            }
            return 0;
        }
    };

}

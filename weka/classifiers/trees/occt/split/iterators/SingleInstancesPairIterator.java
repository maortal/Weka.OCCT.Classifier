package weka.classifiers.trees.occt.split.iterators;

import weka.classifiers.trees.occt.utils.OCCTPair;
import weka.core.Instances;

public class SingleInstancesPairIterator extends GeneralInstancesIterator {

    public SingleInstancesPairIterator(Instances allInstances, Instances[] splittedInstances) {
        super(allInstances, splittedInstances);
    }

    /**
     * In this case all weights are equal (according to the subsets count)
     */
    protected double calculateCurrentWeight() {
        return 1.0;
    }

    public boolean _hasNext() {
        return this.currentInstancesIndex < this.splittedInstances.length;
    }

    public OCCTPair<OCCTPair<Instances, Instances>, Double> _next() {
        OCCTPair<Instances, Instances> instancesPair = new OCCTPair<Instances, Instances>(
                this.splittedInstances[this.currentInstancesIndex], null);
        return new OCCTPair<OCCTPair<Instances, Instances>, Double>(instancesPair,
                this.calculateCurrentWeight());
    }
}

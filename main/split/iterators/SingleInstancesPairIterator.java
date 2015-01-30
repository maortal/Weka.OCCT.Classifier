package split.iterators;

import utils.Pair;
import weka.core.Instances;

public class SingleInstancesPairIterator extends GeneralInstancesIterator {

    public SingleInstancesPairIterator(Instances allInstances, Instances[] splittedInstances) {
        super(allInstances, splittedInstances);
    }

    /**
     * In this case all weights are equal (according to the subsets count)
     */
    protected double calculateCurrentWeight() {
        return 1.0 / this.splittedInstances.length;
    }

    public boolean _hasNext() {
        System.out.println(this.currentInstancesIndex);
        System.out.println(this.splittedInstances.length);
        return this.currentInstancesIndex < this.splittedInstances.length;
    }

    public Pair<Pair<Instances, Instances>, Double> _next() {
        Pair<Instances, Instances> instancesPair = new Pair<Instances, Instances>(
                this.splittedInstances[this.currentInstancesIndex], null);
        return new Pair<Pair<Instances, Instances>, Double>(instancesPair,
                this.calculateCurrentWeight());
    }
}

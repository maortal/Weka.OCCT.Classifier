package weka.classifiers.trees.occt.split.iterators;

import weka.classifiers.trees.occt.utils.Pair;
import weka.core.Instances;

import java.util.NoSuchElementException;

/**
 * Created by sepetnit on 16/01/15.
 *
 */
public abstract class GeneralInstancesIterator {
    protected Instances allInstances;
    protected Instances[] splittedInstances;
    protected int currentInstancesIndex;

    public GeneralInstancesIterator(Instances allInstances, Instances[] splittedInstances) {
        this.allInstances = allInstances;
        this.splittedInstances = splittedInstances;
        this.currentInstancesIndex = 0;
    }

    /**
     * The weight of this pair is proportional to the size of the subsets to
     * ensure that the resulted splitting value of an attribute will be influenced mainly by
     * records having more dominant values and not by esoteric ones
     */
    protected double calculateCurrentWeight() {
        return (this.splittedInstances[this.currentInstancesIndex].numInstances() /
                (double) this.allInstances.numInstances());
    }

    protected abstract boolean _hasNext();

    public boolean hasNext() {
        // We don't want to treat empty instances groups
        while (this.currentInstancesIndex < this.splittedInstances.length &&
                (this.splittedInstances[this.currentInstancesIndex].numInstances() == 0)) {
            ++this.currentInstancesIndex;
        }
        return this._hasNext();
    }

    protected abstract Pair<Pair<Instances, Instances>, Double> _next();

    public Pair<Pair<Instances, Instances>, Double> next() {
        // Assure there is next element
        if (!this.hasNext()) {
            throw new NoSuchElementException();
        }
        Pair<Pair<Instances, Instances>, Double> toReturn = this._next();
        ++this.currentInstancesIndex;
        return toReturn;
    }
}

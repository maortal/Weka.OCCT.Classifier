package weka.classifiers.trees.occt.split.iterators;

import weka.classifiers.trees.occt.utils.OCCTPair;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

public class PairedInstancesIterator extends GeneralInstancesIterator {

    public PairedInstancesIterator(Instances allInstances, Instances[] splittedInstances) {
        super(allInstances, splittedInstances);
    }

    public boolean _hasNext() {
        // Treat boundary conditions
        if (this.splittedInstances.length <= 1) {
            return false;
        // In case of an array of size 2 - we compare only the first element to the second
        } else if (this.splittedInstances.length == 2) {
            return currentInstancesIndex ==  0;
        }
        return this.currentInstancesIndex < this.splittedInstances.length;
    }

    public OCCTPair<OCCTPair<Instances, Instances>, Double> _next() {
        OCCTPair<Instances, Instances> instancesPair;
        OCCTPair<OCCTPair<Instances, Instances>, Double> toReturn;
        // Treat a special case of only two elements in the array
        if (this.splittedInstances.length == 2) {
            instancesPair = new OCCTPair<Instances, Instances>(this.splittedInstances[0],
                    this.splittedInstances[1]);
            toReturn = new OCCTPair<OCCTPair<Instances, Instances>, Double>(instancesPair, 1.0);
        } else {
            // i1 contains all instances with ai == d and i2 contains all instances with ai != d
            Instances i1 = new Instances(this.splittedInstances[this.currentInstancesIndex]);
            Instances i2 = new Instances(this.allInstances, this.allInstances.numInstances() -
                    i1.numInstances());
            for (int j = 0; j < this.splittedInstances.length; ++j) {
                if (j != this.currentInstancesIndex) {
                    Enumeration currentSplittedEnum = this.splittedInstances[j].enumerateInstances();
                    while (currentSplittedEnum.hasMoreElements()) {
                        Instance current = (Instance) currentSplittedEnum.nextElement();
                        i2.add(current);
                    }
                }
            }
            instancesPair = new OCCTPair<Instances, Instances>(i1, i2);
            toReturn = new OCCTPair<OCCTPair<Instances, Instances>, Double>(instancesPair,
                    this.calculateCurrentWeight());
        }
        return toReturn;
    }
}

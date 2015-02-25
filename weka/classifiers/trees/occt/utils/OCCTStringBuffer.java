package weka.classifiers.trees.occt.utils;

import java.io.Serializable;

/**
 * Created by sepetnit on 03/01/15.
 *
 */
public class OCCTStringBuffer implements Serializable {
    private static final long serialVersionUID = -4857228316948131004L;

    private StringBuffer m_internalStringBuffer;

    public OCCTStringBuffer() {
        this.m_internalStringBuffer = new StringBuffer();
    }

    public synchronized OCCTStringBuffer append(Object... args) {
        for (Object arg : args) {
            this.m_internalStringBuffer.append(arg);
        }
        return this;
    }

    public synchronized String toString() {
        return this.m_internalStringBuffer.toString();
    }
}

package utils;

/**
 * Created by sepetnit on 03/01/15.
 *
 */
public class MyStringBuffer {
    private StringBuffer m_internalStringBuffer;

    public MyStringBuffer() {
        this.m_internalStringBuffer = new StringBuffer();
    }

    public synchronized MyStringBuffer append(Object... args) {
        for (Object arg : args) {
            this.m_internalStringBuffer.append(arg);
        }
        return this;
    }

    public synchronized String toString() {
        return this.m_internalStringBuffer.toString();
    }
}

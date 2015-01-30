package utils;

import java.io.Serializable;

/**
 * Generic pair
 * 
 * @author Amir Harel
 */
public class Pair<A, B> implements Serializable {
	private static final long serialVersionUID = 4241613232566784963L;

	private A first;
	private B second;

	public Pair(A first, B second) {
		this.first = first;
		this.second = second;
	}

	public A getFirst() {
		return this.first;
	}

	public B getSecond() {
		return this.second;
	}

	public void setFirst(A first) {
		this.first = first;
	}

	public void setSecond(B second) {
		this.second = second;
	}
	
	@Override
	public boolean equals(Object o) {
		if(o != null && o instanceof Pair<?,?>) {
			Pair<A, B> other;
			try {
				other = (Pair<A,B>)o;
			} catch (ClassCastException e) {
				return false;
			} 
			return other.getFirst().equals(this.first) && other.getSecond().equals(this.second);
		}
		return false;
	}

	@Override
	public String toString() {
		return "<" + this.first + "," + this.second + ">";
	}

}

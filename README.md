# OCCT Classifier implementation for Weka


####OCCT: A One-Class Clustering Tree for Implementing One-to-Many Data Linkage
######Abstract

One-to-many data linkage is an essential task in many domains, yet only a handful of prior publications have addressed this
issue. Furthermore, while traditionally data linkage is performed among entities of the same type, it is extremely necessary to develop
linkage techniques that link between matching entities of different types as well. In this paper, we propose a new one-to-many data
linkage method that links between entities of different natures. The proposed method is based on a one-class clustering weka.trees.classifiers.occt.split.tree (OCCT)
that characterizes the entities that should be linked together. The weka.trees.classifiers.occt.split.tree is built such that it is easy to understand and transform into
association rules, i.e., the inner nodes consist only of features describing the first set of entities, while the leaves of the weka.trees.classifiers.occt.split.tree represent
features of their matching entities from the second data set. We propose four splitting criteria and two different pruning methods which
can be used for inducing the OCCT. The method was evaluated using data sets from three different domains. The results affirm the
effectiveness of the proposed method and show that the OCCT yields better performance in terms of precision and recall (in most
cases it is statistically significant) when compared to a C4.5 decision weka.trees.classifiers.occt.split.tree-based linkage method.

Ma’ayan Dror, Asaf Shabtai, Lior Rokach, and Yuval Elovici

[published in IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6427741)

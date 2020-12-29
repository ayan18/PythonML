#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import graphviz
dot_data = tree.export_graphviz(classification_tree, out_file=None,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")
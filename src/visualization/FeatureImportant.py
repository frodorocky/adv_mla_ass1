#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class FeatureImportance:
    @staticmethod
    def compute_and_plot(model, feature_names):
        importances = model.named_steps['classifier'].feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(20, 10))
        plt.bar(np.arange(len(importances)), importances[sorted_indices], align='center')
        plt.xticks(np.arange(len(importances)), np.array(feature_names)[sorted_indices], rotation=90)
        plt.xlabel('Feature Names')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importances in LightGBM')
        plt.show()
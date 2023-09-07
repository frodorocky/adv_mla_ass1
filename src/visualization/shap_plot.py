#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shap

def generate_shap_summary(model, data, feature_names=None, model_type="tree", meta=False):
    """
    Generate SHAP summary plot for the given model and data.
    
    Parameters:
    - model: Trained model for which SHAP values need to be calculated.
    - data: Preprocessed data to be passed for SHAP values calculation.
    - feature_names: Names of the features in the data.
    - model_type: Type of the model. Supports "tree" or "linear".
    - meta: Boolean indicating if the model is a meta model.
    
    Returns:
    - SHAP summary plot.
    """
    
    # Determine explainer based on model type
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        # If binary classification, take the second array of SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, data)
        shap_values = explainer.shap_values(data)
    else:
        raise ValueError("Unsupported model type!")
    
    # Plot the SHAP values
    if meta:
        shap.summary_plot(shap_values, data, feature_names=['LGBM_Predictions', 'RF_Predictions'])
    else:
        shap.summary_plot(shap_values, data, feature_names=feature_names)

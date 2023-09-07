#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataProcessor:
    def load_and_preprocess(filepath):
        df = pd.read_csv(filepath)
        df = df.set_index(['player_id'])
        return df


    def separate_target_variable(df, target_col):
        y = df.pop(target_col)
        return df, y


    def get_feature_types(df):
        num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        return num_cols, cat_cols


    def categorical_pipeline():
        def convert_to_string(X):
            return X.astype(str)

        stringify_transformer = FunctionTransformer(func=convert_to_string, validate=False)
        categorical_transformer = Pipeline(steps=[
            ('stringify', stringify_transformer),
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        return categorical_transformer

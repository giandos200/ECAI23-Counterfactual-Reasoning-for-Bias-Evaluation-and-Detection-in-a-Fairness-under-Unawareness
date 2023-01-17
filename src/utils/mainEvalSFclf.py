from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


SFgridDict = {'Adult-debiased':{
    'gender':{
        'RF':RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=200,
                       random_state=42),
        'MLP':MLPClassifier(alpha=0.05, early_stopping=True, hidden_layer_sizes=(32, 64),
              learning_rate='adaptive', max_iter=120, random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6,
              enable_categorical=False, eval_metric='logloss', gamma=0.01,
              gpu_id=-1, importance_type=None, interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=3,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=50, n_jobs=12, num_parallel_tree=1, predictor='auto',
              random_state=42, reg_alpha=0.1, reg_lambda=1,
              scale_pos_weight=0.4813641988789401, seed=42, subsample=1.0,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1)
},
    'marital-status':{
        'RF':RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=50,
                       random_state=42),
        'MLP':MLPClassifier(activation='tanh', alpha=0.05, early_stopping=True,
              hidden_layer_sizes=(64,), learning_rate='adaptive', max_iter=120, random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6,
              enable_categorical=False, eval_metric='logloss', gamma=0.1,
              gpu_id=-1, importance_type=None, interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=12, num_parallel_tree=1,
              predictor='auto', random_state=42, reg_alpha=0.01, reg_lambda=1,
              scale_pos_weight=1.0898074454428754, seed=42, subsample=0.8,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1)}},
'Adult':{
    'gender':{
        'RF':RandomForestClassifier(bootstrap=False, max_depth=20, max_features='auto',
                       min_samples_leaf=2, n_estimators=200, random_state=42),
        'MLP':MLPClassifier(alpha=0.05, early_stopping=True, hidden_layer_sizes=(32, 64, 128),
              max_iter=120, random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric='logloss', gamma=0.01, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_bin=256,
              max_cat_to_onehot=4, max_delta_step=0, max_depth=6, max_leaves=0,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=12, num_parallel_tree=1, predictor='auto',
              random_state=42, reg_alpha=0.1, reg_lambda=1)
},
    'marital-status':{
        'RF':RandomForestClassifier(max_depth=20, max_features='auto', n_estimators=200,
                       random_state=42),
        'MLP':MLPClassifier(early_stopping=True, hidden_layer_sizes=(64,), max_iter=120,
        random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric='logloss', gamma=0.1, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_bin=256,
              max_cat_to_onehot=4, max_delta_step=0, max_depth=3, max_leaves=0,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=12, num_parallel_tree=1, predictor='auto',
              random_state=42, reg_alpha=0.1, reg_lambda=1)
}},
'Crime':{
    'race':{
        'RF':RandomForestClassifier(bootstrap=False, max_depth=20, max_features='auto',
                       min_samples_split=5, random_state=42),
        'MLP':MLPClassifier(alpha=0.05, early_stopping=True, hidden_layer_sizes=(32, 64, 128),
              max_iter=120, random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.8,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric='logloss', gamma=0.01, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_bin=256,
              max_cat_to_onehot=4, max_delta_step=0, max_depth=3, max_leaves=0,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=0, num_parallel_tree=1, predictor='auto',
              random_state=42, reg_alpha=0.01, reg_lambda=1)
}},
'German':{
    'gender':{
        'RF':RandomForestClassifier(max_depth=10, max_features='auto', min_samples_leaf=4,
                       min_samples_split=10, n_estimators=50, random_state=42),
        'MLP':MLPClassifier(early_stopping=True, hidden_layer_sizes=(32, 64, 128),
              max_iter=120, random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eval_metric='logloss', gamma=0.1, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=5, min_child_weight=1,
              monotone_constraints='()', n_estimators=300, n_jobs=16,
              num_parallel_tree=1, predictor='auto', random_state=42,
              reg_alpha=0.1, reg_lambda=1, scale_pos_weight=0.4492753623188406,
              seed=42, subsample=0.6, tree_method='exact',
              use_label_encoder=False, validate_parameters=1)},
    'age':{
        'RF':RandomForestClassifier(bootstrap=False, max_depth=20, max_features='auto',
                       min_samples_leaf=4, n_estimators=50, random_state=42),
        'MLP':MLPClassifier(early_stopping=True, hidden_layer_sizes=(32, 64, 128),
              max_iter=120, random_state=42),
        'XGB':XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric='logloss', gamma=0.01, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_bin=256,
              max_cat_to_onehot=4, max_delta_step=0, max_depth=6, max_leaves=0,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=12, num_parallel_tree=1, predictor='auto',
              random_state=42, reg_alpha=0.01, reg_lambda=1)}}
                    }


loadMap = {'Adult-debiased':{'gender':'adult-gender','marital-status':'adult-ms'},
            'Adult':{'gender':'adult-gender-biased','marital-status':'adult-ms-biased'},
            'Crime':{'race':'crime-race'},
            'German': {'age':'german-age','gender':'german-gender'}
            }

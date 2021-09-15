import pandas as pd 
from src.modeling.sklearn.classifier import ClassifierModel 
from sklearn.metrics import mean_squared_error
from typing import Union 

class ModelPipline:
  def __init__(self, model_type: Union["classifier", "regression"], random_state: int=0):
    if model_type == "classifier":
      self.model = ClassifierModel(random_state).models 
    else:
      raise NotImplementedError

    self.is_tree = ['XGBClassifier',
                    'DecisionTreeClassifier',
                    'RandomForestClassifier',
                    'GradientBoostingClassifier',
                    'GradientBoostingRegression',
                    'AdaBoostReggressor',
                    'ExtraTreesClassifier']
    self.model_type = model_type
  
  def predict(self, x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series],
           x_val: pd.DataFrame, y_val: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    model_names, model_score, is_tree = [], [], []
    
    for name, model in self.model:
      model.fit(x_train, y_train)
      score = model.score(x_val, y_val)
      model_names.append(name)
      model_score.append(score)
      if name in self.is_tree:
        is_tree.append("●")
      else:
        is_tree.append("✗")

    df = pd.DataFrame({"score": model_score, "is_tree": is_tree, "model_names": model_names})
    df = df.groupby(["is_tree", "model_names"]).max()
    return df.style.background_gradient(cmap="Blues")
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def build_bagging_model(random_state: int = 42) -> BaggingClassifier:
    base_estimator = DecisionTreeClassifier(random_state=random_state)
    return BaggingClassifier(estimator=base_estimator, n_estimators=100, random_state=random_state)

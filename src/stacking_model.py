from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier


def build_stacking_model(random_state: int = 42) -> StackingClassifier:
    estimators = [
        ("naive_bayes", BernoulliNB()),
        ("tree", DecisionTreeClassifier(random_state=random_state)),
        ("random_forest", RandomForestClassifier(random_state=random_state)),
    ]
    final_estimator = LogisticRegression(max_iter=2000, random_state=random_state)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=2,
        n_jobs=-1,
    )

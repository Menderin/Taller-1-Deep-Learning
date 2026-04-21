from sklearn.ensemble import GradientBoostingClassifier


def build_boosting_model(random_state: int = 42) -> GradientBoostingClassifier:
    """Construye un modelo de Boosting utilizando gradient boosting."""
    return GradientBoostingClassifier(random_state=random_state)

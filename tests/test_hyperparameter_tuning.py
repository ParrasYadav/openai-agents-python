from examples.rl.hyperparameter_tuning import hyperparameter_search


def test_hyperparameter_search_runs() -> None:
    params = hyperparameter_search()
    assert "gamma" in params
    assert "lr" in params
    assert "epsilon" in params
    assert "hidden_size" in params

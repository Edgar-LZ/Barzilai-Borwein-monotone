def get_params() -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "function name": "wood",
        "method": "barzilai",
        "BB type": 1,
        "max iterations": 1e5,
        "c1": 1e-4,
        "c2": 0.9,
        "tau": 1e-6,
    }
    params = _define_search_methods(params)
    return params


def _define_search_methods(params) -> dict:
    methods = {
        "barzilai": {"search algorithm": "barzilai",
                     "search name": "barzilai"},
        "steepest": {"search algorithm": "steepest",
                     "search name": "bisection"},
    }
    params.update(methods[params["method"]])
    return params

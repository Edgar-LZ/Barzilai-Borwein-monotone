def get_params() -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "function name": "rosembrock",
        "method": "ANGR1",
        "max iterations": 1e5,
        "c1": 1e-4,
        "c2": 0.9,
        "alpha": 0.1,
        "tau": 1e-6,
    }
    params = _define_search_methods(params)
    return params


def _define_search_methods(params) -> dict:
    methods = {
        "barzilai": {"search algorithm": "barzilai",
                     "search name": "barzilai",
                     "BB type": 1},
        "steepest": {"search algorithm": "steepest",
                     "search name": "bisection"},
        "ANGM": {"search algorithm": "ANGM",
                 "search name": "ANGM",
                 "tau 1": 0.85,
                 "tau 2": 1.3,
                 "BB type": 1, },
        "ANGR1": {"search algorithm": "ANGR1",
                  "search name": "ANRG1",
                  "tau 1": 0.85,
                  "tau 2": 1.3,
                  "BB type": 1, },
    }
    params.update(methods[params["method"]])
    return params

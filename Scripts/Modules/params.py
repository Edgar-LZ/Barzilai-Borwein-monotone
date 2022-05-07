def get_params() -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "function name": "wood",
        "method": "ANGR2",
        "max iterations": 1e5,
        "c1": 1e-4,
        "c2": 0.9,
        "tau": 1e-6,
    }
    params = _define_function_params(params)
    return params


def _define_function_params(params) -> dict:
    f_params = {
        "rosembrock": {
            "barzilai": {"search algorithm": "barzilai",
                         "search name": "barzilai",
                         "BB type": 1,
                         "alpha": 0.1,
                         },
            "steepest": {"search algorithm": "steepest",
                         "search name": "bisection",
                         },
            "ANGM": {"search algorithm": "ANGM",
                     "search name": "ANGM",
                     "tau 1": 0.85,
                     "tau 2": 1.3,
                     "BB type": 1,
                     "alpha": 0.1,
                     },
            "ANGR1": {"search algorithm": "ANGR1",
                      "search name": "ANRG1",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 0.1,
                      },
            "ANGR2": {"search algorithm": "ANGR1",
                      "search name": "ANRG1",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 0.1,
                      }, },
        "wood": {
            "barzilai": {"search algorithm": "barzilai",
                         "search name": "barzilai",
                         "BB type": 1,
                         "alpha": 0.1,
                         },
            "steepest": {"search algorithm": "steepest",
                         "search name": "bisection",
                         },
            "ANGM": {"search algorithm": "ANGM",
                     "search name": "ANGM",
                     "tau 1": 0.85,
                     "tau 2": 1.3,
                     "BB type": 1,
                     "alpha": 0.01,
                     },
            "ANGR1": {"search algorithm": "ANGR1",
                      "search name": "ANRG1",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 0.5,
                      },
            "ANGR2": {"search algorithm": "ANGR1",
                      "search name": "ANRG1",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 0.5,
                      },
        },
    }
    params.update(f_params[params["function name"]][params["method"]])
    return params

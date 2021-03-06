def get_params() -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "gamma folder": "gamma",
        "position folder": "position",
        "animation folder": "movie",
        "function name": "rosembrock",
        "method": "steepest",
        "max iterations": 1e4,
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
                         "alpha": 1e-5,
                         },
            "steepest": {"search algorithm": "steepest",
                         "search name": "bisection",
                         },
            "ANGM": {"search algorithm": "ANGM",
                     "search name": "ANGM",
                     "tau 1": 0.85,
                     "tau 2": 1.3,
                     "BB type": 1,
                     "alpha": 1e-6,
                     },
            "ANGR1": {"search algorithm": "ANGR1",
                      "search name": "ANRG1",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 1e-4,
                      },
            "ANGR2": {"search algorithm": "ANGR2",
                      "search name": "ANRG2",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 1e-6,
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
                      "alpha": 1e-6,
                      },
            "ANGR2": {"search algorithm": "ANGR2",
                      "search name": "ANRG2",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 1e-5,
                      },
        },
        "paper": {
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
            "ANGR2": {"search algorithm": "ANGR2",
                      "search name": "ANRG2",
                      "tau 1": 0.1,
                      "tau 2": 1,
                      "BB type": 1,
                      "alpha": 0.001,
                      },
        },
        "quadratic": {
            "barzilai": {"search algorithm": "barzilai",
                         "search name": "barzilai",
                         "BB type": 1,
                         "alpha": 0.1,
                         "lambda": 100,
                         },
            "steepest": {"search algorithm": "steepest",
                         "search name": "bisection",
                         "lambda": 100,
                         },
            "ANGM": {"search algorithm": "ANGM",
                     "search name": "ANGM",
                     "tau 1": 0.85,
                     "tau 2": 1.3,
                     "BB type": 1,
                     "alpha": 0.1,
                     "lambda": 100,
                     },
            "ANGR1": {"search algorithm": "ANGR1",
                      "search name": "ANRG1",
                      "tau 1": 0.85,
                      "tau 2": 1.3,
                      "BB type": 1,
                      "alpha": 0.1,
                      "lambda": 100,
                      },
            "ANGR2": {"search algorithm": "ANGR2",
                      "search name": "ANRG2",
                      "tau 1": 0.1,
                      "tau 2": 1,
                      "BB type": 1,
                      "alpha": 0.001,
                      "lambda": 100,
                      }, },
    }
    params.update(f_params[params["function name"]][params["method"]])
    return params


def get_graphics_params(params: dict) -> dict:
    datasets = {"ANGR1": {"x lim": [0, 275],
                          "x delta": 25,
                          },
                "ANGR2": {"x lim": [0, 275],
                          "x delta": 25,
                          },
                "ANGM": {"x lim": [0, 275],
                         "x delta": 20,
                         },
                "steepest": {"x lim": [0, 1800],
                             "x delta": 200,
                             },
                "barzilai": {"x lim": [0, 850],
                             "x delta": 100,
                             },
                }
    params.update(datasets[params["method"]])
    return params


def get_methods_names() -> dict:
    methods = {
        "steepest": {"color": "#03071E",
                     "title": "SD"},
        "barzilai": {"color": "#6A040F",
                     "title": "BB"},
        "ANGM":     {"color": "#D00000",
                     "title": "ANGM"},
        "ANGR1":    {"color": "#E85D04",
                     "title": "ANGR1"},
        "ANGR2":    {"color": "#FAA307",
                     "title": "ANGR2"},
    }
    return methods


def get_function_names() -> dict:
    functions = [
        "quadratic",
        "paper",
        "rosembrock",
        "wood"
    ]
    return functions


def get_search_methods_names() -> dict:
    methods = {
        "bisection": "#03071E",
        "barzilai": "#6A040F",
        "ANGM": "#D00000",
        "ANRG1": "#E85D04",
        "ANRG2": "#FAA307",
    }
    return methods

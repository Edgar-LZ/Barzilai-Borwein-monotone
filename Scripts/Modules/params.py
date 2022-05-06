def get_params() -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "function name": "wood",
        "search algorithm": "barzilai",
        "search name": "barzilai",
        "max iterations": 1e10,
        "c1": 1e-4,
        "c2": 0.9,
        "tau": 1e-3,
    }
    return params

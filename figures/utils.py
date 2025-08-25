def name_map(name:str) -> str:
    return {
        "fasion_mnist_784": "Fashion",
        "nuswide": "Nuswide",
        "msong": "Msong",
        "sift1m": "Sift1m",
        "glove25": "Glove25",
        "HandOutlines": "Hand Outlines",
        "StarLightCurves": "Star Light Curves",
    }.get(name, name)
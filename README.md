# pmml-zoo
![logo](docs/logo.png)

A REST server to quickly create test PMML models.

## About

`pmml-zoo` allows you to quickly generate simulated data and create **test** PMML models by sending a JSON payload to a REST server, getting back the trained model.

*What it is not*

`pmml-zoo` **doesn't** aim at creating production models, it is intended to create models for smoke, integration and unit tests.

## Usage

As an example, let's create a linear regression.
We can send the following JSON payload to `$SERVER/model/linearregression`:

```json
{"data": {
    "size": 1000,
    "inputs": [
        {"name": "feature-1",
         "type": "continuous",
         "points": [[10.0, 30.0], [20.0, 40.0], [50, 15.0], [100, 16.0]]
        },
        {"name": "feature-2",
        "type": "discrete",
        "points": [[0, 3], [2, 4], [8, 1], [9, 16]]
        },
        {"name": "feature-3",
        "type": "categorical",
        "points": [["low", 2.0], ["medium", 4.0], ["high", 1.0]]
        }
        ],
        "outputs": [
            {"name": "output-1",
            "type": "categorical",
            "points": [["low", 2.0], ["medium", 4.0], ["high", 1.0]]}
            ]
        }
}
```

### What is happening?

Data is simulated by first creating an empirical distribution by interpolating the provided `points`.
This empirical distribution is then sampled `size` times and that will be the variable data.

A complete explanation is provided in the documentation.

- `size` is the size of the dataset.
- `points` is a list of data points to use to construct the interpolation, in the format `(value, weight)`. For instance a list of `[(1.0, 2.0), (2.0, 4.0)]` means that value `2.0` will more frequent.
- `name` is the feature name, which be used in the PMML model
- `type` can be one of `continuous`, `discrete` or `categorical`
- `inputs` and `outputs` have the same format, with the obvious difference implied in the name.

After sending the above payload, a response consisting of the PMML's XML is returned.

## Supported models

For now, these are the supported models:

- Linear regression
- Random forest classification
# pmml-zoo ![Tests](https://github.com/ruivieira/pmml-zoo/workflows/Tests/badge.svg)
![logo](docs/logo.png)

A REST server to quickly create test PMML models.

## About

`pmml-zoo` allows you to quickly generate simulated data and create **test** PMML models by sending a JSON payload to a REST server, getting back the trained model.

*What it is not*

`pmml-zoo` **doesn't** aim at creating production models, it is intended to create models for smoke, integration and unit tests.

## Usage

The best way to get started is using `pmml-zoo` container image.

```shell
$ docker pull ruivieira/pmml-zoo:latest
$ docker run -i --rm -p 5000:5000 ruivieira/pmml-zoo
```

As an example, let's create a linear regression.
Assuming the server is running locally, we can send the following JSON payload to `0.0.0.0:5000/model/linearregression`:

```json
curl --request POST \
  --url http://0.0.0.0:5000/model/linearregression \
  --header 'content-type: application/json' \
  --data '
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
                    {"name": "feature-4",
         "type": "continuous",
         "points": [[1.0, 2.0], [4.0, 7.3], [7.0, 1.0], [100, 16.0]]
        }]
    }
}' \
-o model.pmml 
```

### What is happening?

Data is simulated by first creating an empirical distribution by interpolating the provided `points`.
This empirical distribution is then sampled `size` times and that will be the variable data.

An important note is that all variables are independent (although spurious correlation may occur).

A complete explanation is provided in the documentation.

- `size` is the size of the dataset.
- `points` is a list of data points to use to construct the interpolation, in the format `(value, weight)`. For instance a list of `[(1.0, 2.0), (2.0, 4.0)]` means that value `2.0` will more frequent.
- `name` is the feature name, which be used in the PMML model
- `type` can be one of `continuous`, `discrete` or `categorical`
- `inputs` and `outputs` have the same format, with the obvious difference implied in the name.

After sending the above payload, a response consisting of the PMML's XML is returned, which is save (in this example) to the `model.pmml` file.

## Supported models

For now, these are the supported models:

- Linear regression (`/model/linearregression`)
- Random forest classification (`/model/randomforest`)

## Contributing

Please use the [issues](https://github.com/ruivieira/pmml-zoo/issues) for any suggestions, feedback, PRs or bugs.
Thank you!
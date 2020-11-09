"""Tool to create test PMML models"""
# INFO: Tool to create test PMML models
# pylint: disable=R0903,R0913,C0103
from abc import ABC, abstractmethod
import random
from typing import Union, Any, List, Tuple
import uuid

import numpy as np  # type: ignore
from scipy import interpolate  # type: ignore
from flask import Flask, request, Response
from flasgger import Swagger, swag_from  # type: ignore
import pandas as pd  # type: ignore
from sklearn import preprocessing  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from nyoka import skl_to_pmml  # type: ignore


def _choose_interp_method(n_points: int) -> str:
    if n_points >= 4:
        method = "cubic"
    elif n_points == 3:
        method = "quadratic"
    elif n_points == 2:
        method = "slinear"
    elif n_points == 1:
        method = "zero"
    else:
        raise RuntimeError("Must have at least one point")
    return method


def create_continuous_histogram(start: int, end: int, data, size: int):
    """Create a continuous histogram"""
    x = [p[0] for p in data]
    y = [p[1] for p in data]

    n_points = len(data)

    f = interpolate.interp1d(
        x, y, fill_value="extrapolate", kind=_choose_interp_method(n_points)
    )

    xnew = np.linspace(start, end, size)
    ynew = f(xnew)
    result = np.empty([size, 2])
    result[:, 0] = xnew
    result[:, 1] = ynew

    return result


def sample_from_histogram(histogram, size):
    """Sample data points from the interpolated histogram"""
    weights = histogram[:, 1].astype(float)
    return random.choices(population=histogram[:, 0], weights=weights, k=size)


def create_discrete_histogram(start: int, end: int, data, size: int):
    """Create a discrete series"""
    result = create_continuous_histogram(start=start, end=end, data=data, size=size)
    result[:, 0] = result[:, 0].astype(int)

    return result


def create_categorical_histogram(data):
    """Create a categorical series"""
    return np.array(data)


class Variable(ABC):
    """Variable abstract class"""

    def __init__(self, name, size, points):
        self.name = name
        self.points = points
        self.size = size

    @abstractmethod
    def generate_data(self):
        """Generate the data associated with this variable"""


class ContinuousVariable(Variable):
    """Continuous variable"""

    def __init__(self, name, size, points, start, end):
        super().__init__(name, size, points)
        self.start = start
        self.end = end

    def generate_data(self):
        histogram = create_continuous_histogram(
            start=self.start,
            end=self.end,
            data=self.points,
            size=self.size,
        )
        return sample_from_histogram(histogram, self.size)

    def __repr__(self):
        return f"""ContinuousVariable{{
            name={self.name}, start={self.start}, end={self.end}, size={self.size}
            }}"""


class DiscreteVariable(Variable):
    """Continuous variable"""

    def __init__(self, name, size, points, start, end):
        super().__init__(name, size, points)
        self.start = start
        self.end = end

    def generate_data(self):
        histogram = create_discrete_histogram(
            start=self.start,
            end=self.end,
            data=self.points,
            size=1000,
        )
        return sample_from_histogram(histogram, self.size)

    def __repr__(self):
        return f"""DiscreteVariable{{
            name={self.name}, start={self.start}, end={self.end}, size={self.size}
            }}"""


class CategoricalVariable(Variable):
    """Continuos variable"""

    def generate_data(self):
        histogram = create_categorical_histogram(data=self.points)
        return sample_from_histogram(histogram, self.size)

    def __repr__(self):
        return f"CategoricalVariable{{name={self.name}, size={self.size}}}"


# Model generation
TVariable = Union[ContinuousVariable, DiscreteVariable, CategoricalVariable]
Inputs = List[TVariable]
Outputs = Inputs


def __create_dataframe(variables: List[TVariable]) -> pd.DataFrame:
    df = pd.DataFrame()
    for variable in variables:
        # drop indices
        if isinstance(variable, CategoricalVariable):
            le = preprocessing.LabelEncoder()
            # encode categorical variables
            df[variable.name] = le.fit_transform(variable.generate_data())
        elif isinstance(variable, DiscreteVariable):
            df[variable.name] = variable.generate_data()
            df[variable.name] = df[variable.name].astype(int)
        else:
            df[variable.name] = variable.generate_data()
    return df


def generate_linear_regression(inputs: Inputs, outputs: Outputs):
    """Generate a linear regression from the supplied variables"""
    input_df = __create_dataframe(inputs)
    output_df = __create_dataframe(outputs)
    pipeline_obj = Pipeline([("model", LinearRegression())])
    pipeline_obj.fit(input_df, output_df)
    temporary_file = f"/tmp/{uuid.uuid4()}.pmml"
    skl_to_pmml(pipeline_obj, input_df.columns, output_df.columns, temporary_file)
    with open(temporary_file, "r") as file:
        data = file.read()
    return data


def generate_random_forest(inputs: Inputs, outputs: Outputs):
    """Generate a random forest from the supplied variables"""
    input_df = __create_dataframe(inputs)
    output_df = __create_dataframe(outputs)
    pipeline_obj = Pipeline([("model", RandomForestClassifier())])
    pipeline_obj.fit(input_df, output_df)
    temporary_file = f"/tmp/{uuid.uuid4()}.pmml"
    skl_to_pmml(pipeline_obj, input_df.columns, output_df.columns, temporary_file)
    with open(temporary_file, "r") as file:
        data = file.read()
    return data


def build_variable(json_str, data_size: int) -> TVariable:
    """Parse variable information from JSON"""
    name: str = json_str["name"]
    points = json_str["points"]
    variable_type: str = json_str["type"]
    if variable_type != "categorical":
        start = json_str.get("start", min([p[0] for p in points]))
        end = json_str.get("end", max([p[0] for p in points]))
    else:
        start = 0.0
        end = 0.0

    if variable_type == "continuous":
        variable_obj: TVariable = ContinuousVariable(
            name=name,
            start=start,
            end=end,
            size=data_size,
            points=points,
        )
    elif variable_type == "discrete":
        variable_obj = DiscreteVariable(
            name=name,
            start=start,
            end=end,
            size=data_size,
            points=points,
        )
    else:
        variable_obj = CategoricalVariable(
            name=name,
            size=data_size,
            points=points,
        )
    return variable_obj


def parse_payload(payload: Any) -> Tuple[Inputs, Outputs]:
    """Parse JSON payload into inputs and outputs"""
    data_size = int(payload["size"])
    # get inputs
    inputs = []
    for _input in payload["inputs"]:
        inputs.append(build_variable(_input, data_size))
    outputs = []
    for _output in payload["outputs"]:
        outputs.append(build_variable(_output, data_size))
    return (inputs, outputs)


# REST server
app = Flask(__name__)
app.config["SWAGGER"] = {"title": "pmml-zoo", "uiversion": 3, "openapi": "3.0.2"}
swagger = Swagger(app)


@app.route("/model/linear-regression", methods=["POST"])
@swag_from("openapi/linear-regression.yml")
def linear_regression():
    """Linear regression model endpoint"""
    inputs, outputs = parse_payload(request.json)
    model = generate_linear_regression(inputs, outputs)

    return Response(model, mimetype="text/xml")


@app.route("/model/random-forest", methods=["POST"])
@swag_from("openapi/random-forest.yml")
def random_forest():
    """Random forest model endpoint"""
    inputs, outputs = parse_payload(request.json)
    model = generate_random_forest(inputs, outputs)

    return Response(model, mimetype="text/xml")


if __name__ == "__main__":
    app.run(host="0.0.0.0")

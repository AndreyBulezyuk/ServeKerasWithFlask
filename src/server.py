import sys, os, json
sys.path.insert(1, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from flask import Flask, request
from service import Service

"""
Author: Andrey Bulezyuk @ German IT Academy (https://git-academy.com)
Date: 18.01.2020
"""

application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello World!"


@application.route("/<string:service_name>/<string:model_name>", methods=["GET", "POST"])
def train(service_name=None, model_name=None):
    service = Service(model_name=model_name)

    # GET Request is enough to trigger a training process
    if service_name == 'train':
        service.train()
    # POST Request is required to get the X data for prediction process
    elif service_name == 'predict':
        service.predict()

    return f"Service: {service_name}. Model: {model_name}. Success."

if __name__ == "__main__":
    application.run(debug=True)

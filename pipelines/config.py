"""Shared settings. Infrastructure values are read from the CloudFormation stack outputs."""

import os
from functools import lru_cache

import boto3

REGION = os.environ.get("AWS_REGION", "eu-north-1")
PROJECT = "plant-disease"
STACK_NAME = f"{PROJECT}-infra"

TRAIN_PIPELINE_NAME = f"{PROJECT}-train"
DEPLOY_PIPELINE_NAME = f"{PROJECT}-deploy"
MODEL_PACKAGE_GROUP = f"{PROJECT}-models"
ENDPOINT_NAME = f"{PROJECT}-serverless"

# AWS Deep Learning Containers - prebuilt, so nothing is stored (or paid for) in your ECR.
PYTORCH_VERSION = "2.6.0"
PY_VERSION = "py312"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")


def script(name):
    """Relative path to a script in src/. The SDK parses `code=` as a URL, so an absolute
    Windows path like "D:/..." is mistaken for a URL with scheme "d"."""
    return os.path.relpath(os.path.join(SRC_DIR, name)).replace(os.sep, "/")


@lru_cache(maxsize=1)
def stack_outputs():
    cfn = boto3.client("cloudformation", region_name=REGION)
    stack = cfn.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}

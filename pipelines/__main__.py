"""CLI: python -m pipelines {upsert|train|deploy|show} [options]

  upsert                      create/update both pipelines in SageMaker
  train [--smoke]             start the training pipeline (--smoke: 50 imgs/class, 1 epoch, CPU)
  deploy --package-arn ARN    deploy a specific approved model package manually
  show                        print both pipeline definitions (JSON)
"""

import argparse
import json

import boto3
from sagemaker.core import image_uris
from sagemaker.core.workflow.pipeline_context import PipelineSession

from pipelines import config, deploy_pipeline, train_pipeline


def pipelines():
    out = config.stack_outputs()
    session = PipelineSession(
        boto_session=boto3.Session(region_name=config.REGION),
        default_bucket=out["BucketName"],
    )
    role = out["SageMakerRoleArn"]
    return (
        train_pipeline.build(session, role, out["BucketName"]),
        deploy_pipeline.build(session, role),
        role,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["upsert", "train", "deploy", "show"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--package-arn")
    args = parser.parse_args()

    train, deploy, role = pipelines()
    tags = [{"Key": "project", "Value": config.PROJECT}]

    if args.command == "show":
        for p in (train, deploy):
            print(json.dumps(json.loads(p.definition()), indent=2))
    elif args.command == "upsert":
        for p in (train, deploy):
            p.upsert(role_arn=role, tags=tags)
            print(f"Upserted {p.name}")
    elif args.command == "train":
        params = {}
        if args.smoke:
            params = {"MaxImagesPerClass": 50, "Epochs": 1, "TrainInstanceType": "ml.m5.xlarge",
                      "TrainImage": image_uris.retrieve(
                          "pytorch", region=config.REGION, version=config.PYTORCH_VERSION,
                          py_version=config.PY_VERSION, instance_type="ml.m5.xlarge",
                          image_scope="training"),
                      "MinTestAccuracy": 0.0}
        execution = train.start(parameters=params)
        print(f"Started: {execution.arn}")
    elif args.command == "deploy":
        if not args.package_arn:
            parser.error("deploy requires --package-arn")
        execution = deploy.start(parameters={"ModelPackageArn": args.package_arn})
        print(f"Started: {execution.arn}")


if __name__ == "__main__":
    main()

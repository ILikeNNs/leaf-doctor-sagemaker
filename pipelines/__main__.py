"""CLI: python -m pipelines {upsert|train|deploy|show} [options]

  upsert                      create/update both pipelines in SageMaker
  train [--smoke]             upsert + start the training pipeline (--smoke: 50 imgs/class, 1 epoch, CPU)
        [--instance-type T]   GPU training instance, e.g. ml.g5.xlarge when g4dn spot has no capacity
        [--on-demand]         train on on-demand instead of spot capacity
  deploy --package-arn ARN    deploy a specific approved model package manually
  show                        print both pipeline definitions (JSON)
"""

import argparse
import json

import boto3
from sagemaker.core.workflow.pipeline_context import PipelineSession

from pipelines import config, deploy_pipeline, train_pipeline


def pipelines(spot=True):
    out = config.stack_outputs()
    session = PipelineSession(
        boto_session=boto3.Session(region_name=config.REGION),
        default_bucket=out["BucketName"],
    )
    role = out["SageMakerRoleArn"]
    return (
        train_pipeline.build(session, role, out["BucketName"], spot=spot),
        deploy_pipeline.build(session, role, out["DeployEndpointFunctionArn"]),
        role,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["upsert", "train", "deploy", "show"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--package-arn")
    parser.add_argument("--instance-type")
    parser.add_argument("--on-demand", action="store_true")
    args = parser.parse_args()

    train, deploy, role = pipelines(spot=not args.on_demand)
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
                      "TrainImage": config.pytorch_image("ml.m5.xlarge"),
                      "MinTestAccuracy": 0.0}
        elif args.instance_type:
            params = {"TrainInstanceType": args.instance_type}
        # Spot vs on-demand is baked into the definition, so sync it before starting.
        train.upsert(role_arn=role, tags=tags)
        execution = train.start(parameters=params)
        print(f"Started: {execution.arn}")
    elif args.command == "deploy":
        if not args.package_arn:
            parser.error("deploy requires --package-arn")
        execution = deploy.start(parameters={"ModelPackageArn": args.package_arn})
        print(f"Started: {execution.arn}")


if __name__ == "__main__":
    main()

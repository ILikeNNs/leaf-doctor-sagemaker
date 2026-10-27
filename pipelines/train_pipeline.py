"""Training pipeline: Preprocess -> Train (spot) -> Evaluate -> accuracy gate -> Register.

The registered model package waits in PendingManualApproval. Approving it in the
Model Registry fires an EventBridge rule that starts the deploy pipeline.
"""

from sagemaker.core import image_uris
from sagemaker.core.model_metrics import MetricsSource, ModelMetrics
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import ProcessingInput, ProcessingOutput, ProcessingS3Input, ProcessingS3Output
from sagemaker.core.training.configs import CheckpointConfig, Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join, JsonGet
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow import CacheConfig, ConditionStep, FailStep, ModelStep, Pipeline, ProcessingStep, TrainingStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer

from pipelines import config


def build(session: PipelineSession, role: str, bucket: str, spot: bool = True) -> Pipeline:
    s3 = f"s3://{bucket}"
    region = session.boto_region_name

    train_instance = ParameterString("TrainInstanceType", default_value="ml.g4dn.xlarge")
    epochs = ParameterInteger("Epochs", default_value=5)
    arch = ParameterString("Arch", default_value="efficientnet_b0")
    max_per_class = ParameterInteger("MaxImagesPerClass", default_value=0)  # e.g. 50 for a cheap smoke test
    min_accuracy = ParameterFloat("MinTestAccuracy", default_value=0.95)
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    cpu_image = image_uris.retrieve(
        "pytorch", region=region, version=config.PYTORCH_VERSION, py_version=config.PY_VERSION,
        instance_type="ml.m5.xlarge", image_scope="training",
    )
    gpu_image = image_uris.retrieve(
        "pytorch", region=region, version=config.PYTORCH_VERSION, py_version=config.PY_VERSION,
        instance_type="ml.g4dn.xlarge", image_scope="training",
    )
    # Must match TrainInstanceType (GPU image for ml.g4dn/g5, CPU image for ml.m5/c5).
    train_image = ParameterString("TrainImage", default_value=gpu_image)
    inference_image = image_uris.retrieve(
        "pytorch", region=region, version=config.PYTORCH_VERSION, py_version=config.PY_VERSION,
        instance_type="ml.m5.large", image_scope="inference",
    )

    def processor(name, volume_gb):
        return ScriptProcessor(
            image_uri=cpu_image, command=["python3"], role=role,
            instance_type="ml.m5.xlarge", instance_count=1, volume_size_in_gb=volume_gb,
            max_runtime_in_seconds=3600, base_job_name=f"{config.PROJECT}-{name}",
            sagemaker_session=session,
        )

    # 1. Download + split. Cached for 30 days so retraining doesn't re-download 2 GB.
    preprocess = ProcessingStep(
        name="Preprocess",
        step_args=processor("preprocess", 30).run(
            code=config.script("preprocess.py"),
            arguments=["--max-per-class", max_per_class.to_string()],
            outputs=[
                ProcessingOutput(output_name="dataset", s3_output=ProcessingS3Output(
                    s3_uri=f"{s3}/data/plantvillage", local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob")),
            ],
        ),
        cache_config=CacheConfig(enable_caching=True, expire_after="P30D"),
    )
    dataset_uri = preprocess.properties.ProcessingOutputConfig.Outputs["dataset"].S3Output.S3Uri

    # 2. Fine-tune on managed spot capacity (or on-demand if spot=False), checkpointing every epoch.
    #    Spot is a build-time choice: the SDK's Compute config doesn't accept a pipeline parameter for it.
    trainer = ModelTrainer(
        training_image=train_image,
        source_code=SourceCode(source_dir=config.SRC_DIR, entry_script="train.py"),
        compute=Compute(instance_type=train_instance, instance_count=1, volume_size_in_gb=30,
                        enable_managed_spot_training=spot),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=2 * 3600,
                                             max_wait_time_in_seconds=4 * 3600 if spot else None),
        # Per-execution prefix: a spot restart resumes its own run, never a previous one.
        checkpoint_config=CheckpointConfig(
            s3_uri=Join(on="/", values=[f"{s3}/checkpoints", ExecutionVariables.PIPELINE_EXECUTION_ID])),
        output_data_config=OutputDataConfig(s3_output_path=f"{s3}/models"),
        hyperparameters={"epochs": epochs.to_string(), "arch": arch, "batch-size": 64, "lr": 1e-3},
        base_job_name=f"{config.PROJECT}-train",
        role=role,
        sagemaker_session=session,
    )
    train = TrainingStep(
        name="Train",
        step_args=trainer.train(input_data_config=[
            InputData(channel_name="train", data_source=Join(on="/", values=[dataset_uri, "train"])),
            InputData(channel_name="val", data_source=Join(on="/", values=[dataset_uri, "val"])),
        ]),
    )
    model_data = train.properties.ModelArtifacts.S3ModelArtifacts

    # 3. Evaluate on the official PlantVillage test split.
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    evaluate = ProcessingStep(
        name="Evaluate",
        step_args=processor("evaluate", 20).run(
            code=config.script("evaluate.py"),
            inputs=[
                ProcessingInput(input_name="model", s3_input=ProcessingS3Input(
                    s3_uri=model_data, local_path="/opt/ml/processing/model",
                    s3_data_type="S3Prefix", s3_input_mode="File")),
                ProcessingInput(input_name="test", s3_input=ProcessingS3Input(
                    s3_uri=Join(on="/", values=[dataset_uri, "test"]), local_path="/opt/ml/processing/test",
                    s3_data_type="S3Prefix", s3_input_mode="File")),
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation", s3_output=ProcessingS3Output(
                    s3_uri=f"{s3}/evaluation", local_path="/opt/ml/processing/evaluation",
                    s3_upload_mode="EndOfJob")),
            ],
        ),
        property_files=[evaluation_report],
    )
    evaluation_uri = evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri

    # 4. Register the model package (model.tar.gz already contains code/inference.py).
    model = ModelBuilder(
        image_uri=inference_image,
        s3_model_data_url=model_data,
        env_vars={"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"},
        role_arn=role,
        sagemaker_session=session,
        dependencies=None,  # code is already inside model.tar.gz; skip script-mode upload/env overrides
    )
    register = ModelStep(
        name="Register",
        step_args=model.register(
            model_package_group_name=config.MODEL_PACKAGE_GROUP,
            content_types=["image/jpeg", "image/png"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            approval_status=approval,
            model_metrics=ModelMetrics(model_statistics=MetricsSource(
                content_type="application/json",
                s3_uri=Join(on="/", values=[evaluation_uri, "evaluation.json"]),
            )),
            domain="COMPUTER_VISION",
            task="IMAGE_CLASSIFICATION",
        ),
    )

    # 5. Only register models that beat the accuracy bar.
    gate = ConditionStep(
        name="AccuracyGate",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluate.name, property_file=evaluation_report,
                         json_path="multiclass_classification_metrics.accuracy.value"),
            right=min_accuracy,
        )],
        if_steps=[register],
        else_steps=[FailStep(name="AccuracyTooLow",
                             error_message=Join(on=" ", values=["Test accuracy below", min_accuracy]))],
    )

    return Pipeline(
        name=config.TRAIN_PIPELINE_NAME,
        parameters=[train_instance, train_image, epochs, arch, max_per_class, min_accuracy, approval],
        steps=[preprocess, train, evaluate, gate],
        sagemaker_session=session,
        pipeline_experiment_config=None,
    )

"""Deploy pipeline: approved model package -> Model -> serverless EndpointConfig -> Endpoint.

EndpointStep creates the endpoint on the first run and updates it in place
afterwards. Serverless inference has no idle cost, which is what keeps the
project under budget.
"""

from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow import EndpointConfigStep, EndpointStep, ModelStep, Pipeline

from pipelines import config


class _PackageModel:
    """Minimal stand-in for a Model object so ModelStep can capture a CreateModel call
    that references a registered model package (instead of an image + artifact URI)."""

    def __init__(self, session):
        self.sagemaker_session = session


def build(session: PipelineSession, role: str) -> Pipeline:
    package_arn = ParameterString("ModelPackageArn")
    memory_mb = ParameterInteger("MemorySizeInMB", default_value=3072)
    max_concurrency = ParameterInteger("MaxConcurrency", default_value=2)

    # Unique per execution so an update never collides with the live config/model.
    suffix = ExecutionVariables.PIPELINE_EXECUTION_ID
    model_name = Join(on="-", values=[config.PROJECT, suffix])
    config_name = Join(on="-", values=[config.PROJECT, "cfg", suffix])

    # Under a PipelineSession, create_model() records the request on session.context
    # instead of calling the API; that captured context is the ModelStep's step_args.
    session.init_model_step_arguments(_PackageModel(session))
    session.create_model(name=model_name, role=role, container_defs={"ModelPackageName": package_arn})
    create_model = ModelStep(name="CreateModel", step_args=session.context)
    session.context = None

    endpoint_config = EndpointConfigStep(
        name="CreateServerlessConfig",
        step_args=session.create_endpoint_config(
            name=config_name,
            production_variants=[{
                "VariantName": "AllTraffic",
                "ModelName": create_model.properties.ModelName,
                "ServerlessConfig": {"MemorySizeInMB": memory_mb, "MaxConcurrency": max_concurrency},
            }],
        ),
    )
    endpoint = EndpointStep(
        name="DeployEndpoint",
        step_args=session.create_endpoint(
            endpoint_name=config.ENDPOINT_NAME,
            config_name=endpoint_config.properties.EndpointConfigName,
        ),
    )

    return Pipeline(
        name=config.DEPLOY_PIPELINE_NAME,
        parameters=[package_arn, memory_mb, max_concurrency],
        steps=[create_model, endpoint_config, endpoint],
        sagemaker_session=session,
        pipeline_experiment_config=None,
    )

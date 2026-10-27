# Leaf Doctor: plant disease detection on SageMaker

An end-to-end deep learning project on AWS (eu-north-1, Stockholm). It fine-tunes an
ImageNet CNN (EfficientNet-B0 by default) on the public **PlantVillage** dataset
(54k leaf images, 38 crop/disease classes). The model is trained, evaluated, registered
and deployed with **SageMaker Pipelines**, served from a **serverless endpoint**, and
used through a small web page.

```
                      plant-disease-train (SageMaker Pipeline)
 ┌────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐
 │ Preprocess │──▶│ Train (spot  │──▶│ Evaluate │──▶│ AccuracyGate │──▶│ Register model   │
 │ download + │   │ GPU, ckpts)  │   │ test set │   │ acc >= 0.95? │   │ PendingApproval  │
 │ leaf split │   └──────────────┘   └──────────┘   └──────┬───────┘   └────────┬─────────┘
 └────────────┘                                     else ▼ Fail                 │ you approve
                                                                                 ▼
                     EventBridge "Model Package State Change" ─▶ Lambda (trigger)
                                                                                 │
                      plant-disease-deploy (SageMaker Pipeline)                  ▼
                 ┌─────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐
                 │ CreateModel │──▶│ EndpointConfig           │──▶│ Endpoint (create or    │
                 │ from package│   │ Serverless 3 GB, conc. 2 │   │ update in place)       │
                 └─────────────┘   └──────────────────────────┘   └───────────┬────────────┘
                                                                              │
         web/index.html ──POST image──▶ HTTP API ──▶ Lambda ──InvokeEndpoint──┘
```

## Highlights

- **No data leakage.** PlantVillage has several photos of each leaf. The validation split is made *per leaf* (hashing the leaf ID), and testing uses the official held-out split.
- **Quality gate.** The model is registered only if test accuracy is at least the `MinTestAccuracy` parameter. Accuracy, macro F1 and the confusion matrix are attached to the model package in the Model Registry.
- **Human in the loop.** New models wait in `PendingManualApproval`. Approving one triggers the deploy pipeline automatically.
- **Same preprocessing for evaluation and serving.** `evaluate.py` imports the endpoint's own `inference.py`, so the test scores come from the exact code path that serves predictions.
- **Spot training with resume.** The training script checkpoints every epoch to S3 and resumes if the spot instance is reclaimed.
- **No custom containers.** It uses AWS PyTorch 2.6 Deep Learning Containers, so nothing is stored in ECR.

## Repository layout

| Path | What it is |
|---|---|
| `src/preprocess.py` | Processing job: downloads PlantVillage from Hugging Face and writes train/val/test ImageFolders |
| `src/train.py` | Training job: fine-tuning with AMP, OneCycle and label smoothing, then a TorchScript export |
| `src/inference.py` | Endpoint handlers (packed into `model.tar.gz` under `code/`) |
| `src/evaluate.py` | Processing job: test-set metrics written to `evaluation.json` |
| `pipelines/` | Definitions of both pipelines plus a CLI (`python -m pipelines ...`) |
| `infra/template.yaml` | S3, IAM role, model package group, approval trigger, HTTP API and predict Lambda |
| `infra/budget.yaml` | $10/month budget with email alerts (us-east-1) |
| `web/index.html` | Static demo page (GitHub Pages or any static host) |

## One-time access setup (administrator)

The project runs under its own least-privilege IAM user, `plant-disease-dev`. It signs in through the console (with MFA) and uses `aws login`, and it has no access keys. `infra/iam-deployer.yaml` creates:

- **The user**, with two managed policies:
  - **`plant-disease-deployer-infra`**: CloudFormation on the `plant-disease-infra` and `plant-disease-budget` stacks only, plus the S3, Lambda, HTTP API, EventBridge, log and budget resources named `plant-disease-*`.
  - **`plant-disease-deployer-ml`**: SageMaker pipelines, model registry and endpoint named `plant-disease-*`, read-only SageMaker access in eu-north-1, job logs, cost reads, SageMaker quota requests, and managing its own password and MFA.
  - It also gets the AWS managed `SignInLocalDevelopmentAccess` policy, which `aws login` requires.
- **`plant-disease-boundary`**: a permissions boundary. The user can create IAM roles only if they are named `plant-disease-*` **and** carry this boundary, and can't remove it. So any role the stacks create is capped at this project's S3 bucket, SageMaker in eu-north-1, and pulling AWS's prebuilt containers. This blocks using CloudFormation to escalate privileges.

Run these once, signed in as an administrator:

```powershell
aws login        # your admin identity
aws cloudformation deploy --region eu-north-1 --stack-name plant-disease-access `
  --template-file infra/iam-deployer.yaml --capabilities CAPABILITY_NAMED_IAM

# Enable console sign-in with a temporary password that must be changed at first login
aws iam create-login-profile --user-name plant-disease-dev --password-reset-required `
  --password "<temporary-strong-password>"
aws logout
```

Then, as the new user:
1. Open `https://<account-id>.signin.aws.amazon.com/console`, sign in as `plant-disease-dev`, and set a new password.
2. Register an MFA device (top-right menu → Security credentials).
3. Sign in from the CLI and make this the default profile for the session:
   ```powershell
   aws login --profile plant-disease
   $env:AWS_PROFILE = "plant-disease"
   ```

Sessions last up to 12 hours. Run `aws login --profile plant-disease` again when one expires.

The deployer can't change its own permissions or modify the `plant-disease-access` stack. Changing what the user may do needs the administrator.

## Prerequisites

1. AWS CLI v2.32+ signed in as `plant-disease-dev` (see above)
2. Python 3.12 with dependencies: `python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt`
3. **Service quotas in eu-north-1.** New accounts often start at 0. In Service Quotas → Amazon SageMaker, request:
   - `ml.g4dn.xlarge for spot training job usage` → 1
   - `ml.m5.xlarge for processing job usage` → 1 (usually already available)
   - `ml.m5.xlarge for spot training job usage` → 1 (only for `--smoke` runs)

   If ml.g4dn isn't offered in eu-north-1 for your account, start a run with
   `TrainInstanceType=ml.g5.xlarge` and the GPU image (see *Pipeline parameters*).

## Deploy

Run everything from the repository root (PowerShell):

```powershell
# 1. Infrastructure
aws cloudformation deploy --region eu-north-1 --stack-name plant-disease-infra `
  --template-file infra/template.yaml --capabilities CAPABILITY_NAMED_IAM

aws cloudformation deploy --region us-east-1 --stack-name plant-disease-budget `
  --template-file infra/budget.yaml --parameter-overrides BudgetEmail=you@example.com

# 2. Create/update both pipelines
python -m pipelines upsert

# 3. Cheap end-to-end smoke test: 50 images/class, 1 epoch, CPU, no accuracy gate
python -m pipelines train --smoke

# 4. Full training run (GPU spot)
python -m pipelines train
```

Watch the runs in SageMaker Studio → Pipelines, or with `aws sagemaker list-pipeline-executions --pipeline-name plant-disease-train`.

### Approve and deploy

```powershell
aws sagemaker list-model-packages --model-package-group-name plant-disease-models `
  --query "ModelPackageSummaryList[].[ModelPackageArn,ModelApprovalStatus]" --output table

aws sagemaker update-model-package --model-approval-status Approved `
  --model-package-arn <arn-from-above>
```

Approval starts `plant-disease-deploy` automatically. The first run creates the endpoint `plant-disease-serverless`, and later runs update it in place. To redeploy a specific version by hand, run `python -m pipelines deploy --package-arn <arn>`.

### Test it

```powershell
$url = aws cloudformation describe-stacks --region eu-north-1 --stack-name plant-disease-infra `
  --query "Stacks[0].Outputs[?OutputKey=='PredictUrl'].OutputValue" --output text
curl.exe -X POST -H "Content-Type: image/jpeg" --data-binary "@leaf.jpg" $url
```

Then set `API_URL` in `web/index.html` to that URL and publish the page (for example on GitHub Pages). Redeploy the stack with `--parameter-overrides AllowedOrigin=https://<you>.github.io` to restrict CORS to your page.

## Pipeline parameters

| Parameter | Default | Notes |
|---|---|---|
| `TrainInstanceType` | `ml.g4dn.xlarge` | Must match `TrainImage` (GPU vs CPU) |
| `TrainImage` | PyTorch 2.6 GPU DLC | `--smoke` switches this to the CPU image |
| `Epochs` | 5 | Usually enough for ~97–99% on PlantVillage |
| `Arch` | `efficientnet_b0` | or `mobilenet_v3_large` (smaller and faster on CPU) |
| `MaxImagesPerClass` | 0 (all) | Caps the dataset for quick runs |
| `MinTestAccuracy` | 0.95 | The quality gate |
| `ModelApprovalStatus` | `PendingManualApproval` | Set to `Approved` to auto-deploy |

Override them with `start(parameters=...)` in `pipelines/__main__.py`, or in Studio when you start a run.

## Cost (budget: under $10/month)

What you pay for (check [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/) for eu-north-1):

- **Training**: a few minutes to under an hour of spot GPU per run. This is usually the largest item.
- **Processing**: Preprocess runs on ml.m5.xlarge and is cached for 30 days. Evaluate takes a few minutes.
- **Serverless endpoint**: billed per millisecond of inference only. Zero when idle.
- **S3**: about 1 GB of images plus models. Checkpoints expire after 7 days.
- **Lambda, API Gateway, EventBridge**: effectively free at demo traffic.

Things that would break the budget, and how this project avoids them: real-time endpoints (none are used), NAT gateways and VPC endpoints (no VPC), Studio apps left running (everything runs from the CLI), and abuse of the public API (HTTP API throttling at 5 req/s plus endpoint `MaxConcurrency` of 2).

## Teardown

```powershell
aws sagemaker delete-endpoint --region eu-north-1 --endpoint-name plant-disease-serverless
aws sagemaker delete-pipeline --region eu-north-1 --pipeline-name plant-disease-train
aws sagemaker delete-pipeline --region eu-north-1 --pipeline-name plant-disease-deploy
aws s3 rm s3://plant-disease-<account-id>-eu-north-1 --recursive
aws cloudformation delete-stack --region eu-north-1 --stack-name plant-disease-infra
```

Delete the model packages first, or CloudFormation can't remove the model package group.
Finally, as the administrator, delete the budget stack in us-east-1 and then `plant-disease-access` (it must go last, because the project's roles use its boundary). Endpoint configs and models cost nothing, but you can clean them up in the console.

## Ideas for next steps

- Grad-CAM heatmaps in the response (export the eager model alongside TorchScript)
- Evaluate on PlantDoc (field photos) to measure the domain shift, and fine-tune on it
- GitHub Actions: lint, then `python -m pipelines upsert` on every merge to main

## Dataset and licence

PlantVillage by Hughes and Salathé (2015), via [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), licensed under CC BY-SA 3.0.
Mohanty, Hughes and Salathé, *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science, 2016.

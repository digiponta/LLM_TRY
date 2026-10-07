# LLM_TRY v10.13.2 Modified-Subject Retraining
#
# Rebuilds data-nagato-derived Semantic Memory using:
#   modifier + subject + predicate
#       -> canonical subject => full original proposition
# then executes /sleep on the current online checkpoint.

$ErrorActionPreference = "Stop"

python .\verify_modified_subject_learning_v10132.py
if ($LASTEXITCODE -ne 0) {
    throw "v10.13.2 modified-subject regression failed."
}

$model = "model/model-gpu-v1.6.2-online.pt"
if (-not (Test-Path $model)) {
    $model = "model/model-llm-try-nagato-chat-v94.pt"
}

Write-Host ""
Write-Host "Using checkpoint: $model"
Write-Host "Rebuilding Nagato Semantic Memory and starting semantic sleep..."
Write-Host ""

@(
    "/sleep status"
    "/sleep"
    "/sleep status"
    "/exit"
) | python .\chat.py --model $model

if ($LASTEXITCODE -ne 0) {
    throw "Semantic sleep retraining failed."
}

Write-Host ""
Write-Host "v10.13.2 modified-subject retraining completed."

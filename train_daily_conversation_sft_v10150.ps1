$ErrorActionPreference = "Stop"

python .\\build_daily_conversation_sft_v10150.py
if ($LASTEXITCODE -ne 0) { throw "Daily Conversation SFT dataset build failed." }

$base = "model/model-gpu-v1.6.2-online.pt"
if (-not (Test-Path $base)) {
    $base = "model/model-llm-try-nagato-chat-v94.pt"
}

python .\\online_train.py `
    --chat-data data/daily_conversation_sft_v10150.jsonl `
    --replay-data data/conversation-ja.txt `
    --stability-data data/stability_replay_v104.jsonl `
    --state data/daily_conversation_sft_state_v10150.json `
    --base-model $base `
    --output model/model-gpu-v10.15-daily-chat.pt `
    --epochs 12 `
    --learning-rate 5e-6 `
    --batch-size 8 `
    --manual-weight 6 `
    --replay-ratio 2.0 `
    --stability-weight 2 `
    --trusted-replay-weight 1

if ($LASTEXITCODE -ne 0) { throw "Daily Conversation SFT training failed." }

Write-Host ""
Write-Host "Daily Conversation SFT completed."
Write-Host "Test with:"
Write-Host "python .\\chat.py --model model/model-gpu-v10.15-daily-chat.pt --daily-chat-generation"

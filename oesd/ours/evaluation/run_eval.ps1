# $ModelPath = "oesd/ours/train_results/checkpoints/model.zip"
$ModelPath = "/home/juhasz/Desktop/UZH/Reinforcement_Learning/Project_31/Open-Ended-Skill-Discovery/oesd/ours/train_results/checkpoints/rl_model_3200_steps.zip"
$EnvName = "minigrid"

# Check if model exists
if (-Not (Test-Path $ModelPath)) {
    Write-Host "Warning: Model file not found at $ModelPath" -ForegroundColor Yellow
    Write-Host "Please ensure the .zip file is downloaded to oesd/ours/train_results/checkpoints/" -ForegroundColor Yellow
}

python oesd/ours/evaluation/eval.py `
    --env_name $EnvName `
    --skill_count_per_algo 8 `
    --config_path "oesd/ours/configs/config1.py" `
    --model_path $ModelPath `
    --num_episodes 10 `
    --output_dir "oesd/ours/evaluation/results"

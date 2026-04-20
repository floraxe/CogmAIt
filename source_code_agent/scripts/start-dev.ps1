Param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Start-Infrastructure {
    Write-Host "==> 启动 MySQL 容器..."
    docker compose up -d mysql

    $minioExists = docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch "minio-server"
    if ($minioExists) {
        Write-Host "==> 检测到已有 minio-server，复用现有容器..."
        $running = docker inspect -f "{{.State.Running}}" minio-server
        if ($running -ne "true") {
            docker start minio-server | Out-Null
        }
    } else {
        Write-Host "==> 未检测到 minio-server，使用 compose 创建..."
        docker compose up -d minio
    }
}

Start-Infrastructure

Write-Host "==> 检查并安装依赖 (Poetry, --no-root)..."
poetry install --no-root

Write-Host "==> 启动后端服务..."
poetry run python run.py

Param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "==> 停止 MySQL 容器..."
docker compose stop mysql | Out-Null

$minioExists = docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch "minio-server"
if ($minioExists) {
    Write-Host "==> 停止现有 minio-server 容器..."
    docker stop minio-server | Out-Null
} else {
    Write-Host "==> 停止 compose 管理的 MinIO 容器..."
    docker compose stop minio | Out-Null
}

Write-Host "==> 已停止。"

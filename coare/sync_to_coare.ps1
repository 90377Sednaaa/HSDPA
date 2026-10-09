<#
.SYNOPSIS
    Syncs the local HSDPA repository to DOST-ASTI COARE scratch storage.
.DESCRIPTION
    Uploads the HSDPA codebase to /scratch3/lean.murillo/HSDPA via scp.
#>

param (
    [string]$RemoteHost = "saliksik",
    [string]$RemoteUser = "lean.murillo",
    [string]$RemoteDest = "/scratch3/lean.murillo/HSDPA"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Syncing HSDPA Codebase to COARE HPC (Saliksik)" -ForegroundColor Cyan
Write-Host "Source     : $RepoRoot"
Write-Host "Destination: ${RemoteHost}:${RemoteDest}"
Write-Host "==========================================================" -ForegroundColor Cyan

# Test SSH connection first
Write-Host "Checking connection to $RemoteHost..." -ForegroundColor Yellow
$test = ssh -o BatchMode=yes -o ConnectTimeout=5 $RemoteHost "echo connected" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Cannot reach $RemoteHost yet. (Cluster may still be synchronizing user provisioning)." -ForegroundColor Red
    Write-Host "Please ensure your account has completed setup before syncing." -ForegroundColor Yellow
    exit 1
}

# Create remote directory
ssh $RemoteHost "mkdir -p $RemoteDest"

# Use scp to upload essential directories
Write-Host "Uploading codebase (cfg, ultralytics, coare, setup.py)..." -ForegroundColor Green
scp -r "$RepoRoot\cfg" "${RemoteHost}:${RemoteDest}/"
scp -r "$RepoRoot\ultralytics" "${RemoteHost}:${RemoteDest}/"
scp -r "$RepoRoot\coare" "${RemoteHost}:${RemoteDest}/"
scp "$RepoRoot\setup.py" "${RemoteHost}:${RemoteDest}/"

Write-Host "Codebase successfully uploaded to ${RemoteDest}!" -ForegroundColor Green

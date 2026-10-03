# Push a Kaggle kernel from this repo (Kaggle CLI 2.x).
#
# The CLI only accepts: kaggle kernels push -p FOLDER
# where FOLDER contains the notebook plus a file named EXACTLY
# kernel-metadata.json. This script stages those two files into a temp
# dir (repo stays single-source) and pushes.
#
# Usage (run from the kaggle/ folder, token already set in this shell):
#   $env:KAGGLE_API_TOKEN = "<PASTE-...-KGAT-TOKEN>"
#   .\push.ps1 r2000-baseline
#
# Keys:
#   r2000-baseline | r2000-cr050 | r2000-cr050-k3 | r050-s1 | r050-s2 | r050-s3 | r050-s4
param([Parameter(Mandatory=$true)][string]$Kernel)

$map = @{
  "r2000-baseline" = @("train_r2000_baseline_200ep.ipynb", "kernel-metadata-r2000-baseline.json")
  "r2000-cr050"    = @("train_r2000_cr050_200ep.ipynb", "kernel-metadata-r2000-cr050.json")
  "r2000-cr050-k3" = @("train_r2000_cr050_k3_200ep.ipynb", "kernel-metadata-r2000-cr050-k3.json")
  "r050-s1"        = @("train_r050_stage1.ipynb", "kernel-metadata-r050-s1.json")
  "r050-s2"        = @("train_r050_stage2.ipynb", "kernel-metadata-r050-s2.json")
  "r050-s3"        = @("train_r050_stage3.ipynb", "kernel-metadata-r050-s3.json")
  "r050-s4"        = @("train_r050_stage4.ipynb", "kernel-metadata-r050-s4.json")
}

if (-not $map.ContainsKey($Kernel)) {
  Write-Error "Unknown kernel '$Kernel'. Valid keys: $($map.Keys -join ', ')"
  exit 1
}
if (-not $env:KAGGLE_API_TOKEN) {
  Write-Error "KAGGLE_API_TOKEN is not set in this shell. Set it first: `$env:KAGGLE_API_TOKEN = ""<PASTE-...-KGAT-TOKEN>"""
  exit 1
}

$nb, $meta = $map[$Kernel]
foreach ($f in @($nb, $meta)) {
  if (-not (Test-Path -LiteralPath $f)) { Write-Error "Missing file: $f (run from the kaggle/ folder)"; exit 1 }
}

$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("kpush_" + $Kernel)
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath $nb -Destination (Join-Path $stage $nb)
Copy-Item -LiteralPath $meta -Destination (Join-Path $stage "kernel-metadata.json")

Write-Output "Staged $nb + kernel-metadata.json in $stage"
kaggle kernels push -p $stage

# setup_windows.ps1 — Noor training environment on Windows
# Run in PowerShell as Administrator.
#
# What this script does:
#   1. Installs Rust (if not present)
#   2. Downloads OpenBLAS pre-built binaries
#   3. Sets OPENBLAS_PATH user environment variable
#   4. Builds noor-cli in release mode

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== Noor Windows Setup ===" -ForegroundColor Green

# ── 1. Rust ─────────────────────────────────────────────────────────────────
if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "Rust not found — installing rustup..." -ForegroundColor Yellow
    $rustupInstaller = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInstaller
    & $rustupInstaller -y --default-toolchain stable
    $env:PATH += ";$env:USERPROFILE\.cargo\bin"
    Write-Host "Rust installed." -ForegroundColor Green
} else {
    Write-Host "Rust already installed: $(cargo --version)" -ForegroundColor Green
}

# ── 2. OpenBLAS ─────────────────────────────────────────────────────────────
$openblasVersion = "0.3.28"
$openblasUrl     = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${openblasVersion}/OpenBLAS-${openblasVersion}-x64.zip"
$openblasRoot    = "$env:USERPROFILE\.noor\openblas"
$openblasLib     = "$openblasRoot\lib"

if (!(Test-Path "$openblasLib\openblas.lib") -and !(Test-Path "$openblasLib\libopenblas.lib")) {
    Write-Host "Downloading OpenBLAS $openblasVersion..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $openblasRoot | Out-Null
    $zipPath = "$env:TEMP\openblas.zip"
    Invoke-WebRequest -Uri $openblasUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $openblasRoot -Force
    Remove-Item $zipPath

    # The zip may unpack into a versioned sub-directory — find the lib folder.
    $libCandidate = Get-ChildItem -Path $openblasRoot -Recurse -Filter "openblas.lib" |
                    Select-Object -First 1
    if ($libCandidate) {
        $openblasLib = $libCandidate.DirectoryName
    }
    Write-Host "OpenBLAS installed to: $openblasLib" -ForegroundColor Green
} else {
    Write-Host "OpenBLAS already present at: $openblasLib" -ForegroundColor Green
}

# ── 3. Set environment variable ──────────────────────────────────────────────
$env:OPENBLAS_PATH = $openblasLib
[System.Environment]::SetEnvironmentVariable("OPENBLAS_PATH", $openblasLib, "User")
Write-Host "OPENBLAS_PATH = $openblasLib (saved to user environment)" -ForegroundColor Green

# Also add OpenBLAS DLL directory to PATH so the binary can find openblas.dll at runtime.
$openblasBin = Join-Path (Split-Path $openblasLib -Parent) "bin"
if (Test-Path $openblasBin) {
    $currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$openblasBin*") {
        [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$openblasBin", "User")
        $env:PATH += ";$openblasBin"
        Write-Host "Added $openblasBin to PATH" -ForegroundColor Green
    }
}

# ── 4. Build Noor ────────────────────────────────────────────────────────────
Write-Host "Building noor-cli (release)..." -ForegroundColor Yellow
cargo build --release -p noor-cli

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Run: cargo run --release -p noor-cli -- --help" -ForegroundColor Cyan

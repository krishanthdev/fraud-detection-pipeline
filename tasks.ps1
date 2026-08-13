<#
.SYNOPSIS
    Windows task runner. The same targets as the Makefile, for machines without make.

.DESCRIPTION
    The Makefile is the reference runner and it is what CI and the Docker image use.
    Windows does not ship make, so this script mirrors the same targets so the pipeline
    can be driven the same way on this laptop.

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 test
    .\tasks.ps1 all
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = "help",

    [string]$Config = "configs/config.yaml"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Get-Python {
    if (Test-Path $VenvPython) { return $VenvPython }
    return "python"
}

function Invoke-Fraud {
    param([string[]]$FraudArgs)
    $py = Get-Python
    & $py -m fraud_pipeline --config $Config @FraudArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Tool {
    param([string]$Tool, [string[]]$ToolArgs)
    $py = Get-Python
    & $py -m $Tool @ToolArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target.ToLower()) {
    "help" {
        Write-Output ""
        Write-Output "Usage: .\tasks.ps1 <target>"
        Write-Output ""
        Write-Output "  setup         Create .venv and install everything"
        Write-Output "  install       Install dependencies into the current environment"
        Write-Output "  lint          Run ruff checks"
        Write-Output "  format        Reformat the code with ruff"
        Write-Output "  test          Run the full test suite"
        Write-Output "  test-fast     Run only the quick tests"
        Write-Output ""
        Write-Output "  ingest        Stage 1"
        Write-Output "  validate      Stage 2"
        Write-Output "  features      Stage 3"
        Write-Output "  train         Stage 4"
        Write-Output "  evaluate      Stage 5"
        Write-Output "  register      Stage 6"
        Write-Output "  all           Stages 1 to 6"
        Write-Output "  serve         Stage 7, FastAPI"
        Write-Output "  app           Stage 7, Streamlit"
        Write-Output ""
        Write-Output "  docker-build  Stage 8, build the images"
        Write-Output "  docker-up     Start everything in Docker"
        Write-Output "  docker-down   Stop the containers"
        Write-Output "  clean         Remove caches and generated artifacts"
        Write-Output ""
    }
    "setup" {
        $base = (Get-Command py -ErrorAction SilentlyContinue)
        if ($base) { & py -3.12 -m venv $Venv } else { & python -m venv $Venv }
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r (Join-Path $Root "requirements-dev.txt")
        & $VenvPython -m pip install -e $Root
    }
    "install" {
        $py = Get-Python
        & $py -m pip install -r (Join-Path $Root "requirements-dev.txt")
        & $py -m pip install -e $Root
    }
    "lint" {
        Invoke-Tool ruff @("check", "src", "tests", "app")
        Invoke-Tool ruff @("format", "--check", "src", "tests", "app")
    }
    "format" {
        Invoke-Tool ruff @("format", "src", "tests", "app")
        Invoke-Tool ruff @("check", "--fix", "src", "tests", "app")
    }
    "test"      { Invoke-Tool pytest @() }
    "test-fast" { Invoke-Tool pytest @("-m", "not slow and not needs_data") }

    "ingest"    { Invoke-Fraud @("ingest") }
    "validate"  { Invoke-Fraud @("validate") }
    "features"  { Invoke-Fraud @("features") }
    "train"     { Invoke-Fraud @("train") }
    "evaluate"  { Invoke-Fraud @("evaluate") }
    "register"  { Invoke-Fraud @("register") }
    "all"       { Invoke-Fraud @("run-all") }
    "serve"     { Invoke-Fraud @("serve") }
    "app"       { Invoke-Tool streamlit @("run", (Join-Path $Root "app\streamlit_app.py")) }

    "docker-build" { & docker compose build }
    "docker-up"    { & docker compose up }
    "docker-down"  { & docker compose down }

    "clean" {
        foreach ($item in @(".pytest_cache", ".ruff_cache", ".coverage", "htmlcov", "coverage.xml")) {
            $path = Join-Path $Root $item
            if (Test-Path $path) { Remove-Item -Recurse -Force $path }
        }
        foreach ($dir in @("data\interim", "data\processed", "models")) {
            $path = Join-Path $Root $dir
            if (Test-Path $path) {
                Get-ChildItem $path -Exclude ".gitkeep" | Remove-Item -Recurse -Force
            }
        }
        Get-ChildItem $Root -Recurse -Directory -Filter "__pycache__" |
            Remove-Item -Recurse -Force
        Write-Output "Cleaned. Raw data was left alone."
    }
    default {
        Write-Error "Unknown target '$Target'. Run .\tasks.ps1 help for the list."
        exit 1
    }
}

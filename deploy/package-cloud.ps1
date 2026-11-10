param(
    [string]$OutputPath = ".tmp\moneymore-cloud.tar.gz"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath
} else {
    Join-Path $ProjectRoot $OutputPath
}
$OutputDirectory = Split-Path -Parent $ResolvedOutput

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
if (Test-Path -LiteralPath $ResolvedOutput) {
    Remove-Item -LiteralPath $ResolvedOutput -Force
}

$RequiredPaths = @(
    "apps",
    "configs",
    "deploy",
    "docs",
    "pyproject.toml",
    "README.md",
    "scripts",
    "src",
    "data\processed"
)
foreach ($RelativePath in $RequiredPaths) {
    $FullPath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath)) {
        throw "Required deployment path is missing: $FullPath"
    }
}

$TarArguments = @(
    "-czf", $ResolvedOutput,
    "--exclude=.git",
    "--exclude=*/.git",
    "--exclude=apps/dashboard/node_modules",
    "--exclude=apps/dashboard/.vinext",
    "--exclude=apps/dashboard/.next",
    "--exclude=apps/dashboard/.wrangler",
    "--exclude=apps/dashboard/build",
    "--exclude=apps/dashboard/dist",
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=.pytest_cache",
    "--exclude=.env",
    "--exclude=.env.*",
    "apps",
    "configs",
    "deploy",
    "docs",
    "pyproject.toml",
    "README.md",
    "scripts",
    "src",
    "data/processed"
)

foreach ($PaperPath in @("data/us_overnight_paper_only/open2open_v1/plans", "data/us_overnight_paper_only/universe")) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $PaperPath)) {
        $TarArguments += $PaperPath
    }
}

Push-Location $ProjectRoot
try {
    & tar @TarArguments
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

$Archive = Get-Item -LiteralPath $ResolvedOutput
Write-Host ("Created {0} ({1:N1} MB)" -f $Archive.FullName, ($Archive.Length / 1MB))
Write-Host "Provision server secrets and migrate authorized runtime data separately."

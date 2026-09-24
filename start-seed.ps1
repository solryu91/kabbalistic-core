[CmdletBinding()]
param(
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$sourceRoot = Join-Path $projectRoot "src"
$uiRoot = Join-Path $projectRoot "ui"
$vinextCli = Join-Path $uiRoot "node_modules\vinext\dist\cli.js"
$apiPort = 8765
$uiPort = 3000
$modelPort = 1234
$apiUrl = "http://127.0.0.1:$apiPort"
$uiUrl = "http://127.0.0.1:$uiPort"
$modelUrl = "http://127.0.0.1:$modelPort/v1"
$modelAlias = "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

function Add-RuntimeCandidate {
    param(
        [System.Collections.Generic.List[object]]$List,
        [string]$Path,
        [string[]]$PrefixArguments = @()
    )

    if (-not [string]::IsNullOrWhiteSpace($Path)) {
        [void]$List.Add([pscustomobject]@{
            Path = $Path.Trim('"')
            PrefixArguments = @($PrefixArguments)
        })
    }
}

function Resolve-SeedPython {
    $candidates = [System.Collections.Generic.List[object]]::new()

    Add-RuntimeCandidate $candidates ([Environment]::GetEnvironmentVariable("SEED_PYTHON", "Process"))
    Add-RuntimeCandidate $candidates (Join-Path $projectRoot ".venv\Scripts\python.exe")

    $runtimeRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes"
    if (Test-Path -LiteralPath $runtimeRoot) {
        $bundled = Get-ChildItem -LiteralPath $runtimeRoot -Filter "python.exe" -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\dependencies\\python\\python\.exe$' } |
            Sort-Object LastWriteTimeUtc -Descending
        foreach ($item in $bundled) {
            Add-RuntimeCandidate $candidates $item.FullName
        }
    }

    $py = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        Add-RuntimeCandidate $candidates $py.Source @("-3.12")
    }
    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        Add-RuntimeCandidate $candidates $python.Source
    }

    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($candidate in $candidates) {
        $key = "$($candidate.Path)|$($candidate.PrefixArguments -join ' ')"
        if (-not $seen.Add($key)) {
            continue
        }
        if (-not (Test-Path -LiteralPath $candidate.Path)) {
            continue
        }
        try {
            $prefix = @($candidate.PrefixArguments)
            $versionText = ((& $candidate.Path @prefix --version 2>&1) | Out-String).Trim()
            if ($versionText -match 'Python\s+(\d+)\.(\d+)\.(\d+)') {
                $version = [version]::Parse("$($Matches[1]).$($Matches[2]).$($Matches[3])")
                if ($version -ge [version]"3.12.0") {
                    return [pscustomobject]@{
                        Path = $candidate.Path
                        PrefixArguments = $prefix
                        Version = $versionText
                    }
                }
            }
        }
        catch {
            continue
        }
    }

    throw "Python 3.12 or newer was not found. Set SEED_PYTHON to a trusted local python.exe or install a compatible runtime."
}

function Resolve-SeedNode {
    $candidates = [System.Collections.Generic.List[object]]::new()

    Add-RuntimeCandidate $candidates ([Environment]::GetEnvironmentVariable("SEED_NODE", "Process"))

    $runtimeRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes"
    if (Test-Path -LiteralPath $runtimeRoot) {
        $bundled = Get-ChildItem -LiteralPath $runtimeRoot -Filter "node.exe" -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\dependencies\\node\\bin\\node\.exe$' } |
            Sort-Object LastWriteTimeUtc -Descending
        foreach ($item in $bundled) {
            Add-RuntimeCandidate $candidates $item.FullName
        }
    }

    $node = Get-Command "node.exe" -ErrorAction SilentlyContinue
    if ($null -ne $node) {
        Add-RuntimeCandidate $candidates $node.Source
    }

    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($candidate in $candidates) {
        if (-not $seen.Add($candidate.Path)) {
            continue
        }
        if (-not (Test-Path -LiteralPath $candidate.Path)) {
            continue
        }
        try {
            $versionText = ((& $candidate.Path --version 2>&1) | Out-String).Trim()
            if ($versionText -match '^v?(\d+)\.(\d+)\.(\d+)') {
                $version = [version]::Parse("$($Matches[1]).$($Matches[2]).$($Matches[3])")
                if ($version -ge [version]"22.13.0") {
                    return [pscustomobject]@{
                        Path = $candidate.Path
                        Version = $versionText
                    }
                }
            }
        }
        catch {
            continue
        }
    }

    throw "Node.js 22.13 or newer was not found. Set SEED_NODE to a trusted local node.exe or install a compatible runtime."
}

function Resolve-SeedModelServer {
    $explicit = [Environment]::GetEnvironmentVariable("SEED_LLAMA_SERVER", "Process")
    if (-not [string]::IsNullOrWhiteSpace($explicit) -and (Test-Path -LiteralPath $explicit -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $explicit).Path
    }

    $lmStudioRoot = Join-Path $env:USERPROFILE ".lmstudio\extensions\backends"
    if (Test-Path -LiteralPath $lmStudioRoot) {
        $bundled = Get-ChildItem -LiteralPath $lmStudioRoot -Filter "llama-server.exe" -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match 'llama\.cpp-win-x86_64-vulkan-avx2-[^\\]+\\llama-server\.exe$' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($null -ne $bundled) {
            return $bundled.FullName
        }
    }

    $command = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    return $null
}

function Resolve-SeedQwenModel {
    $explicit = [Environment]::GetEnvironmentVariable("SEED_QWEN_MODEL", "Process")
    if (-not [string]::IsNullOrWhiteSpace($explicit) -and (Test-Path -LiteralPath $explicit -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $explicit).Path
    }

    $preferred = Join-Path $env:USERPROFILE ".lmstudio\models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf"
    if (Test-Path -LiteralPath $preferred -PathType Leaf) {
        return $preferred
    }

    $modelRoot = Join-Path $env:USERPROFILE ".lmstudio\models"
    if (Test-Path -LiteralPath $modelRoot) {
        $candidate = Get-ChildItem -LiteralPath $modelRoot -Filter "*.gguf" -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match 'Qwen3.*8B.*Q4_K_M' } |
            Sort-Object FullName |
            Select-Object -First 1
        if ($null -ne $candidate) {
            return $candidate.FullName
        }
    }
    return $null
}

function Test-SeedModelReady {
    param([string]$ModelsUrl)

    try {
        $response = Invoke-RestMethod -Method Get -Uri $ModelsUrl -TimeoutSec 2
        return @($response.data | Where-Object {
            $null -ne $_.id -and $_.id.ToString() -match '(?i)qwen3' -and $_.id.ToString() -match '(?i)8b'
        }).Count -gt 0
    }
    catch {
        return $false
    }
}

function Wait-SeedModel {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$ModelsUrl,
        [string]$ErrorLog
    )

    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if ($Process.HasExited) {
            throw "The local Qwen helper exited before becoming ready. Read $ErrorLog"
        }
        if (Test-SeedModelReady $ModelsUrl) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "The local Qwen helper did not become ready within 90 seconds. Read $ErrorLog"
}

function Test-LoopbackPortInUse {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $pending = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(250)) {
            return $false
        }
        $client.EndConnect($pending)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-SeedBackend {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$HealthUrl,
        [string]$ErrorLog
    )

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($Process.HasExited) {
            throw "The SEED backend exited before becoming ready. Read $ErrorLog"
        }
        try {
            $health = Invoke-RestMethod -Method Get -Uri $HealthUrl -TimeoutSec 2
            if ($health.service -like "seed-poc:*") {
                return $health
            }
        }
        catch {
            # Readiness is retried within the bounded startup window.
        }
        Start-Sleep -Milliseconds 500
    }

    throw "The SEED backend did not become ready on loopback. Read $ErrorLog"
}

function Wait-SeedInterface {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Url,
        [string]$ErrorLog
    )

    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($Process.HasExited) {
            throw "The SEED interface exited before becoming ready. Read $ErrorLog"
        }
        try {
            $response = Invoke-WebRequest -Method Get -Uri $Url -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        }
        catch {
            # Readiness is retried within the bounded startup window.
        }
        Start-Sleep -Milliseconds 500
    }

    throw "The SEED interface did not become ready on loopback. Read $ErrorLog"
}

if (-not (Test-Path -LiteralPath $vinextCli)) {
    throw "The local UI dependencies are missing. Restore ui/node_modules before running this launcher; it will not install or download dependencies."
}

$occupiedPorts = @(
    foreach ($port in @($apiPort, $uiPort)) {
        if (Test-LoopbackPortInUse $port) {
            $port
        }
    }
)
if ($occupiedPorts.Count -gt 0 -and -not $CheckOnly) {
    throw "Loopback port(s) $($occupiedPorts -join ', ') are already in use. The launcher will not stop or replace an existing listener."
}

$python = Resolve-SeedPython
$node = Resolve-SeedNode
$modelServer = Resolve-SeedModelServer
$qwenModel = Resolve-SeedQwenModel
$modelPortInUse = Test-LoopbackPortInUse $modelPort
$modelAlreadyReady = Test-SeedModelReady "$modelUrl/models"

if ($CheckOnly) {
    Write-Host "Python: $($python.Version)"
    Write-Host "Node: $($node.Version)"
    if ($occupiedPorts.Count -eq 0) {
        Write-Host "Launcher prerequisites and fixed loopback ports are ready."
    }
    else {
        Write-Warning "Prerequisites are available, but loopback port(s) $($occupiedPorts -join ', ') are currently occupied."
    }
    if ($modelAlreadyReady) {
        Write-Host "Qwen3 8B: already available at $modelUrl"
    }
    elseif ($modelPortInUse) {
        Write-Warning "Port $modelPort is occupied by a service that did not report Qwen3 8B; the launcher will leave it untouched and SEED may use fallback."
    }
    elseif ($null -ne $modelServer -and $null -ne $qwenModel) {
        Write-Host "Qwen3 8B: local model helper and GGUF are ready for automatic loopback startup."
    }
    else {
        Write-Warning "Qwen3 8B auto-start prerequisites were not found; the deterministic fallback remains available."
    }
    Write-Host "No helper process was started."
    return
}

$logRoot = Join-Path $projectRoot "runs\launcher"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$backendOut = Join-Path $logRoot "backend-$stamp.out.log"
$backendErr = Join-Path $logRoot "backend-$stamp.err.log"
$uiOut = Join-Path $logRoot "interface-$stamp.out.log"
$uiErr = Join-Path $logRoot "interface-$stamp.err.log"
$modelOut = Join-Path $logRoot "model-$stamp.out.log"
$modelErr = Join-Path $logRoot "model-$stamp.err.log"

Write-Host "Starting the public-edition, loopback-only SEED POC."
Write-Host "Python: $($python.Version)"
Write-Host "Node: $($node.Version)"

$modelProcess = $null
if ($modelAlreadyReady) {
    Write-Host "Using the existing loopback Qwen3 8B service at $modelUrl"
}
elseif ($modelPortInUse) {
    Write-Warning "Port $modelPort is already occupied by a non-matching service. It will not be replaced; SEED will report its actual model or fallback status."
}
elseif ($null -ne $modelServer -and $null -ne $qwenModel) {
    $modelArguments = @(
        "--model", $qwenModel,
        "--alias", $modelAlias,
        "--host", "127.0.0.1",
        "--port", "$modelPort",
        "--ctx-size", "16384",
        "--gpu-layers", "all",
        "--flash-attn", "on",
        "--reasoning", "off",
        "--threads-http", "1",
        "--offline",
        "--no-webui",
        "--cors-origins", "localhost"
    )
    $modelStart = @{
        FilePath = $modelServer
        ArgumentList = $modelArguments
        WorkingDirectory = $projectRoot
        WindowStyle = "Hidden"
        RedirectStandardOutput = $modelOut
        RedirectStandardError = $modelErr
        PassThru = $true
    }
    $modelProcess = Start-Process @modelStart
    Write-Host "Local Qwen helper PID: $($modelProcess.Id)"
    Wait-SeedModel $modelProcess "$modelUrl/models" $modelErr
}
else {
    Write-Warning "A local Qwen helper or Qwen3 8B GGUF was not found. The visibly labeled deterministic fallback remains available."
}

$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$seedPythonPath = if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $sourceRoot
}
else {
    "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
}

try {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $seedPythonPath, "Process")
    $backendArguments = @($python.PrefixArguments) + @(
        "-m",
        "kabbalistic_core.poc_server",
        "--host",
        "127.0.0.1",
        "--port",
        "$apiPort"
    )
    $backendStart = @{
        FilePath = $python.Path
        ArgumentList = $backendArguments
        WorkingDirectory = $projectRoot
        WindowStyle = "Hidden"
        RedirectStandardOutput = $backendOut
        RedirectStandardError = $backendErr
        PassThru = $true
    }
    $backendProcess = Start-Process @backendStart
}
finally {
    [Environment]::SetEnvironmentVariable("PYTHONPATH", $previousPythonPath, "Process")
}

Write-Host "Backend helper PID: $($backendProcess.Id)"
$health = Wait-SeedBackend $backendProcess "$apiUrl/api/health" $backendErr

$interfaceArguments = @(
    "node_modules/vinext/dist/cli.js",
    "dev",
    "--hostname",
    "127.0.0.1",
    "--port",
    "$uiPort"
)
$interfaceStart = @{
    FilePath = $node.Path
    ArgumentList = $interfaceArguments
    WorkingDirectory = $uiRoot
    WindowStyle = "Hidden"
    RedirectStandardOutput = $uiOut
    RedirectStandardError = $uiErr
    PassThru = $true
}
$interfaceProcess = Start-Process @interfaceStart

Write-Host "Interface helper PID: $($interfaceProcess.Id)"
Wait-SeedInterface $interfaceProcess $uiUrl $uiErr

$modelStatus = if ($null -ne $health.model) { $health.model.status } else { "unknown" }
Write-Host "Local model status: $modelStatus"
Write-Host "Backend logs: $backendOut and $backendErr"
Write-Host "Interface logs: $uiOut and $uiErr"
if ($null -ne $modelProcess) {
    Write-Host "Model logs: $modelOut and $modelErr"
}
Write-Host "Opening the visible guided interface at $uiUrl"

Start-Process -FilePath $uiUrl | Out-Null

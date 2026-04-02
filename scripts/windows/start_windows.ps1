[CmdletBinding()]
param(
    [switch]$SetupOnly,
    [switch]$InstallBackendDeps,
    [switch]$ForceNodeRefresh,
    [switch]$NoBrowser,
    [int]$Port = 12000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$WindowsCacheDir = Join-Path $RepoRoot ".windows-cache"
$WindowsToolsDir = Join-Path $RepoRoot ".windows-tools"
$WorkspaceDir = Join-Path $RepoRoot "workspace"
$LogsDir = Join-Path $RepoRoot "logs"
$FrontendDir = Join-Path $RepoRoot "frontend"
$PythonInstallerVersion = "3.12.10"
$PortableNodeRoot = Join-Path $WindowsToolsDir "node"
$PortableNodeExe = Join-Path $PortableNodeRoot "node.exe"
$PortableNpmCmd = Join-Path $PortableNodeRoot "npm.cmd"
$FrontendStamp = Join-Path $WindowsCacheDir "frontend-install.stamp"
$BackendStamp = Join-Path $WindowsCacheDir "backend-install.stamp"
$RequiredNodeVersion = [Version]"22.12.0"
$UiUrl = "http://127.0.0.1:$Port/"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Note {
    param([string]$Message)
    Write-Host "[OpenHands] $Message" -ForegroundColor Gray
}

function Write-WarnLine {
    param([string]$Message)
    Write-Host "[OpenHands] $Message" -ForegroundColor Yellow
}

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory
    )

    $location = if ($WorkingDirectory) { $WorkingDirectory } else { (Get-Location).Path }
    Push-Location $location
    try {
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -ne 0) {
        $joinedArguments = $Arguments -join " "
        throw ("Command failed with exit code {0}: {1} {2}" -f $exitCode, $FilePath, $joinedArguments)
    }
}

function Invoke-FileDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    Write-Note "Downloading $Uri"
    Invoke-WebRequest -Uri $Uri -OutFile $Destination
}

function Test-HttpEndpoint {
    param([string]$Uri)
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Start-BrowserWhenReady {
    param([string]$Uri)

    if ($NoBrowser) {
        return
    }

    $script = @"
for (`$i = 0; `$i -lt 180; `$i++) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri '$Uri' -TimeoutSec 2 | Out-Null
        Start-Process '$Uri'
        exit 0
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
"@

    Start-Process -WindowStyle Hidden -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $script
    ) | Out-Null
}

function Get-Python312Exe {
    try {
        $resolved = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
    }
    catch {
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Python312\python.exe")
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Ensure-Python312 {
    $pythonExe = Get-Python312Exe
    if ($pythonExe) {
        Write-Note "Using Python 3.12 at $pythonExe"
        return $pythonExe
    }

    Write-Step "Installing Python 3.12"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --exact --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
        Start-Sleep -Seconds 5
        $pythonExe = Get-Python312Exe
        if ($pythonExe) {
            Write-Note "Python 3.12 installed via winget"
            return $pythonExe
        }
        Write-WarnLine "winget completed but Python was not yet visible in PATH. Falling back to the official installer."
    }

    $pythonInstaller = Join-Path $WindowsCacheDir "python-$PythonInstallerVersion-amd64.exe"
    if (-not (Test-Path $pythonInstaller)) {
        Invoke-FileDownload -Uri "https://www.python.org/ftp/python/$PythonInstallerVersion/python-$PythonInstallerVersion-amd64.exe" -Destination $pythonInstaller
    }

    $process = Start-Process -FilePath $pythonInstaller -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=1",
        "Include_test=0",
        "Include_launcher=1",
        "SimpleInstall=1"
    ) -PassThru -Wait

    if ($process.ExitCode -ne 0) {
        throw "Python installer exited with code $($process.ExitCode)"
    }

    Start-Sleep -Seconds 5
    $pythonExe = Get-Python312Exe
    if (-not $pythonExe) {
        throw "Python 3.12 installation finished, but python.exe could not be located."
    }

    Write-Note "Python 3.12 installed at $pythonExe"
    return $pythonExe
}

function Get-NodeVersion {
    param([string]$NodeExecutable)

    if (-not (Test-Path $NodeExecutable)) {
        return $null
    }

    try {
        $raw = & $NodeExecutable -v 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            return $null
        }
        return [Version]($raw.Trim().TrimStart("v"))
    }
    catch {
        return $null
    }
}

function Ensure-PortableNode {
    if ($ForceNodeRefresh -and (Test-Path $PortableNodeRoot)) {
        Remove-Item -LiteralPath $PortableNodeRoot -Recurse -Force
    }

    $portableVersion = Get-NodeVersion -NodeExecutable $PortableNodeExe
    if ($portableVersion -and $portableVersion -ge $RequiredNodeVersion) {
        Write-Note "Using bundled Node.js $portableVersion"
        return $PortableNodeExe
    }

    Write-Step "Downloading portable Node.js"
    $index = Invoke-RestMethod -Uri "https://nodejs.org/dist/index.json"
    $nodeRelease = $index |
        Where-Object {
            $_.lts -ne $false -and [Version]($_.version.TrimStart("v")) -ge $RequiredNodeVersion
        } |
        Select-Object -First 1

    if (-not $nodeRelease) {
        throw "Could not find a suitable Node.js release for Windows."
    }

    $nodeVersion = $nodeRelease.version
    $nodeZip = Join-Path $WindowsCacheDir ("node-{0}-win-x64.zip" -f $nodeVersion)
    if (-not (Test-Path $nodeZip)) {
        Invoke-FileDownload -Uri ("https://nodejs.org/dist/{0}/node-{0}-win-x64.zip" -f $nodeVersion) -Destination $nodeZip
    }

    $expandedNodeDir = Join-Path $WindowsToolsDir ("node-{0}-win-x64" -f $nodeVersion)
    if (Test-Path $expandedNodeDir) {
        Remove-Item -LiteralPath $expandedNodeDir -Recurse -Force
    }

    Expand-Archive -Path $nodeZip -DestinationPath $WindowsToolsDir -Force

    if (Test-Path $PortableNodeRoot) {
        Remove-Item -LiteralPath $PortableNodeRoot -Recurse -Force
    }
    Rename-Item -LiteralPath $expandedNodeDir -NewName "node"

    $portableVersion = Get-NodeVersion -NodeExecutable $PortableNodeExe
    if (-not $portableVersion) {
        throw "Portable Node.js download completed, but node.exe could not be executed."
    }

    Write-Note "Bundled Node.js $portableVersion"
    return $PortableNodeExe
}

function Ensure-Poetry {
    param([string]$PythonExe)

    try {
        & $PythonExe -m poetry --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Note "Poetry already available for Python 3.12"
            return
        }
    }
    catch {
    }

    Write-Step "Installing Poetry"
    Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--user", "--upgrade", "pip", "poetry>=2.1.2")
}

function Get-PathStampNeedsRefresh {
    param(
        [Parameter(Mandatory = $true)][string]$StampPath,
        [Parameter(Mandatory = $true)][string]$ReferencePath
    )

    if (-not (Test-Path $StampPath)) {
        return $true
    }

    return (Get-Item $StampPath).LastWriteTimeUtc -lt (Get-Item $ReferencePath).LastWriteTimeUtc
}

function Install-FrontendDependencies {
    param([string]$NpmCmd)

    $needsInstall = -not (Test-Path (Join-Path $FrontendDir "node_modules")) -or
        (Get-PathStampNeedsRefresh -StampPath $FrontendStamp -ReferencePath (Join-Path $FrontendDir "package-lock.json"))

    if (-not $needsInstall) {
        Write-Note "Frontend dependencies are already installed"
        return
    }

    Write-Step "Installing frontend dependencies"
    Invoke-Checked -FilePath $NpmCmd -Arguments @("install", "--no-fund", "--no-audit") -WorkingDirectory $FrontendDir
    [System.IO.File]::WriteAllText($FrontendStamp, (Get-Date).ToString("O"), [System.Text.Encoding]::ASCII)
}

function Install-BackendDependencies {
    param([string]$PythonExe)

    $referencePath = Join-Path $RepoRoot "poetry.lock"
    $needsInstall = Get-PathStampNeedsRefresh -StampPath $BackendStamp -ReferencePath $referencePath
    if (-not $needsInstall) {
        Write-Note "Python dependencies are already installed"
        return
    }

    Write-Step "Installing backend Python dependencies"
    $env:TZ = "UTC"
    $env:INSTALL_PLAYWRIGHT = "false"
    $env:PIP_DEFAULT_TIMEOUT = "180"

    Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "poetry", "env", "use", $PythonExe) -WorkingDirectory $RepoRoot
    Invoke-Checked -FilePath $PythonExe -Arguments @("-m", "poetry", "install", "--with", "dev,test,runtime") -WorkingDirectory $RepoRoot
    [System.IO.File]::WriteAllText($BackendStamp, (Get-Date).ToString("O"), [System.Text.Encoding]::ASCII)
}

Ensure-Directory $WindowsCacheDir
Ensure-Directory $WindowsToolsDir
Ensure-Directory $WorkspaceDir
Ensure-Directory $LogsDir

Write-Host "OpenHands Windows Easy Start" -ForegroundColor Green
Write-Host "Native Windows mode launches the mock frontend for UI testing." -ForegroundColor Green
Write-Host "Full backend/runtime still requires WSL or Linux." -ForegroundColor Yellow

$pythonExe = Ensure-Python312
$nodeExe = Ensure-PortableNode
$nodeDir = Split-Path -Parent $nodeExe
$env:PATH = "$nodeDir;$env:PATH"
Ensure-Poetry -PythonExe $pythonExe
Install-FrontendDependencies -NpmCmd $PortableNpmCmd

if ($InstallBackendDeps) {
    Install-BackendDependencies -PythonExe $pythonExe
}

if ($SetupOnly) {
    Write-Step "Setup completed"
    Write-Host "Dependencies are ready. Start the mock UI later with Start-Windows.bat" -ForegroundColor Green
    exit 0
}

if (Test-HttpEndpoint -Uri $UiUrl) {
    Write-Note "OpenHands mock UI is already running at $UiUrl"
    if (-not $NoBrowser) {
        Start-Process $UiUrl
    }
    exit 0
}

Write-Step "Starting mock frontend on $UiUrl"
Start-BrowserWhenReady -Uri $UiUrl
Push-Location $FrontendDir
try {
    & $PortableNpmCmd "run" "dev:mock" "--" "--host" "127.0.0.1" "--port" "$Port"
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode

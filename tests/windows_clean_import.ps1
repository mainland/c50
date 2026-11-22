# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Prove that a Windows wheel needs the Visual C++ Redistributable and works
# with it. The wheel must bundle no Microsoft runtime DLL, and on a system
# without the redistributable it must fail to import. After the script installs
# the redistributable, the wheel must import, use the system msvcp140.dll,
# and train. Last, an msvcp140.dll older than 14.38, which the wheel's mutexes
# must tolerate, is planted in three ways and the wheel must lock a mutex in
# each. Run it in a Windows Server Core container with these directories
# mounted:
#
#   C:\wheelhouse  holds the wheels, read-only.
#   C:\embed       holds an unpacked embeddable CPython, read-only.
#   C:\tests       holds this script and windows_clean_import.py, read-only.
#   C:\vcredist    holds vc_redist.x64.exe, read-only.
#   C:\stale       holds an msvcp140.dll older than 14.38, read-only.
#
# Usage: windows_clean_import.ps1 -Tag cp312

param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^cp3[0-9]+$')]
    [string]$Tag
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# The test proves nothing if the system already has the C++ runtime, so
# refuse to run. These are the redistributable's C++ library names. The .NET
# Framework's private msvcp140_clr0400.dll is not one: no import of
# msvcp140.dll resolves to it.
$runtime = @(
    'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll', 'concrt140.dll'
)
$searched = @(
    (Join-Path $env:SystemRoot 'System32'),
    (Join-Path $env:SystemRoot 'SysWOW64')
) + @($env:Path -split ';' | Where-Object { $_ -and (Test-Path -LiteralPath $_) })
$found = @()
foreach ($directory in ($searched | Select-Object -Unique)) {
    $found += Get-ChildItem -LiteralPath $directory -File -ErrorAction Stop |
        Where-Object { $runtime -contains $_.Name.ToLowerInvariant() }
}
if ($found.Count -gt 0) {
    throw "The C++ runtime is installed, so the test is void: $($found.FullName -join ', ')"
}
Write-Host "None of $($runtime -join ', ') in: $($searched -join '; ')"

# Copy the interpreter, so that the mounts stay read-only.
$work = 'C:\work'
$python = Join-Path $work 'python'
$site = Join-Path $work 'site'
New-Item -ItemType Directory -Path $python, $site | Out-Null
Copy-Item -Path 'C:\embed\*' -Destination $python -Recurse
foreach ($dll in 'vcruntime140.dll', 'vcruntime140_1.dll') {
    if (-not (Test-Path -LiteralPath (Join-Path $python $dll))) {
        throw "The embeddable CPython lacks $dll."
    }
}
$exe = Join-Path $python 'python.exe'

# An embeddable CPython ignores PYTHONPATH and reads its search path from the
# `._pth` file, so list the directory that holds the unpacked wheel there.
$pth = Get-ChildItem -LiteralPath $python -Filter 'python3*._pth' -File
if (@($pth).Count -ne 1) {
    throw 'Expected one python3*._pth file.'
}
# The file may not end in a newline, and a line that joined the final comment
# would be ignored.
[System.IO.File]::AppendAllText($pth.FullName, "`r`n$site`r`n")

$wheels = @(Get-ChildItem -LiteralPath 'C:\wheelhouse' -File -Filter "c50-*-$Tag-$Tag-win_amd64.whl")
if ($wheels.Count -ne 1) {
    throw "Expected one $Tag wheel, found $($wheels.Count)."
}
Write-Host "Wheel: $($wheels[0].Name)"

# The wheel itself must not carry the runtime.
& $exe 'C:\tests\windows_clean_import.py' wheel $wheels[0].FullName
if ($LASTEXITCODE -ne 0) { throw "The wheel check failed with status $LASTEXITCODE." }

# A wheel is a ZIP archive whose root is the site directory, so unpacking it
# is an installation. The wheel has no runtime dependencies.
& $exe -m zipfile -e $wheels[0].FullName $site
if ($LASTEXITCODE -ne 0) { throw 'Could not unpack the wheel.' }

# Without the redistributable, the import must fail with ImportError.
& $exe 'C:\tests\windows_clean_import.py' absent
if ($LASTEXITCODE -ne 0) { throw "The import check failed with status $LASTEXITCODE." }

# Plant an old msvcp140.dll next to python.exe, where the loader finds it
# before System32. Run each case in its own process, and collect the failures
# so that one case does not hide the others.
$stale = 'C:\stale\msvcp140.dll'
if (-not (Test-Path -LiteralPath $stale)) {
    throw "The old runtime $stale is missing."
}
$planted = Join-Path $python 'msvcp140.dll'
$failures = @()
function Invoke-StaleCase {
    param([string]$Case, [string]$Dll)
    & $exe 'C:\tests\windows_clean_import.py' stale $Case $Dll
    if ($LASTEXITCODE -ne 0) {
        $script:failures += "$Case (status $LASTEXITCODE)"
    }
}

# An old runtime where the system has no other: the old copy is all there is.
Copy-Item -LiteralPath $stale -Destination $planted
Invoke-StaleCase 'system' $planted
Remove-Item -LiteralPath $planted

# Install the redistributable only if Microsoft signed the installer.
$installer = 'C:\vcredist\vc_redist.x64.exe'
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne 'Valid') {
    throw "The signature of $installer is $($signature.Status)."
}
$signer = $signature.SignerCertificate.Subject
if ($signer -notmatch '(^|, )CN=Microsoft Corporation(,|$)') {
    throw "$installer is signed by $signer, not Microsoft Corporation."
}
Write-Host "Installer signed by: $signer"
$install = Start-Process -FilePath $installer `
    -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
# 3010 asks for a restart, which the test does not need.
if (@(0, 3010) -notcontains $install.ExitCode) {
    throw "The redistributable installer exited with status $($install.ExitCode)."
}
$systemRuntime = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
if (-not (Test-Path -LiteralPath $systemRuntime)) {
    throw "The installer did not provide $systemRuntime."
}

# With the redistributable, the wheel imports, uses the system msvcp140.dll,
# and trains.
& $exe 'C:\tests\windows_clean_import.py' present
if ($LASTEXITCODE -ne 0) { throw "The import test failed with status $LASTEXITCODE." }

# The other two cases need the current runtime in System32, which the old copy
# must displace. A stale conda environment keeps its copy next to python.exe,
# and another package may load an old copy before c50 does.
Copy-Item -LiteralPath $stale -Destination $planted
Invoke-StaleCase 'conda' $planted
Remove-Item -LiteralPath $planted
Invoke-StaleCase 'preloaded' $stale
if ($failures.Count -gt 0) {
    throw "The stale runtime check failed for: $($failures -join ', ')."
}

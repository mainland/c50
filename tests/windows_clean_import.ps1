# Copyright 2026 Geoffrey Mainland.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Prove that a Windows wheel imports and trains on a system without the
# Visual C++ Redistributable. Run it in a Windows Server Core container with
# these directories mounted:
#
#   C:\wheelhouse  holds the wheels, read-only.
#   C:\embed       holds an unpacked embeddable CPython, read-only.
#   C:\tests       holds this script and windows_clean_import.py, read-only.
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

# A wheel is a ZIP archive whose root is the site directory, so unpacking it
# is an installation. The wheel has no runtime dependencies.
& $exe -m zipfile -e $wheels[0].FullName $site
if ($LASTEXITCODE -ne 0) { throw 'Could not unpack the wheel.' }

& $exe 'C:\tests\windows_clean_import.py'
if ($LASTEXITCODE -ne 0) { throw "The import test failed with status $LASTEXITCODE." }

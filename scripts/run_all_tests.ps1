# Run all tests and sanity checks (Windows PowerShell)
$ErrorActionPreference = "Stop"

# Move to repo root (this script lives in scripts/)
Set-Location (Split-Path $PSScriptRoot -Parent)

# Ensure output directories exist
if (!(Test-Path -LiteralPath "plots")) { New-Item -ItemType Directory -Path "plots" | Out-Null }
if (!(Test-Path -LiteralPath "Test_Outputs")) { New-Item -ItemType Directory -Path "Test_Outputs" | Out-Null }
if (!(Test-Path -LiteralPath "Test_Outputs/plots")) { New-Item -ItemType Directory -Path "Test_Outputs/plots" | Out-Null }

function Invoke-Step {
	param([string]$Name, [string]$Command)
	Write-Host "`n=== $Name ===" -ForegroundColor Cyan
	Write-Host ">>> $Command" -ForegroundColor DarkGray
	& cmd /c $Command
	if ($LASTEXITCODE -ne 0) {
		Write-Error "Step failed: $Name (exit $LASTEXITCODE)"
		exit $LASTEXITCODE
	}
}

Invoke-Step -Name "Unit tests (pytest)" -Command "uv run -m pytest -q"

# Legacy min-modes -> capture to Test_Outputs
Invoke-Step -Name "Legacy min-modes (fixed-ε)" -Command "uv run python scripts/legacy_min_modes_runner.py --lengths 1024,2048,4096 --eps 1e-1,3e-2 > Test_Outputs/legacy_min_modes_output.txt"

Invoke-Step -Name "Proportional CLI (small)" -Command "uv run python scripts/run_cli.py --lengths 1024,2048 --fractions 0.01,0.02 --plot"

# Move proportional outputs to Test_Outputs
$latestProp = Get-ChildItem -File -Filter "results_compressibility_proportional_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestProp) { Move-Item -LiteralPath $latestProp.FullName -Destination "Test_Outputs" -Force }
if (Test-Path -LiteralPath "plots/err_curves.png") { Move-Item -LiteralPath "plots/err_curves.png" -Destination "Test_Outputs/plots/err_curves.png" -Force }

Invoke-Step -Name "Proportional CLI (legacy long)" -Command "uv run python scripts/run_cli.py --lengths 4096,16384,65536,262144,524288 --fractions 0.001,0.002,0.005,0.01,0.02 --plot"

# Move long proportional outputs
$latestPropLong = Get-ChildItem -File -Filter "results_compressibility_proportional_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestPropLong) { Move-Item -LiteralPath $latestPropLong.FullName -Destination "Test_Outputs" -Force }
if (Test-Path -LiteralPath "plots/err_curves.png") { Move-Item -LiteralPath "plots/err_curves.png" -Destination "Test_Outputs/plots/err_curves_long.png" -Force }

Invoke-Step -Name "Residue entropy (small)" -Command "uv run python scripts/residue_entropy.py --L 2048 --q-list 3,4 --plot"

# Move residue entropy outputs
$latestResidue = Get-ChildItem -File -Filter "residue_entropy_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestResidue) { Move-Item -LiteralPath $latestResidue.FullName -Destination "Test_Outputs" -Force }
if (Test-Path -LiteralPath "plots/residue_entropy_L2048.png") { Move-Item -LiteralPath "plots/residue_entropy_L2048.png" -Destination "Test_Outputs/plots/residue_entropy_L2048.png" -Force }

Invoke-Step -Name "Critical line sweep (quick)" -Command "uv run python scripts/critical_line_sweep.py --lengths 2048,4096 --fractions 0.01 --t-min 0 --t-max 20 --t-steps 11 --avg-windows 3 --window hann --plot"

# Move critical line outputs
$latestCrit = Get-ChildItem -File -Filter "critical_line_ratio_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestCrit) { Move-Item -LiteralPath $latestCrit.FullName -Destination "Test_Outputs" -Force }
if (Test-Path -LiteralPath "plots/ratio_vs_t.png") { Move-Item -LiteralPath "plots/ratio_vs_t.png" -Destination "Test_Outputs/plots/ratio_vs_t.png" -Force }

Invoke-Step -Name "ML example (CPU)" -Command "uv run python scripts/ml_example.py --L 4096 --M 64"

Write-Host "`nAll tests and checks completed successfully." -ForegroundColor Green

# Optional: summarize key metrics into Test_Outputs/test_run_summary.csv
$summaryRows = @()
function Add-SummaryRow { param([string]$Category, [string]$Name, [string]$Value, [string]$Path) $script:summaryRows += [pscustomobject]@{ Category = $Category; Name = $Name; Value = $Value; Path = $Path } }

$propCsv = Get-ChildItem -File -Path "Test_Outputs" -Filter "results_compressibility_proportional_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($propCsv) {
	$csv = Import-Csv -LiteralPath $propCsv.FullName
	$ratios = @(); foreach ($row in $csv) { $ratios += [double]$row.'ratio_odd_over_even' }
	if ($ratios.Count -gt 0) {
		$avg = ( $ratios | Measure-Object -Average ).Average
		$min = ( $ratios | Measure-Object -Minimum ).Minimum
		$max = ( $ratios | Measure-Object -Maximum ).Maximum
		Add-SummaryRow -Category "proportional" -Name "ratio.mean" -Value ("{0:N3}" -f $avg) -Path $propCsv.FullName
		Add-SummaryRow -Category "proportional" -Name "ratio.min"  -Value ("{0:N3}" -f $min) -Path $propCsv.FullName
		Add-SummaryRow -Category "proportional" -Name "ratio.max"  -Value ("{0:N3}" -f $max) -Path $propCsv.FullName
	}
}

$critCsv = Get-ChildItem -File -Path "Test_Outputs" -Filter "critical_line_ratio_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($critCsv) {
	$csv = Import-Csv -LiteralPath $critCsv.FullName
	$rat = @(); foreach ($row in $csv) { $rat += [double]$row.'ratio_odd_over_even' }
	if ($rat.Count -gt 0) {
		$avgC = ( $rat | Measure-Object -Average ).Average
		$minC = ( $rat | Measure-Object -Minimum ).Minimum
		$maxC = ( $rat | Measure-Object -Maximum ).Maximum
		Add-SummaryRow -Category "critical_line" -Name "ratio.mean" -Value ("{0:N3}" -f $avgC) -Path $critCsv.FullName
		Add-SummaryRow -Category "critical_line" -Name "ratio.min"  -Value ("{0:N3}" -f $minC) -Path $critCsv.FullName
		Add-SummaryRow -Category "critical_line" -Name "ratio.max"  -Value ("{0:N3}" -f $maxC) -Path $critCsv.FullName
	}
}

$resCsv = Get-ChildItem -File -Path "Test_Outputs" -Filter "residue_entropy_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($resCsv) {
	$csv = Import-Csv -LiteralPath $resCsv.FullName
	$H = @(); foreach ($row in $csv) { $H += [double]$row.'entropy' }
	if ($H.Count -gt 0) {
		$avgH = ( $H | Measure-Object -Average ).Average
		$minH = ( $H | Measure-Object -Minimum ).Minimum
		$maxH = ( $H | Measure-Object -Maximum ).Maximum
		Add-SummaryRow -Category "residue_entropy" -Name "entropy.mean" -Value ("{0:N3}" -f $avgH) -Path $resCsv.FullName
		Add-SummaryRow -Category "residue_entropy" -Name "entropy.min"  -Value ("{0:N3}" -f $minH) -Path $resCsv.FullName
		Add-SummaryRow -Category "residue_entropy" -Name "entropy.max"  -Value ("{0:N3}" -f $maxH) -Path $resCsv.FullName
	}
}

if (Test-Path -LiteralPath "Test_Outputs/legacy_min_modes_output.txt") {
	Add-SummaryRow -Category "legacy_min_modes" -Name "output_path" -Value "see file" -Path (Resolve-Path "Test_Outputs/legacy_min_modes_output.txt").Path
}

$summaryPath = "Test_Outputs/test_run_summary.csv"
$summaryRows | Export-Csv -NoTypeInformation -Path $summaryPath
Write-Host "Summary written to $summaryPath" -ForegroundColor Yellow

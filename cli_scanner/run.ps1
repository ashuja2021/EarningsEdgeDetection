# EarningsEdgeDetection CLI Scanner Runner
# Usage: 
#   .\run.ps1                     - Run scanner with current date
#   .\run.ps1 MM/DD/YYYY          - Run scanner with specified date
#   .\run.ps1 -l                  - Run with list format
#   .\run.ps1 -i                  - Run with iron fly calculations
#   .\run.ps1 -a TICKER           - Analyze a specific ticker
#   .\run.ps1 -a TICKER -i        - Analyze ticker with iron fly strategy

# Get number of available cores for parallel processing
$NUM_CORES = (Get-CimInstance -ClassName Win32_Processor).NumberOfLogicalProcessors

# Use half the available cores (minimum 2, maximum 6 for stability)
$WORKERS = [math]::Floor($NUM_CORES / 2)
if ($WORKERS -lt 2) {
    $WORKERS = 2
}
elseif ($WORKERS -gt 6) {
    $WORKERS = 6
}

# Initialize flags
$LIST_FLAG = ""
$IRONFLY_FLAG = ""
$ANALYZE_FLAG = ""
$FINNHUB_FLAG = ""
$DOLTHUB_FLAG = ""
$COMBINED_FLAG = ""

# Check for flags
$args | ForEach-Object {
    switch ($_) {
        { $_ -in "-l", "--list" } { $LIST_FLAG = "--list" }
        { $_ -in "-i", "--iron-fly" } { $IRONFLY_FLAG = "--iron-fly" }
        { $_ -in "-f", "--use-finnhub" } { $FINNHUB_FLAG = "--use-finnhub" }
        { $_ -in "-u", "--use-dolthub" } { $DOLTHUB_FLAG = "--use-dolthub" }
        { $_ -in "-c", "--all-sources" } { $COMBINED_FLAG = "--all-sources" }
        { $_ -in "-a", "--analyze" } { 
            $ANALYZE_MODE = $true
        }
        default {
            if ($ANALYZE_MODE) {
                $ANALYZE_TICKER = $_
                $ANALYZE_FLAG = "--analyze $ANALYZE_TICKER"
                $ANALYZE_MODE = $false
            }
        }
    }
}

# Handle analyze mode specifically
if ($ANALYZE_TICKER) {
    Write-Host "Analyzing ticker: $ANALYZE_TICKER"
    python scanner.py $ANALYZE_FLAG $IRONFLY_FLAG $FINNHUB_FLAG $DOLTHUB_FLAG $COMBINED_FLAG
    exit
}

# Run the scanner with parallel processing
if (-not $args[0]) {
    # No arguments - Run with current date
    python scanner.py --parallel $WORKERS $LIST_FLAG $IRONFLY_FLAG $FINNHUB_FLAG $DOLTHUB_FLAG $COMBINED_FLAG
}
elseif ($args[0] -match '^-') {
    # Handle flags when they appear as the first argument
    python scanner.py --parallel $WORKERS $LIST_FLAG $IRONFLY_FLAG $FINNHUB_FLAG $DOLTHUB_FLAG $COMBINED_FLAG
}
else {
    # Date is provided as first argument
    python scanner.py --date $args[0] --parallel $WORKERS $LIST_FLAG $IRONFLY_FLAG $FINNHUB_FLAG $DOLTHUB_FLAG $COMBINED_FLAG
} 
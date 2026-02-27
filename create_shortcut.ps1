# Create a desktop shortcut for Polymarket Tracker

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Polymarket Tracker.lnk")
$Shortcut.TargetPath = "$PSScriptRoot\dist\PolymarketTracker.exe"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Polymarket Tracker - Monitor whale trades and insider activity"
$Shortcut.Save()

Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "You can now launch Polymarket Tracker from your desktop."

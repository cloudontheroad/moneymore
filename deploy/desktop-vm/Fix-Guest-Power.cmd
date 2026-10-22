@echo off
rem Run manually INSIDE the VMware Windows guest. No scheduled task is created.
rem Refuse the physical host. Do not change lock-screen or authentication settings.
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; $machine=Get-CimInstance Win32_ComputerSystem; if($machine.Manufacturer -notmatch 'VMware' -or $machine.Model -notmatch 'VMware'){Write-Error 'REFUSED: this repair is only for a VMware guest, not the host'; exit 2}; Write-Output 'Guest verified. Power settings BEFORE:'; powercfg.exe /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE; if($LASTEXITCODE -ne 0){exit 3}; foreach($option in @('standby-timeout-ac','standby-timeout-dc','hibernate-timeout-ac','hibernate-timeout-dc')){powercfg.exe /change $option 0; if($LASTEXITCODE -ne 0){Write-Error ('Failed to set '+$option); exit 4}}; Write-Output 'Power settings AFTER (0 means Never):'; powercfg.exe /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE; powercfg.exe /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE; Write-Output 'DONE: guest idle sleep/hibernate timers disabled. Locking, host power and security unchanged.'"
echo.
echo Keep this window open if any error is shown. Guest settings still need an idle-runtime test.
pause

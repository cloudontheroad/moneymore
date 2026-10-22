# Download only. Never mounts media, starts setup, or restarts the host.
$ErrorActionPreference = 'Stop'
$mediaDirectory = 'D:\MoneyMoreVM\media'
$partialPath = 'D:\MoneyMoreVM\media\Windows11-Enterprise-Eval-26H2-zh-cn-x64.iso.part'
$completePath = 'D:\MoneyMoreVM\media\Windows11-Enterprise-Eval-26H2-zh-cn-x64.iso'
$officialUrl = 'https://software-static.download.prss.microsoft.com/dbazure/26300.9457.260913-1737.26h2_ge_release_svc_refresh_CLIENTENTERPRISEEVAL_OEMRET_x64FRE_zh-cn.iso'
if (Test-Path -LiteralPath $completePath) {
    Get-FileHash -LiteralPath $completePath -Algorithm SHA256
    exit 0
}
New-Item -ItemType Directory -Path $mediaDirectory -Force | Out-Null
# Resume a partial download; a server without Range support causes a safe failure.
& curl.exe --fail --location --continue-at - --connect-timeout 20 --output $partialPath $officialUrl
if ($LASTEXITCODE -ne 0) {
    throw 'Download incomplete; retained .part for a later explicit resume.'
}
if ((Get-Item -LiteralPath $partialPath).Length -lt 4GB) {
    throw 'Unexpected image size; retained .part for inspection.'
}
if (Test-Path -LiteralPath $completePath) {
    throw 'Completed target appeared during download; no overwrite performed.'
}
Move-Item -LiteralPath $partialPath -Destination $completePath
# This is a reproducibility hash, not comparison with a Microsoft-published hash.
Get-FileHash -LiteralPath $completePath -Algorithm SHA256

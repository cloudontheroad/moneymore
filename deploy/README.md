# MoneyMore 火山云部署

生产形态：一台 Linux ECS、Nginx、一个 Uvicorn worker、一个 Dashboard 进程。每日调度由 API 进程内的单实例调度器执行，禁止启动多个 API worker。GPU 训练留在本地，模型产物通过 `sync-models.ps1` 同步。

部署包包含代码、配置和获准的运行数据。首次迁移时在服务器单独配置 `.env`，并通过独立授权流程迁移需要保留的账本和模型。历史 `data/raw` 可按实际用途单独迁移。

服务器建议：Ubuntu 24.04，至少 4 核 8GB、80GB SSD。安全组仅开放 TCP 22、80；配置域名和证书后再开放 443。8788 和 3000 只监听回环地址，不对公网开放。

本地打包并通过 SSH 上传：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\package-cloud.ps1
powershell -ExecutionPolicy Bypass -File .\deploy\upload-cloud.ps1 -SshTarget root@<公网IP>
```

部署前在服务器创建访问认证：

```bash
sudo htpasswd -c /etc/nginx/.htpasswd-moneymore <username>
sudo bash /opt/moneymore/current/deploy/bootstrap-ubuntu.sh
```

验收：

```bash
systemctl status moneymore-api moneymore-dashboard moneymore-backup.timer
curl -fsS http://127.0.0.1:8788/api/health
curl -I http://127.0.0.1:3000
journalctl -u moneymore-api -n 100 --no-pager
```

公网地址在无域名时为 `http://<公网IP>/`，由 Nginx Basic Auth 保护。正式长期使用建议绑定域名并用 Certbot 开启 HTTPS。

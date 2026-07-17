# Triển khai Production lên Google Cloud VM

Tài liệu này mô tả Checkpoint P1C cho website `credit-scoring-replication`.

Phạm vi P1C chỉ gồm website production, Docker image immutable trên GHCR, SSH vào Google Cloud VM, Docker Compose, health/version check, logging và rollback. P1C không triển khai Phase 2, pipeline thực nghiệm, backend API, database, domain hoặc HTTPS.

## Kiến trúc

```text
GitHub Actions
  -> build Docker image website
  -> tag bằng full Git commit SHA
  -> push lên GHCR
  -> SSH vào Google Cloud VM
  -> cập nhật deployment bundle
  -> docker pull image SHA
  -> docker compose up -d
  -> kiểm tra /, /health/, /version/
  -> rollback tự động nếu deploy lỗi
```

Container website lắng nghe port `8080`. Production map host port `80` sang container port `8080` bằng `docker-compose.prod.yml`.

Không dùng host-level Nginx trong checkpoint này. Domain và HTTPS vẫn Optional.

## Prerequisites trên VM

VM hiện được xác nhận qua workflow `.github/workflows/test-vm-ssh.yml`:

- SSH từ GitHub Actions vào VM hoạt động.
- Docker hoạt động.
- Docker Compose hoạt động.
- Production user chạy được `docker ps` không cần `sudo`.
- Port `80` dùng để công bố website.

Thư mục production cần tồn tại và production user cần có quyền ghi:

```bash
sudo mkdir -p /opt/credit-scoring-deploy/scripts
sudo mkdir -p /opt/credit-scoring-deploy/state
sudo mkdir -p /opt/credit-scoring-deploy/logs
sudo chown -R "$USER":"$USER" /opt/credit-scoring-deploy
```

Không chạy deploy bằng root SSH user.

## GitHub Secrets

Workflow production dùng các secrets hiện có:

| Secret | Mục đích |
|---|---|
| `PROD_HOST` | Public IP hoặc hostname của VM, hiện là `34.142.206.15`. |
| `PROD_USER` | SSH user production trên VM. |
| `PROD_SSH_PORT` | SSH port. |
| `PROD_SSH_PRIVATE_KEY` | Private key để GitHub Actions SSH vào VM. |
| `PROD_KNOWN_HOSTS` | Known hosts entry đã pin host key của VM. |

Workflow dùng strict host-key checking và không tắt `StrictHostKeyChecking`.

## GHCR authentication

Image production có dạng:

```text
ghcr.io/<owner>/<repository>-website:<full-sha>
```

Tên `<owner>/<repository>` được chuẩn hóa chữ thường. Ví dụ:

```text
ghcr.io/<owner>/credit-scoring-replication-website:2035e112295b5f5292854e9f1b3adb52928fed4e
```

GitHub Actions push image bằng `GITHUB_TOKEN` với permission:

```yaml
permissions:
  contents: read
  packages: write
```

Nếu GHCR package là private, VM phải login GHCR trước khi `docker pull`. Tạo token riêng ngoài repo với quyền tối thiểu `read:packages`, sau đó chạy trên VM:

```bash
printf '%s' '<GHCR_READ_PACKAGES_TOKEN>' | docker login ghcr.io -u '<github-username>' --password-stdin
```

Không commit token, không ghi token vào log, và không đưa token vào workflow nếu VM đã login sẵn.

## Docker Compose production

File `docker-compose.prod.yml` yêu cầu biến:

```bash
WEBSITE_IMAGE='ghcr.io/<owner>/<repository>-website:<full-sha>'
```

Các thuộc tính chính:

- `container_name: credit-scoring-website-prod`
- `ports: "80:8080"`
- `restart: unless-stopped`
- healthcheck `/health/`
- không mount source code
- không mount `data/raw/` hoặc `data/processed/`
- không chứa secret

## Deploy tự động

Workflow `.github/workflows/deploy-production.yml` chạy khi push vào `main` và chỉ khi path liên quan thay đổi:

- `website/**`
- `deploy/**`
- `scripts/deploy-production.sh`
- `scripts/rollback-production.sh`
- `docker-compose.prod.yml`
- `.github/workflows/deploy-production.yml`

Workflow không chạy deployment trên Pull Request.

## Deploy thủ công

Vào GitHub Actions, chọn `Deploy Production`, chạy `workflow_dispatch`:

- `action`: `deploy`
- `deploy_sha`: full 40-character commit SHA cần build/deploy, có thể để trống để dùng commit của workflow.
- `image_tag`: optional Docker tag hợp lệ, mặc định bằng `deploy_sha`.

Input được validate để tránh shell injection. Production version vẫn phải khớp `deploy_sha`.

## Rollback thủ công

Chạy `workflow_dispatch`:

- `action`: `rollback`

Workflow SSH vào VM và gọi:

```bash
/opt/credit-scoring-deploy/scripts/rollback-production.sh
```

Rollback fail rõ nếu chưa có `previous.env`.

Có thể chạy trực tiếp trên VM:

```bash
DEPLOY_ROOT=/opt/credit-scoring-deploy \
  /opt/credit-scoring-deploy/scripts/rollback-production.sh
```

## Health và version check

Deploy chỉ được xem là thành công khi tất cả check pass:

```bash
curl -fsS http://127.0.0.1/
curl -fsS http://127.0.0.1/health/
curl -fsS http://127.0.0.1/version/
```

`/health/` phải chứa `OK`. `/version/` phải chứa full commit SHA đang deploy.

Workflow cũng kiểm tra public endpoint:

```bash
curl -fsS http://34.142.206.15/
curl -fsS http://34.142.206.15/health/
curl -fsS http://34.142.206.15/version/
```

## Deployment state và logs

State và logs nằm ngoài Git working tree:

```text
/opt/credit-scoring-deploy/state/current.env
/opt/credit-scoring-deploy/state/previous.env
/opt/credit-scoring-deploy/logs/deploy-*.log
/opt/credit-scoring-deploy/logs/rollback-*.log
```

`current.env` lưu image đang chạy. `previous.env` lưu image thành công gần nhất trước đó để rollback.

Xem logs:

```bash
ls -lt /opt/credit-scoring-deploy/logs
tail -n 200 /opt/credit-scoring-deploy/logs/deploy-*.log
docker logs credit-scoring-website-prod --tail=200
```

## Khi deploy lỗi

`deploy-production.sh` tự rollback nếu:

- `docker pull` hoặc `docker compose up` lỗi;
- `/health/` timeout;
- `/` không trả thành công;
- `/health/` không trả thành công;
- `/version/` không khớp SHA yêu cầu.

Nếu rollback cũng lỗi, workflow fail và log nêu rõ nguyên nhân.

## Image retention

P1C không chạy:

```bash
docker system prune -a
```

Giữ tối thiểu:

- image hiện tại;
- image previous successful;
- ít nhất một production image cũ hơn nếu dung lượng cho phép.

Khi cần dọn disk, kiểm tra state trước:

```bash
cat /opt/credit-scoring-deploy/state/current.env
cat /opt/credit-scoring-deploy/state/previous.env
docker image ls 'ghcr.io/*credit-scoring-replication-website*'
```

Chỉ xóa image không nằm trong state và không cần cho rollback.

## Firewall port 80

Google Cloud firewall cần cho phép inbound TCP `80` tới VM. SSH chỉ nên mở theo chính sách vận hành hiện tại.

Container port `8080` không cần expose trực tiếp ra Internet; production public surface là host port `80`.

## Security checklist

Trước khi chấp nhận production:

- Không có `data/raw/` trong image.
- Không có `data/processed/` trong image.
- Không có production `.env` trong image.
- Không có private key trong image.
- Không có GitHub token trong log.
- Không có Windows local path trong website content.
- SSH strict host-key checking đang bật.
- Không dùng root SSH login.
- Workflow deploy không chạy trên Pull Request.
- `/version/` khớp full commit SHA.

## Acceptance P1C

P1C chỉ hoàn thành chính thức khi đã chạy thật và có evidence:

1. Image được push lên GHCR với full commit SHA.
2. GitHub Actions SSH được vào VM.
3. VM pull đúng immutable image.
4. Website truy cập được tại public IP.
5. `/` trả thành công.
6. `/health/` pass.
7. `/version/` khớp commit đang deploy.
8. Push phù hợp vào `main` kích hoạt deployment.
9. Path filter hoạt động.
10. Concurrency control hoạt động.
11. Rollback thực tế được kiểm thử ít nhất một lần.
12. Failed deployment tự rollback được.
13. Deployment logs tồn tại.
14. Current và previous image được lưu đúng.
15. Không có raw data hoặc secret trong image.
16. Tài liệu deployment đầy đủ.
17. Domain và HTTPS không phải điều kiện bắt buộc.
18. Chưa triển khai Phase 2.

Nếu chưa chạy được GitHub Secrets hoặc VM thật, production acceptance là `Pending`, không được ghi là Completed.

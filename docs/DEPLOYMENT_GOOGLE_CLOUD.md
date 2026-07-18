# Triển khai Production lên Google Cloud VM

Tài liệu này mô tả Checkpoint P1C cho website `credit-scoring-replication`.

Phạm vi P1C chỉ gồm website production, Docker image immutable trên GHCR, SSH vào Google Cloud VM, Docker Compose, health/version check, logging và rollback. P1C không triển khai Phase 2, pipeline thực nghiệm, backend API, database, domain hoặc HTTPS.

## Trạng thái hiện tại

Các fact đã được xác nhận sau khi P1C được merge vào `main`:

- Workflow `Test VM SSH` đã pass.
- Production deployment workflow đã chạy tự động thành công từ `main`.
- Docker image website đã được build, push lên GHCR và deploy lên Google Cloud VM.
- Website production đang hoạt động tại `http://34.142.206.15`.
- Health/version validation trong deployment workflow đã pass vì workflow production báo thành công.
- Website hiện chạy bằng HTTP/public IP.
- Domain và HTTPS vẫn Optional trong phạm vi hiện tại.
- Website chỉ có nội dung công khai, không có authentication hoặc chức năng truyền dữ liệu nhạy cảm.

Các kiểm thử production còn pending:

- Manual rollback production.
- Automatic rollback khi deployment lỗi.

Vì hai kiểm thử rollback chưa được xác nhận thật, P1C ở trạng thái **production deployment operational; rollback verification pending**. Không đánh dấu Phase 1 Completed hoàn toàn cho đến khi hai kiểm thử này pass.

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

VM đã được xác nhận qua workflow `.github/workflows/test-vm-ssh.yml`:

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

Trạng thái hiện tại: automatic deployment từ `main` đã pass và website production đang phục vụ tại `http://34.142.206.15`.

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

## Runbook kiểm thử rollback

Manual rollback production đã PASS thì có thể ghi evidence bằng workflow dispatch:

- `action`: `rollback`

Workflow sẽ SSH vào VM và gọi:

```bash
DEPLOY_ROOT=/opt/credit-scoring-deploy \
  /opt/credit-scoring-deploy/scripts/rollback-production.sh
```

Sau rollback, xác minh production vẫn healthy:

```bash
curl -fsS http://34.142.206.15/ >/dev/null
curl -fsS http://34.142.206.15/health/
curl -fsS http://34.142.206.15/version/
```

Kiểm tra state trên VM:

```bash
cat /opt/credit-scoring-deploy/state/current.env
cat /opt/credit-scoring-deploy/state/previous.env
ls -lt /opt/credit-scoring-deploy/logs
tail -n 200 /opt/credit-scoring-deploy/logs/rollback-*.log
```

### Forced automatic rollback test

Chỉ dùng test hook này qua manual `workflow_dispatch`. Không bật test hook này trong automatic push vào `main`.

Chạy workflow `.github/workflows/deploy-production.yml` thủ công:

- `action`: `deploy`
- `deploy_sha`: full 40-character commit SHA cần build/deploy.
- `image_tag`: để trống để dùng `deploy_sha`, hoặc nhập Docker tag hợp lệ.
- `force_post_deploy_failure`: `true`

`force_post_deploy_failure` không phải secret. Workflow validate giá trị boolean và automatic deployment từ push vào `main` luôn resolve thành `false`.

Expected result khi `force_post_deploy_failure=true`:

1. Build image pass.
2. Smoke test image pass, gồm `/`, `/health/`, và `/version/` khớp `deploy_sha`.
3. Push image pass.
4. Copy deployment bundle pass.
5. VM pull candidate image pass.
6. Candidate container start pass.
7. Production health/version validation pass.
8. Log có marker: `Forced post-deploy failure requested for rollback verification.`
9. Automatic rollback start.
10. Previous good image được restore.
11. Restored image health/version validation pass.
12. `current.env` quay lại previous good image.
13. Workflow kết thúc failure để phản ánh candidate deployment không thành công.
14. Production vẫn chạy previous good image.

Log deploy cần có các dòng về candidate image, previous good image, health-check result, forced failure marker, rollback start, restored image health result, và final rollback result. Không log GHCR token, private key, hoặc secret values.

Sau test, xác minh state trên VM:

```bash
cat /opt/credit-scoring-deploy/state/current.env
cat /opt/credit-scoring-deploy/state/previous.env
tail -n 200 /opt/credit-scoring-deploy/logs/deploy-*.log
```

`current.env` phải trỏ về previous good image. Bad/test candidate image không được ghi thành successful current state. Nếu test được tạo trên branch riêng, dọn branch test sau khi evidence đã được ghi nhận:

```bash
git branch -d <test-branch>
git push origin --delete <test-branch>
```

Không đánh dấu Phase 1 Completed cho đến khi manual rollback production và forced automatic rollback test đều PASS trên production thật.

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

Trạng thái acceptance hiện tại:

| Tiêu chí | Trạng thái |
|---|---|
| Image được push lên GHCR với full commit SHA | Passed theo workflow production đã thành công |
| GitHub Actions SSH được vào VM | Passed |
| VM pull đúng immutable image | Passed theo workflow production đã thành công |
| Website truy cập được tại public IP | Passed: `http://34.142.206.15` |
| `/` trả thành công | Passed theo workflow production đã thành công |
| `/health/` pass | Passed theo workflow production đã thành công |
| `/version/` khớp commit đang deploy | Passed theo workflow production đã thành công |
| Push phù hợp vào `main` kích hoạt deployment | Passed |
| Path filter hoạt động | Passed theo automatic deployment từ `main` |
| Concurrency control hoạt động | Configured; chưa có evidence cạnh tranh đồng thời |
| Rollback thực tế được kiểm thử ít nhất một lần | Pending |
| Failed deployment tự rollback được | Pending |
| Deployment logs tồn tại | Expected từ deploy script; cần xác nhận trên VM nếu cần evidence file |
| Current và previous image được lưu đúng | Expected từ deploy script; cần xác nhận trên VM nếu cần evidence file |
| Không có raw data hoặc secret trong image | Passed theo CI/deployment image scan |
| Tài liệu deployment đầy đủ | Updated |
| Domain và HTTPS không phải điều kiện bắt buộc | Confirmed Optional |
| Chưa triển khai Phase 2 | Confirmed |

Phase 1 chỉ nên chuyển sang Completed sau khi manual rollback production và automatic failed-deployment rollback được kiểm thử thật.

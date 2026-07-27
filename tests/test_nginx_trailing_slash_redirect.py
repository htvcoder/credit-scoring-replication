from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nginx_redirects_do_not_expose_container_port():
    config = (ROOT / "website" / "nginx.conf").read_text(encoding="utf-8")
    dockerfile = (ROOT / "website" / "Dockerfile").read_text(encoding="utf-8")

    assert "listen 8080;" in config
    assert "port_in_redirect off;" in config
    assert "absolute_redirect off;" in config
    assert "try_files $uri $uri/ =404;" in config
    assert "COPY nginx.conf /etc/nginx/conf.d/default.conf" in dockerfile


def test_frontend_navigation_targets_exported_trailing_slash_routes():
    navigation = (ROOT / "website" / "components" / "Nav.tsx").read_text(encoding="utf-8")
    for route in ("/gioi-thieu/", "/datasets/", "/phuong-phap/", "/tien-do/", "/ket-qua/"):
        assert f'"{route}"' in navigation


def test_deployment_smoke_uses_case_guard_and_checks_redirect_destination():
    workflow = (ROOT / ".github" / "workflows" / "deploy-production.yml").read_text(encoding="utf-8")

    assert 'case "${location}" in' in workflow
    assert 'Internal container port leaked in Location' in workflow
    assert '*/"${route}"/)' in workflow
    assert 'test "${location}" != *":8080"*' not in workflow

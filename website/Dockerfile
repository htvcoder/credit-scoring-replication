# syntax=docker/dockerfile:1

FROM node:24-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:24-alpine AS builder
WORKDIR /app
ARG BUILD_SHA=unknown
ENV NEXT_PUBLIC_BUILD_SHA=$BUILD_SHA
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine AS runner
ARG BUILD_SHA=unknown
LABEL org.opencontainers.image.title="credit-scoring-replication-website"
LABEL org.opencontainers.image.description="Static website for the credit-scoring-replication internship project"
LABEL org.opencontainers.image.revision=$BUILD_SHA
COPY --from=builder /app/out /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/health/ | grep -q "OK"

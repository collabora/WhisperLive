# MinIO Dockerfile
# Built from source on Alpine for a minimal, production-grade S3-compatible store.

# Builder stage to compile MinIO and mc from source
FROM alpine:latest AS builder

# Install Go for building
RUN apk add --no-cache go

# Set GOPATH and compile minio and mc
ENV GOPATH=/go
RUN go install github.com/minio/minio@latest && \
    go install github.com/minio/mc@latest

# Final stage
FROM alpine:latest

# Install dependencies
RUN apk add --no-cache \
    ca-certificates \
    curl \
    iproute2 \
    openssl \
    && rm -rf /var/cache/apk/*

# Copy compiled binaries from builder
COPY --from=builder /go/bin/minio /usr/local/bin/minio
COPY --from=builder /go/bin/mc /usr/local/bin/mc

# Create directories for data and certificates
RUN mkdir -p /data /root/.minio/certs

# Generate self-signed certificates (for testing)
RUN openssl req -x509 -newkey rsa:4096 -keyout /root/.minio/certs/private.key \
    -out /root/.minio/certs/public.crt -days 365 -nodes \
    -subj "/CN=*.minio.example.com" -addext "subjectAltName=DNS:minio.example.com,DNS:*.minio.example.com" \
    && chmod 600 /root/.minio/certs/private.key /root/.minio/certs/public.crt

# Copy custom entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose MinIO ports (9000 for S3 API, 9001 for Console)
EXPOSE 9000 9001

# Set the working directory
WORKDIR /root

# Set environment variable for domain (non-sensitive default)
ENV MINIO_DOMAIN=minio.example.com

# Use custom entrypoint to configure virtual host support
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the command to run MinIO server with TLS
CMD ["minio", "server", "/data", "--address", ":9000", "--console-address", ":9001"]

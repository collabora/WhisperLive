# WhisperLive Remote Setup Guide

## Overview
This setup allows you to run WhisperLive on your intranet machine and expose it through a public cloud instance via SSH remote port forwarding.

## Architecture
```
Browser → Nginx (Cloud) → SSH Tunnel → WhisperLive Server (Intranet)
```

## Setup Steps

### 1. WhisperLive Server Setup (Intranet Machine)

First, install and run WhisperLive on your intranet machine:

```bash
# Install WhisperLive
pip install whisper-live
# OR
git clone https://github.com/collabora/WhisperLive
# OR use this repo till final merge:
git clone https://github.com/klonikar/WhisperLive

# Start the server (default port 9090)
python run_server.py -fw deepdml/faster-whisper-large-v3-turbo-ct2
```

### 2. SSH Remote Port Forwarding

From your intranet machine, create an SSH tunnel to your cloud instance:

```bash
# Basic SSH tunnel - forwards local port 9090 to cloud instance port 9090
ssh -R 9090:localhost:9090 user@your-cloud-instance.com

# Keep the tunnel alive with auto-reconnect
ssh -R 9090:localhost:9090 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 user@your-cloud-instance.com

# Run in background with autossh (install autossh first)
autossh -M 0 -R 9090:localhost:9090 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 user@your-cloud-instance.com
```

### 3. Nginx Configuration (Cloud Instance)

Apply the nginx configuration provided in the artifacts:

```bash
# Edit your nginx configuration
sudo nano /etc/nginx/sites-available/your-site

# Test the configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 4. SSL Certificate (Recommended)

For WebSocket connections over HTTPS, you'll need an SSL certificate:

```bash
# Using Let's Encrypt with certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourserver.com
```

### 5. Firewall Configuration

Ensure your cloud instance firewall allows the necessary ports:

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443

# If using a specific port for the tunnel
sudo ufw allow 9090
```

## Testing the Setup

### 0. Test client from filesystem
Simply open the file whisperlive_client.html from the file explorer and connect it to a whisperlive server on the localhost

### 1. Test WhisperLive Server
```bash
# On your intranet machine
curl http://localhost:9090/health
```

### 2. Test SSH Tunnel
```bash
# On your cloud instance
curl http://localhost:9090/health
```

### 3. Test Nginx Proxy
```bash
# From outside
curl http://yourserver.com/whisper-ws
```

## Browser Client Usage

1. Open the HTML page in your browser
2. Update the WebSocket URL to: `wss://yourserver.com/whisper-ws` (or `ws://` for HTTP)
3. Configure sample rate and language
4. Click "Start Recording" to begin transcription

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if SSH tunnel is active
   - Verify nginx configuration
   - Check firewall settings

2. **Audio Not Streaming**
   - Ensure microphone permissions are granted
   - Check browser console for errors
   - Verify audio format compatibility

3. **SSH Tunnel Disconnects**
   - Use `autossh` for auto-reconnection
   - Increase `ServerAliveInterval` settings
   - Check network stability

### Debug Commands

```bash
# Check if WhisperLive is running
ps aux | grep whisper

# Check SSH tunnel status
ps aux | grep ssh

# Check nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Test WebSocket connection
wscat -c ws://localhost:9090  # Install wscat: npm install -g wscat
```

## Security Considerations

1. **Use SSH Key Authentication**
   ```bash
   # Generate SSH key if not exists
   ssh-keygen -t rsa -b 4096
   
   # Copy to cloud instance
   ssh-copy-id user@your-cloud-instance.com
   ```

2. **Restrict SSH Access**
   ```bash
   # In /etc/ssh/sshd_config on cloud instance
   AllowUsers your-username
   PermitRootLogin no
   PasswordAuthentication no
   ```

3. **Use SSL/TLS**
   - Always use HTTPS in production
   - Configure proper SSL certificates
   - Use secure WebSocket connections (wss://)

## Performance Optimization

1. **Audio Quality Settings**
   - Use 16kHz sample rate for better performance
   - Enable noise suppression and echo cancellation
   - Adjust chunk size based on network conditions

2. **Network Optimization**
   - Use compression in SSH tunnel: `ssh -C -R ...`
   - Optimize nginx buffer settings
   - Consider using a VPN for better tunnel stability

3. **WhisperLive Settings**
   ```bash
   # Start with optimized settings
   python -m whisper_live.server \
     --port 9090 \
     --host 0.0.0.0 \
     --model base \
     --device cuda  # if GPU available
   ```

## Systemd Service (Optional)

Create a systemd service for auto-starting the SSH tunnel:

```ini
# /etc/systemd/system/whisper-tunnel.service
[Unit]
Description=WhisperLive SSH Tunnel
After=network.target

[Service]
Type=simple
User=your-username
ExecStart=/usr/bin/autossh -M 0 -R 9090:localhost:9090 -o ServerAliveInterval=60 -o ServerAliveCountMax=3 user@your-cloud-instance.com
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start the service
sudo systemctl enable whisper-tunnel.service
sudo systemctl start whisper-tunnel.service
sudo systemctl status whisper-tunnel.service
```
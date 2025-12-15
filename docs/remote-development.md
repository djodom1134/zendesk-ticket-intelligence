# Remote Development Setup

This document outlines how to set up seamless development between your local machine and the remote GPU machine.

## Current Configuration

- **GPU Machine**: DGX Spark (spark-b4eb)
- **GPU Machine IP**: 192.168.87.134
- **SSH Host Alias**: `zti-gpu`
- **User**: djodom
- **GPU**: NVIDIA GB10 (Grace Blackwell)
- **CUDA**: 13.0
- **RAM**: 119GB
- **Storage**: 3.7TB

## SSH Configuration

### 1. SSH Config (Already Configured)

The following has been added to `~/.ssh/config`:

```
# ZTI GPU Machine for Zendesk Ticket Intelligence project
Host zti-gpu
    HostName 192.168.87.134
    User djodom
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### 2. SSH Key Authentication ✅

SSH key has been copied to the GPU machine. Passwordless access is configured.

### 3. Test Connection
```bash
ssh zti-gpu "hostname && nvidia-smi --query-gpu=name --format=csv"
```

### 4. Quick Access
```bash
# SSH into GPU machine
ssh zti-gpu

# Project directory on GPU machine
cd ~/zendesk-ticket-intelligence
```

## VS Code Remote Development

### 1. Install Extensions
- Remote - SSH
- Remote - Containers
- Python
- Docker

### 2. Connect to Remote Machine
1. Open VS Code
2. Press `Ctrl+Shift+P`
3. Type "Remote-SSH: Connect to Host"
4. Select `zti-gpu`

### 3. Clone Repository on Remote Machine
```bash
# On GPU machine
git clone --recursive https://github.com/[username]/zendesk-ticket-intelligence.git
cd zendesk-ticket-intelligence
```

## Development Workflow

### 1. File Synchronization
Use VS Code's built-in sync or set up rsync:
```bash
# Sync local changes to remote
rsync -avz --exclude='.git' ./ zti-gpu:~/zendesk-ticket-intelligence/

# Sync remote changes to local
rsync -avz zti-gpu:~/zendesk-ticket-intelligence/ ./
```

### 2. Docker Development
```bash
# On GPU machine
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 3. Port Forwarding
Forward ports for local access:
```bash
# Forward UI port
ssh -L 3000:localhost:3000 zti-gpu

# Forward all services
ssh -L 3000:localhost:3000 -L 8001:localhost:8001 -L 8529:localhost:8529 -L 6333:localhost:6333 zti-gpu
```

## Debugging Setup

### 1. Python Remote Debugging
Add to service Dockerfiles:
```dockerfile
# Install debugpy
RUN pip install debugpy

# Add debug entrypoint
COPY debug-entrypoint.sh /debug-entrypoint.sh
RUN chmod +x /debug-entrypoint.sh
```

### 2. VS Code Launch Configuration
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/app"
                }
            ]
        }
    ]
}
```

## Security Considerations

### 1. SSH Hardening
On GPU machine `/etc/ssh/sshd_config`:
```
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
AllowUsers [your-username]
```

### 2. Firewall Rules
```bash
# Allow SSH and application ports
sudo ufw allow ssh
sudo ufw allow 3000
sudo ufw allow 8001
sudo ufw enable
```

### 3. Environment Variables
Never commit secrets. Use `.env` files:
```bash
# On GPU machine
cp .env.example .env
# Edit with actual values
```

## Troubleshooting

### SSH Connection Issues
```bash
# Test connection
ssh -v zti-gpu

# Check SSH agent
ssh-add -l

# Restart SSH service (on remote)
sudo systemctl restart ssh
```

### Docker Issues
```bash
# Check Docker daemon
sudo systemctl status docker

# Check GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Port Forwarding Issues
```bash
# Check if ports are in use
netstat -tulpn | grep :3000

# Kill existing SSH tunnels
pkill -f "ssh.*-L"
```
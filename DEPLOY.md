# 🚀 Deployment Guide — Docker + VPS

## Prerequisites
- A Linux VPS (Ubuntu 22.04+ recommended)
- Docker & Docker Compose installed
- Git installed

---

## Step 1: SSH into your VPS
```bash
ssh user@your-vps-ip
```

## Step 2: Clone the repo
```bash
git clone https://github.com/VinayRaikar05/SYSTEMANALYZER.git
cd SYSTEMANALYZER
```

## Step 3: Set API keys (optional but recommended)
```bash
export API_KEYS="your-secret-key-here"
```

## Step 4: Build & run
```bash
docker compose up -d --build
```

That's it! The engine is now live at `http://your-vps-ip:8000/`

---

## Useful Commands

```bash
# View logs
docker compose logs -f

# Check status
docker compose ps

# Stop
docker compose down

# Restart
docker compose restart

# Rebuild after pulling updates
git pull
docker compose up -d --build

# Retrain models inside container
docker exec system-failure-engine python -m backend.train_model

# View structured logs
docker exec system-failure-engine cat /app/logs/system.log
```

---

## Production Checklist

- [ ] Set `API_KEYS` environment variable for auth
- [ ] Set up a reverse proxy (Nginx) for HTTPS
- [ ] Point a domain to your VPS IP
- [ ] Set up firewall (allow only 80/443)
- [ ] Enable automatic Docker restart on reboot

### Nginx Reverse Proxy (optional)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then get HTTPS with:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

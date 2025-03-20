# Deployment Guide

## Local Deployment

1. **Set up Virtual Environment**:
```bash
cd CryptoTradingWeb
python -m venv venv
venv\Scripts\activate  # On Windows
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Environment**:
- Copy `.env.example` to `.env`
- Update the SECRET_KEY
- Add your Delta Exchange API credentials

4. **Initialize Database**:
```bash
flask db init
flask db migrate
flask db upgrade
```

5. **Run Application**:
```bash
python app.py
```

## Production Deployment

### Option 1: Using Gunicorn (Linux/Unix)

1. **Install Production Dependencies**:
```bash
pip install gunicorn
```

2. **Set Environment Variables**:
```bash
export FLASK_ENV=production
export FLASK_APP=app.py
```

3. **Run with Gunicorn**:
```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

### Option 2: Using Windows IIS

1. **Install Required Components**:
- Install IIS on Windows
- Install URL Rewrite Module
- Install WebPlatformInstaller
- Install Python for Windows

2. **Configure IIS**:
- Create new IIS website
- Set up application pool
- Configure FastCGI settings
- Set up web.config

3. **Set up web.config**:
```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
    <system.webServer>
        <handlers>
            <add name="Python FastCGI"
                 path="*"
                 verb="*"
                 modules="FastCgiModule"
                 scriptProcessor="C:\Python39\python.exe|C:\Python39\Lib\site-packages\wfastcgi.py"
                 resourceType="Unspecified"
                 requireAccess="Script" />
        </handlers>
    </system.webServer>
    <appSettings>
        <add key="PYTHONPATH" value="C:\inetpub\wwwroot\CryptoTradingWeb" />
        <add key="WSGI_HANDLER" value="wsgi.app" />
    </appSettings>
</configuration>
```

### Security Considerations

1. **SSL/TLS**:
- Always use HTTPS in production
- Set up SSL certificate
- Configure secure headers

2. **Database**:
- Use a production-grade database (PostgreSQL recommended)
- Regular backups
- Secure database credentials

3. **API Security**:
- Rate limiting
- IP whitelisting
- Regular API key rotation

4. **Monitoring**:
- Set up application logging
- Monitor system resources
- Configure error notifications

5. **Updates**:
- Regular security updates
- Dependency updates
- System patches

### Maintenance

1. **Backup Strategy**:
```bash
# Database backup
sqlite3 instance/users.db .dump > backup.sql

# Configuration backup
cp .env .env.backup
```

2. **Log Rotation**:
- Configure log rotation for application logs
- Monitor disk space

3. **Performance Monitoring**:
- Monitor CPU usage
- Track memory consumption
- Watch database performance

4. **Scaling Considerations**:
- Load balancing setup
- Database replication
- Caching strategy

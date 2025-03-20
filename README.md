# Crypto Trading Web Application

A web-based cryptocurrency trading application with automated trading strategies and real-time monitoring.

## Features

- User authentication and registration
- Real-time trading dashboard
- Paper trading mode
- Multiple cryptocurrency pair monitoring
- Performance tracking and analytics
- Secure API key management

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key
```

4. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

## Running the Application

### Development
```bash
python app.py
```

### Production
```bash
gunicorn wsgi:app
```

## Directory Structure
```
CryptoTradingWeb/
├── app.py                 # Main application file
├── multi_symbol_strategy.py  # Trading strategy implementation
├── wsgi.py               # WSGI entry point
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── instance/           # SQLite database location
│   └── users.db
├── static/             # Static assets
└── templates/          # HTML templates
    ├── base.html
    ├── index.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    └── api_settings.html
```

## Security Notes

- Never commit `.env` file or API credentials
- Use HTTPS in production
- Regularly update dependencies
- Monitor system logs for suspicious activities

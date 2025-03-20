from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from multi_symbol_strategy import MultiSymbolStrategy
import threading
import json

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__, 
                template_folder=os.path.abspath('templates'),
                static_folder=os.path.abspath('static'))
    
    app.config['SECRET_KEY'] = 'your-secret-key-here'  # Fixed secret key
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    return app

app = create_app()

# User model for database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    api_key = db.Column(db.String(200))
    api_secret = db.Column(db.String(200))
    active_bots = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'})
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        return jsonify({'error': 'Invalid username or password'})
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_data = {
        'active_bots': current_user.active_bots,
        'api_configured': bool(current_user.api_key and current_user.api_secret)
    }
    
    if current_user.id in user_bots:
        bot_data = user_bots[current_user.id]
        user_data.update({
            'balance': bot_data['balance'],
            'trades': bot_data['trades']
        })
    
    return render_template('dashboard.html', user_data=user_data)

@app.route('/api-settings', methods=['GET', 'POST'])
@login_required
def api_settings():
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        api_secret = request.form.get('api_secret')
        
        current_user.api_key = api_key
        current_user.api_secret = api_secret
        db.session.commit()
        
        return redirect(url_for('dashboard'))
    
    return render_template('api_settings.html')

@app.route('/start-bot', methods=['POST'])
@login_required
def start_trading():
    if not current_user.api_key or not current_user.api_secret:
        return jsonify({'error': 'API credentials not configured'})
    
    start_bot(current_user.id)
    return jsonify({'success': True})

@app.route('/stop-bot', methods=['POST'])
@login_required
def stop_trading():
    stop_bot(current_user.id)
    return jsonify({'success': True})

@app.route('/bot-status')
@login_required
def bot_status():
    if current_user.id in user_bots:
        bot_data = user_bots[current_user.id]
        return jsonify({
            'running': bot_data['running'],
            'trades': bot_data['trades'],
            'balance': bot_data['balance']
        })
    return jsonify({
        'running': False,
        'trades': [],
        'balance': 10000.0
    })

@app.route('/logout')
@login_required
def logout():
    stop_bot(current_user.id)
    logout_user()
    return redirect(url_for('index'))

# Trading bot instance storage
user_bots = {}

def create_bot_instance(user_id, api_key, api_secret):
    if user_id in user_bots:
        stop_bot(user_id)
    bot = MultiSymbolStrategy(api_key, api_secret)
    user_bots[user_id] = {
        'bot': bot,
        'thread': None,
        'running': False,
        'trades': [],
        'balance': 10000.0  # Initial paper trading balance
    }
    return bot

def start_bot(user_id):
    if user_id not in user_bots:
        user = User.query.get(user_id)
        create_bot_instance(user_id, user.api_key, user.api_secret)
    
    bot_data = user_bots[user_id]
    if not bot_data['running']:
        def run_bot():
            bot_data['bot'].run_paper_trading()
        
        thread = threading.Thread(target=run_bot)
        thread.daemon = True
        thread.start()
        
        bot_data['thread'] = thread
        bot_data['running'] = True
        
        user = User.query.get(user_id)
        user.active_bots = True
        db.session.commit()

def stop_bot(user_id):
    if user_id in user_bots:
        bot_data = user_bots[user_id]
        bot_data['running'] = False
        if bot_data['thread']:
            bot_data['thread'].join(timeout=1)
        user_bots[user_id]['thread'] = None
        
        user = User.query.get(user_id)
        user.active_bots = False
        db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='127.0.0.1', port=5000, debug=True)

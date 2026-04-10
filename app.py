# app.py
from flask import Flask, render_template, request, jsonify, g
import sqlite3, os, joblib, numpy as np, pandas as pd
from datetime import datetime

DB_PATH = 'expense.db'
MODEL_PATH = 'models/rf_model.pkl'   

app = Flask(__name__, static_folder='static', template_folder='templates')

def get_db():  
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/add', methods=['POST'])
def add_expense():
    data = request.json
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO expenses (date, category, amount, notes) VALUES (?,?,?,?)",
        (data.get('date'), data.get('category'), float(data.get('amount')), data.get('notes',''))
    )
    conn.commit()
    return jsonify({"status":"ok"}), 201

@app.route('/api/expenses', methods=['GET'])
def get_expenses():
    conn = get_db()
    df = pd.read_sql_query("SELECT id, date, category, amount, notes FROM expenses ORDER BY date DESC LIMIT 500", conn)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/monthly', methods=['GET'])
def monthly():
    conn = get_db()
    df = pd.read_sql_query("SELECT date, amount FROM expenses", conn)
    if df.empty: return jsonify([])
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.set_index('date').resample('M')['amount'].sum().reset_index()
    monthly['label'] = monthly['date'].dt.strftime('%Y-%m')
    return jsonify(monthly[['label','amount']].to_dict(orient='records'))

@app.route('/api/category', methods=['GET'])
def category():
    conn = get_db()
    df = pd.read_sql_query("SELECT category, SUM(amount) as total FROM expenses GROUP BY category", conn)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/predict', methods=['GET'])
def predict():
    conn = get_db()
    df = pd.read_sql_query("SELECT date, amount FROM expenses", conn)
    if df.empty: return jsonify({"error":"no data"})
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.set_index('date').resample('M')['amount'].sum().reset_index()
    # Use model if exists and data enough, else fallback mean(last3)
    if os.path.exists(MODEL_PATH) and len(monthly) >= 4:
        model = joblib.load(MODEL_PATH)
        last3 = monthly['amount'].tail(3).values
        feat = np.array(last3).reshape(1,-1)
        pred = float(model.predict(feat)[0])
    else:
        pred = float(monthly['amount'].tail(3).mean())
    current = float(monthly['amount'].iloc[-1])
    advice = "Within usual range."
    if pred > current * 1.15:
        advice = "Warning: Predicted overspend next month. Reduce discretionary spending."
    return jsonify({"predicted_next_month": round(pred,2), "advice": advice})

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)

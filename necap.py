# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import os
import sys
import time
import json
import threading
import gc
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

sys.path.insert(0, '/app')

try:
    from yidun_proxyless import *
    import yidun_proxyless as solver
    SOLVER_AVAILABLE = True
    print("✅ CN31 Solver loaded successfully")
except ImportError as e:
    SOLVER_AVAILABLE = False
    print(f"❌ CN31 Solver not available: {e}")

# RAILWAY OPTIMIZED CONFIG
BATCH_SIZE = 10
MAX_WORKERS = 7  # Sweet spot for 10 tokens/3s
BATCH_INTERVAL = 3.0

solver_running = False
solver_thread = None
token_cache = []
token_lock = threading.Lock()
stats = {
    "status": "idle",
    "tokens_generated": 0,
    "tokens_available": 0,
    "tokens_per_second": 0,
    "start_time": None,
    "last_batch_time": None,
    "threads": MAX_WORKERS
}

TOKEN_FILE = "/app/validated_tokens.txt"

def read_tokens_from_file():
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
        return []
    except:
        return []

def get_token_count():
    with token_lock:
        return len(token_cache)

def add_tokens_to_cache(tokens):
    with token_lock:
        token_cache.extend(tokens)
        stats["tokens_generated"] = len(token_cache)
        try:
            with open(TOKEN_FILE, 'a') as f:
                for token in tokens:
                    f.write(f"{token}\n")
        except:
            pass

def get_tokens_from_cache(n=1):
    with token_lock:
        if len(token_cache) >= n:
            result = token_cache[:n]
            token_cache = token_cache[n:]
            return result
        return []

def run_solver_worker():
    global solver_running, stats
    
    print("🚀 Starting 10 Tokens/3s Generator...")
    stats["status"] = "running"
    stats["start_time"] = datetime.now().isoformat()
    
    batch_count = 0
    total_tokens = 0
    
    while solver_running:
        try:
            batch_start = time.time()
            current_count = get_token_count()
            
            # Generate if we have less than 30 tokens buffer
            if current_count < 30:
                batch_tokens = []
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []
                    for i in range(BATCH_SIZE):
                        config = {
                            'ID_': ID,
                            'REFERER': REFERER,
                            'FP_H': FP_H,
                            'UA': UserAgent().random,
                            'DOMAIN': DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
                        }
                        future = executor.submit(run_single_token, i+1, config)
                        futures.append(future)
                    
                    for future in as_completed(futures, timeout=12):
                        try:
                            token = future.result(timeout=6)
                            if token:
                                batch_tokens.append(token)
                        except:
                            continue
                
                if batch_tokens:
                    add_tokens_to_cache(batch_tokens)
                    total_tokens += len(batch_tokens)
                    batch_count += 1
                    
                    batch_time = time.time() - batch_start
                    stats["tokens_per_second"] = len(batch_tokens) / batch_time if batch_time > 0 else 0
                    stats["last_batch_time"] = datetime.now().isoformat()
                    
                    print(f"Batch #{batch_count}: {len(batch_tokens)} tokens | Total: {total_tokens} | Cache: {get_token_count()}")
                    
                    # GC every 5 batches
                    if batch_count % 5 == 0:
                        gc.collect()
                    
                    # Maintain 3s interval
                    if batch_time < BATCH_INTERVAL:
                        time.sleep(BATCH_INTERVAL - batch_time)
            else:
                # Enough tokens, sleep briefly
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Batch error: {e}")
            time.sleep(2)
    
    stats["status"] = "stopped"

@app.route('/')
def status():
    return jsonify({
        "status": stats["status"],
        "tokens_generated": stats["tokens_generated"],
        "tokens_available": get_token_count(),
        "tokens_per_second": stats["tokens_per_second"],
        "threads": stats["threads"],
        "start_time": stats.get("start_time"),
        "last_batch_time": stats.get("last_batch_time"),
        "solver_available": SOLVER_AVAILABLE
    })

@app.route('/health')
def health():
    return jsonify({
        "ok": True,
        "solver_available": SOLVER_AVAILABLE,
        "status": stats["status"],
        "tokens_available": get_token_count()
    })

@app.route('/start', methods=['POST'])
def start_solver():
    global solver_running, solver_thread, stats
    
    if solver_running:
        return jsonify({"error": "Solver already running"}), 400
    
    if not SOLVER_AVAILABLE:
        return jsonify({"error": "CN31 Solver not available"}), 500
    
    data = request.json or {}
    threads = min(data.get("threads", MAX_WORKERS), 8)
    stats["threads"] = threads
    
    model = initialize_global_model()
    if model is None:
        return jsonify({"error": "Failed to load model"}), 500
    
    solver_running = True
    
    solver_thread = threading.Thread(
        target=run_solver_worker,
        daemon=True
    )
    solver_thread.start()
    
    return jsonify({
        "message": "10 Tokens/3s generator started",
        "threads": threads
    })

@app.route('/stop', methods=['POST'])
def stop_solver():
    global solver_running
    
    solver_running = False
    stats["status"] = "stopping"
    
    return jsonify({
        "message": "Stop signal sent",
        "tokens_generated": stats["tokens_generated"],
        "tokens_available": get_token_count()
    })

@app.route('/api/get-token', methods=['GET'])
def get_token():
    tokens = get_tokens_from_cache(1)
    
    if tokens:
        return jsonify({
            "token": tokens[0],
            "remaining": get_token_count()
        })
    
    # Generate one immediately
    try:
        config = {
            'ID_': ID,
            'REFERER': REFERER,
            'FP_H': FP_H,
            'UA': UserAgent().random,
            'DOMAIN': DUN163_DOMAINS[0]
        }
        token = run_single_token(1, config)
        if token:
            add_tokens_to_cache([token])
            return jsonify({
                "token": token,
                "remaining": get_token_count()
            })
    except:
        pass
    
    return jsonify({"error": "No tokens available"}), 404

@app.route('/api/tokens', methods=['GET'])
def get_tokens():
    n = request.args.get('n', 10, type=int)
    n = min(n, 50)
    
    tokens = get_tokens_from_cache(n)
    
    if tokens:
        return jsonify({
            "tokens": tokens,
            "count": len(tokens),
            "remaining": get_token_count()
        })
    
    # Generate some immediately
    try:
        batch_tokens = []
        with ThreadPoolExecutor(max_workers=min(n, 7)) as executor:
            futures = []
            for i in range(min(n, 7)):
                config = {
                    'ID_': ID,
                    'REFERER': REFERER,
                    'FP_H': FP_H,
                    'UA': UserAgent().random,
                    'DOMAIN': DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
                }
                future = executor.submit(run_single_token, i+1, config)
                futures.append(future)
            
            for future in as_completed(futures, timeout=10):
                try:
                    token = future.result(timeout=5)
                    if token:
                        batch_tokens.append(token)
                except:
                    continue
        
        if batch_tokens:
            add_tokens_to_cache(batch_tokens)
            return jsonify({
                "tokens": batch_tokens,
                "count": len(batch_tokens),
                "remaining": get_token_count()
            })
    except:
        pass
    
    return jsonify({"error": "No tokens available"}), 404

@app.route('/api/stats')
def api_stats():
    return jsonify({
        "total_generated": stats["tokens_generated"],
        "available": get_token_count(),
        "rate": stats["tokens_per_second"],
        "status": stats["status"],
        "threads": stats["threads"]
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  CN31 Solver - 10 Tokens/3s Mode                            ║
╠══════════════════════════════════════════════════════════════╣
║  Port       : {port}                                           ║
║  Target     : 10 tokens per 3 seconds                        ║
║  Threads    : {MAX_WORKERS}                                      ║
║  Solver     : {'✅ Available' if SOLVER_AVAILABLE else '❌ Not Available'} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

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

# Configuration
BATCH_SIZE = 20
TARGET_RATE = 20  # tokens per 3 seconds
BATCH_INTERVAL = 3.0  # seconds

solver_running = False
solver_thread = None
token_cache = []
token_lock = threading.Lock()
token_history = []
stats = {
    "status": "idle",
    "tokens_generated": 0,
    "tokens_available": 0,
    "tokens_per_second": 0,
    "start_time": None,
    "last_batch_time": None,
    "threads": 10
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
    """Get current token count"""
    with token_lock:
        return len(token_cache)

def add_tokens_to_cache(tokens):
    """Add tokens to cache"""
    with token_lock:
        token_cache.extend(tokens)
        stats["tokens_generated"] = len(token_cache)
        
        # Also write to file
        try:
            with open(TOKEN_FILE, 'a') as f:
                for token in tokens:
                    f.write(f"{token}\n")
        except:
            pass

def get_tokens_from_cache(n=1):
    """Get tokens from cache"""
    with token_lock:
        if len(token_cache) >= n:
            result = token_cache[:n]
            token_cache = token_cache[n:]
            return result
        return []

def run_solver_worker():
    """Run solver in background with batch processing"""
    global solver_running, stats
    
    print("🚀 Starting high-performance token generator...")
    stats["status"] = "running"
    stats["start_time"] = datetime.now().isoformat()
    
    batch_count = 0
    total_tokens = 0
    
    while solver_running:
        try:
            batch_start = time.time()
            
            # Check if we have enough tokens
            current_count = get_token_count()
            
            if current_count < 50:  # Keep buffer of 50 tokens
                # Generate batch of tokens
                batch_tokens = []
                
                # Use multiple threads for parallel generation
                with ThreadPoolExecutor(max_workers=stats["threads"]) as executor:
                    futures = []
                    for i in range(BATCH_SIZE):
                        # Use fresh config for each request
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
                    total_tokens += len(batch_tokens)
                    batch_count += 1
                    
                    # Update stats
                    batch_time = time.time() - batch_start
                    stats["tokens_per_second"] = len(batch_tokens) / batch_time if batch_time > 0 else 0
                    stats["last_batch_time"] = datetime.now().isoformat()
                    
                    print(f"Batch #{batch_count}: {len(batch_tokens)} tokens in {batch_time:.2f}s "
                          f"| Total: {total_tokens} | Cache: {get_token_count()}")
                    
                    # Sleep to maintain rate
                    if batch_time < BATCH_INTERVAL:
                        time.sleep(BATCH_INTERVAL - batch_time)
            else:
                # Enough tokens, sleep
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Batch error: {e}")
            time.sleep(1)
    
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
        "tokens_available": get_token_count(),
        "target_rate": f"{TARGET_RATE} tokens/3s"
    })

@app.route('/start', methods=['POST'])
def start_solver():
    global solver_running, solver_thread, stats
    
    if solver_running:
        return jsonify({"error": "Solver already running"}), 400
    
    if not SOLVER_AVAILABLE:
        return jsonify({"error": "CN31 Solver not available"}), 500
    
    data = request.json or {}
    threads = min(data.get("threads", 10), 20)  # Max 20 threads
    stats["threads"] = threads
    
    # Initialize model
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
        "message": "Token generator started",
        "threads": threads,
        "target_rate": f"{TARGET_RATE} tokens/3s"
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
    """Get single token"""
    tokens = get_tokens_from_cache(1)
    
    if tokens:
        return jsonify({
            "token": tokens[0],
            "remaining": get_token_count()
        })
    
    # Try to generate one immediately
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
    """Get multiple tokens"""
    n = request.args.get('n', 10, type=int)
    n = min(n, 50)  # Max 50 per request
    
    tokens = get_tokens_from_cache(n)
    
    if tokens:
        return jsonify({
            "tokens": tokens,
            "count": len(tokens),
            "remaining": get_token_count()
        })
    
    # Try to generate some immediately
    try:
        batch_tokens = []
        with ThreadPoolExecutor(max_workers=min(n, 10)) as executor:
            futures = []
            for i in range(min(n, 10)):
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

@app.route('/api/batch', methods=['GET'])
def get_batch():
    """Get batch of tokens - optimized for high throughput"""
    n = request.args.get('n', 20, type=int)
    n = min(n, 50)
    
    tokens = get_tokens_from_cache(n)
    
    if len(tokens) >= n:
        return jsonify({
            "tokens": tokens,
            "count": len(tokens),
            "remaining": get_token_count(),
            "fast": True
        })
    
    # If not enough cache, generate more
    needed = n - len(tokens)
    try:
        batch_tokens = []
        with ThreadPoolExecutor(max_workers=min(needed, 10)) as executor:
            futures = []
            for i in range(min(needed, 10)):
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
            # Get from cache again
            additional = get_tokens_from_cache(batch_tokens)
            all_tokens = tokens + additional
            return jsonify({
                "tokens": all_tokens,
                "count": len(all_tokens),
                "remaining": get_token_count(),
                "fast": len(all_tokens) >= n
            })
    except:
        pass
    
    if tokens:
        return jsonify({
            "tokens": tokens,
            "count": len(tokens),
            "remaining": get_token_count()
        })
    
    return jsonify({"error": "No tokens available"}), 404

@app.route('/api/stats')
def api_stats():
    return jsonify({
        "total_generated": stats["tokens_generated"],
        "available": get_token_count(),
        "rate": stats["tokens_per_second"],
        "status": stats["status"],
        "threads": stats["threads"],
        "target": f"{TARGET_RATE} tokens/3s"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    
    print(f"""
🔐 CN31 Solver - High Performance
─────────────────────────────────────────
Port       : {port}
Target Rate: {TARGET_RATE} tokens/3s
Threads    : {stats['threads']}
Solver     : {'✅ Available' if SOLVER_AVAILABLE else '❌ Not Available'}
""")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
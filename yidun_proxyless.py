# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import json
import os
import random
import re
import string
import time
import warnings
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import execjs
from loguru import logger
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import queue
from functools import lru_cache
from fake_useragent import UserAgent
import gc

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
warnings.filterwarnings("ignore", message=".*SIFT_create.*deprecated.*")

DEBUG = False

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = False
DEVICE = 'cpu'

TOKEN_SERVER_URL = os.environ.get('TOKEN_SERVER_URL', 'https://akamai-storage.onrender.com')
TOKEN_SAVE_ENDPOINT = f"{TOKEN_SERVER_URL}/api/save-token"

# Fast token queue
token_queue = queue.Queue()
token_lock = threading.Lock()
token_counter = 0
TOKEN_OUTPUT_FILE = os.path.join(DIR_PATH, 'validated_tokens.txt')

# Connection pooling
session_pool = []
session_lock = threading.Lock()
MAX_SESSIONS = 15

def get_pooled_session():
    """Get or create a pooled session for faster connections"""
    with session_lock:
        if session_pool:
            return session_pool.pop()
        else:
            session = requests.Session()
            session.headers.update({
                "Accept": "*/*",
                "Accept-Language": "*",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            })
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=20,
                pool_maxsize=20,
                max_retries=2
            )
            session.mount('https://', adapter)
            session.mount('http://', adapter)
            return session

def return_pooled_session(session):
    """Return session to pool"""
    with session_lock:
        if len(session_pool) < MAX_SESSIONS:
            session_pool.append(session)

def send_token_to_server(token):
    try:
        payload = {"token": token}
        r = requests.post(TOKEN_SAVE_ENDPOINT, json=payload, timeout=2)
        return r.status_code in [200, 201]
    except:
        return False

def save_token_locally(token):
    try:
        line = f"{token}\n"
        with open(TOKEN_OUTPUT_FILE, 'a') as f:
            f.write(line)
        return True
    except:
        return False

def add_token_to_queue(token):
    global token_counter
    with token_lock:
        token_queue.put(token)
        token_counter += 1
        # Fast local save
        save_token_locally(token)
        # Async server send
        try:
            threading.Thread(target=send_token_to_server, args=(token,), daemon=True).start()
        except:
            pass

def get_batch_tokens(batch_size=15):
    tokens = []
    try:
        for _ in range(batch_size):
            token = token_queue.get(timeout=0.5)
            tokens.append(token)
    except queue.Empty:
        pass
    return tokens

def get_token_count():
    with token_lock:
        return token_counter

# Model - Fast loading with memory optimization
_model_state = None
_model_lock = threading.Lock()

def initialize_global_model():
    global _model_state
    if _model_state is not None:
        return _model_state
    with _model_lock:
        if _model_state is not None:
            return _model_state
        model_path = os.path.join(DIR_PATH, 'net.pkl')
        if not os.path.exists(model_path):
            logger.error("Model file net.pkl not found")
            return None
        try:
            state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            if 'net' in state:
                state['net'] = state['net'].to('cpu')
                state['net'].eval()
                state['net'] = state['net'].half()
            _model_state = state
            gc.collect()
            logger.success("Model loaded on CPU")
            return _model_state
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return None

def get_global_model():
    global _model_state
    if _model_state is None:
        return initialize_global_model()
    return _model_state

@lru_cache(maxsize=3)
def get_compiled_js_cached(file_name):
    try:
        js_path = os.path.join(DIR_PATH, file_name)
        with open(js_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
        ctx = execjs.compile(js_code)
        return ctx
    except:
        return None

def get_compiled_js(file_name):
    return get_compiled_js_cached(file_name)

_sift_detector = None
_sift_lock = threading.Lock()

def get_sift_detector():
    global _sift_detector
    if _sift_detector is None:
        with _sift_lock:
            if _sift_detector is None:
                try:
                    _sift_detector = cv2.SIFT_create(nfeatures=20, contrastThreshold=0.12)
                except AttributeError:
                    try:
                        _sift_detector = cv2.xfeatures2d.SIFT_create(nfeatures=20, contrastThreshold=0.12)
                    except AttributeError:
                        _sift_detector = cv2.ORB_create(nfeatures=20)
    return _sift_detector

REFERER = "https://mtacc.mobilelegends.com/"
ID = "fef5c67c39074e9d845f4bf579cc07af"
FP_H = "mtacc.mobilelegends.com"

DUN163_DOMAINS = [
    "https://c.dun.163.com",
    "https://c.dun.163yun.com"
]

# Fast helper functions
def safe_list_access(lst, index, default=None):
    try:
        if lst is None or not isinstance(lst, (list, tuple)):
            return default
        if not (0 <= index < len(lst)):
            return default
        return lst[index]
    except:
        return default

def rotate_about_center(src, angle, scale=1.):
    try:
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LINEAR)
    except:
        return src

def parse_y_pred(ypred, anchors, class_types, islist=False, threshold=0.2, nms_threshold=0):
    try:
        if not anchors or not class_types:
            return [] if islist else None
        ceillen = 5 + len(class_types)
        sigmoid = lambda x: 1/(1+math.exp(-x))
        infos = []
        for idx in range(min(len(anchors), 3)):
            try:
                tensor_idx = 4 + idx * ceillen
                if tensor_idx >= ypred.shape[3]:
                    continue
                a = ypred[:,:,:,tensor_idx].cpu().detach().numpy()
                for ii, i in enumerate(a[0]):
                    for jj, j in enumerate(i):
                        infos.append((ii, jj, idx, sigmoid(j)))
            except:
                continue
        if not infos:
            return [] if islist else None
        infos = sorted(infos, key=lambda i: -i[3])
        def get_xyxy_clz_con_emergency(info):
            try:
                gap = 416/ypred.shape[1]
                x, y, idx, con = info
                if idx >= len(anchors):
                    return None
                gp = idx * ceillen
                if (gp + 5 + len(class_types)) > ypred.shape[3]:
                    return None
                contain = torch.sigmoid(ypred[0, x, y, gp+4])
                pred_xy = torch.sigmoid(ypred[0, x, y, gp+0:gp+2])
                pred_wh = ypred[0, x, y, gp+2:gp+4]
                pred_clz = ypred[0, x, y, gp+5:gp+5+len(class_types)]
                pred_xy = pred_xy.cpu().detach().numpy()
                pred_wh = pred_wh.cpu().detach().numpy()
                pred_clz = pred_clz.cpu().detach().numpy()
                exp = math.exp
                cx, cy = float(pred_xy[0]), float(pred_xy[1])
                rx, ry = (cx + x)*gap, (cy + y)*gap
                rw, rh = float(pred_wh[0]), float(pred_wh[1])
                rw, rh = exp(rw)*anchors[idx][0], exp(rh)*anchors[idx][1]
                clz_ = [float(x) for x in pred_clz]
                xx = rx - rw/2
                _x = rx + rw/2
                yy = ry - rh/2
                _y = ry + rh/2
                log_cons = torch.sigmoid(ypred[:,:,:,gp+4]).cpu().detach().numpy()
                log_cons = np.transpose(log_cons, (0, 2, 1))
                clz = 'unknown'
                if clz_:
                    max_val = max(clz_)
                    max_idx = clz_.index(max_val)
                    for key, value in class_types.items():
                        if value == max_idx:
                            clz = key
                            break
                return [xx, yy, _x, _y], clz, con, log_cons
            except:
                return None
        if islist:
            limited_infos = infos[:min(50, len(infos))]
            v = []
            for i in limited_infos:
                if i[3] > threshold:
                    result = get_xyxy_clz_con_emergency(i)
                    if result is not None:
                        v.append(result)
            return v
        else:
            if infos:
                return get_xyxy_clz_con_emergency(infos[0])
            return None
    except Exception as e:
        return [] if islist else None

class Mini(nn.Module):
    class ConvBN(nn.Module):
        def __init__(self, cin, cout, kernel_size=3, stride=1, padding=None):
            super().__init__()
            padding = (kernel_size - 1) // 2 if not padding else padding
            self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(cout, momentum=0.01)
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
    def __init__(self, anchors, class_types, inchennel=3):
        super().__init__()
        self.oceil = len(anchors) * (5 + len(class_types))
        self.model = nn.Sequential(
            OrderedDict([
                ('ConvBN_0', self.ConvBN(inchennel, 32)),
                ('Pool_0', nn.MaxPool2d(2, 2)),
                ('ConvBN_1', self.ConvBN(32, 48)),
                ('Pool_1', nn.MaxPool2d(2, 2)),
                ('ConvBN_2', self.ConvBN(48, 64)),
                ('Pool_2', nn.MaxPool2d(2, 2)),
                ('ConvBN_3', self.ConvBN(64, 80)),
                ('Pool_3', nn.MaxPool2d(2, 2)),
                ('ConvBN_4', self.ConvBN(80, 96)),
                ('Pool_4', nn.MaxPool2d(2, 2)),
                ('ConvBN_5', self.ConvBN(96, 102)),
                ('ConvEND', nn.Conv2d(102, self.oceil, 1)),
            ])
        )
    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1)

def get_clz_rect_from_image(image_data, state):
    try:
        if not state or 'net' not in state:
            return [], None
        net = state['net']
        anchors = state.get('anchors', [])
        class_types = state.get('class_types', {})
        if not anchors or not class_types:
            return [], None
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        npimg = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if npimg is None:
            return [], None
        height, width = npimg.shape[:2]
        npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
        npimg = cv2.resize(npimg, (416, 416), interpolation=cv2.INTER_LINEAR)
        npimg_ = np.transpose(npimg, (2,1,0))
        with torch.no_grad():
            input_tensor = torch.FloatTensor(npimg_).unsqueeze(0).to(DEVICE)
            input_tensor = input_tensor.half()
            y_pred = net(input_tensor)
        v = parse_y_pred(y_pred, anchors, class_types, islist=True, threshold=0.2, nms_threshold=0.4)
        ret = []
        for i in v:
            if len(i) >= 4:
                rect, clz, con, log_cons = i[0], i[1], i[2], i[3]
                rw, rh = width/416, height/416
                rect[0] = int(rect[0]*rw)
                rect[2] = int(rect[2]*rw)
                rect[1] = int(rect[1]*rh)
                rect[3] = int(rect[3]*rh)
                ret.append([clz, rect])
        return ret, npimg
    except Exception as e:
        return [], None

def get_cut_img(npimg, rects):
    ret = []
    try:
        for item in rects:
            if len(item) >= 2:
                clz, rect = item[0], item[1]
                if len(rect) >= 4:
                    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(npimg.shape[1], x2), min(npimg.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        ret.append([clz, npimg[y1:y2,x1:x2,:], (x1,y1,x2,y2)])
    except:
        pass
    return ret

def get_flags_rects_from_image(image_data, state):
    try:
        if state is None:
            return None, None, None
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        s = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if s is None or s.size == 0:
            return None, None, None
        height, width = s.shape[:2]
        if height < 200 or width < 84:
            return None, None, None
        try:
            end_height = min(height, s.shape[0])
            a = s[160:end_height, 0:min(22, width), :]
            b = s[160:end_height, 28:min(50, width), :]
            c = s[160:end_height, 56:min(78, width), :]
            if a.shape[0] < 40 or a.shape[1] < 20:
                return None, None, None
            a1 = a[40:min(60, a.shape[0]), :, :] if a.shape[0] > 40 else a
            a2 = a[0:min(20, a.shape[0]), :, :] if a.shape[0] > 0 else a
            b1 = b[40:min(60, b.shape[0]), :, :] if b.shape[0] > 40 else b
            b2 = b[0:min(20, b.shape[0]), :, :] if b.shape[0] > 0 else b
            c1 = c[40:min(60, c.shape[0]), :, :] if c.shape[0] > 40 else c
            c2 = c[0:min(20, c.shape[0]), :, :] if c.shape[0] > 0 else c
        except:
            return None, None, None
        def get_match_lens_emergency(i1, i2):
            try:
                if i1.size == 0 or i2.size == 0:
                    return 0
                i1 = cv2.resize(i1, (min(i1.shape[1]*3, 600), min(i1.shape[0]*3, 450)), interpolation=cv2.INTER_LINEAR)
                i2 = cv2.resize(i2, (min(i2.shape[1]*2, 400), min(i2.shape[0]*2, 300)), interpolation=cv2.INTER_LINEAR)
                sift = get_sift_detector()
                kp1, des1 = sift.detectAndCompute(i1, None)
                kp2, des2 = sift.detectAndCompute(i2, None)
                if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                    return 0
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = 0
                for match_pair in matches:
                    if len(match_pair) >= 2:
                        m, n = match_pair[0], match_pair[1]
                        if m.distance <= 0.88 * n.distance:
                            good += 1
                return good
            except:
                return 0
        def get_flag_rect_emergency(k12, cut_imgs, st):
            try:
                if len(k12) < 2:
                    return []
                k1, k2 = k12[0], k12[1]
                r = []
                for item in cut_imgs:
                    if len(item) >= 3:
                        clz, npimg, rect = item[0], item[1], item[2]
                        if clz == '1':
                            r1 = get_match_lens_emergency(k1, npimg)
                            r.append([r1, rect, st])
                        elif clz == '2':
                            r2 = get_match_lens_emergency(k2, npimg)
                            r.append([r2, rect, st])
                return sorted(r, key=lambda i: i[0]) if r else []
            except:
                return []
        try:
            rects, processed_img = get_clz_rect_from_image(image_data, state)
            if not rects:
                return None, None, None
            v = get_cut_img(s, rects)
            if len(v) == 0:
                return None, None, None
            rs1 = get_flag_rect_emergency([a1, a2], v, 1)
            rs2 = get_flag_rect_emergency([b1, b2], v, 2)
            rs3 = get_flag_rect_emergency([c1, c2], v, 3)
            rs = rs1 + rs2 + rs3
            if len(rs) < 3:
                return None, None, None
            r = []
            used_types = set()
            for target_type in [1, 2, 3]:
                candidates = [x for x in rs if len(x) >= 3 and x[2] == target_type]
                if candidates:
                    best = max(candidates, key=lambda x: x[0])
                    r.append(best)
                    used_types.add(target_type)
            if len(r) >= 3:
                r = sorted(r[:3], key=lambda x: x[2])
                return r[0][1], r[1][1], r[2][1]
            else:
                return None, None, None
        except Exception as e:
            return None, None, None
    except Exception as e:
        return None, None, None

class Dun163:
    def __init__(self, id_, *, referer, fp_h, ua, thread_id, domain=None):
        self.fp = None
        self.resp_json2 = None
        self.domain = domain if domain else random.choice(DUN163_DOMAINS)
        self.thread_id = thread_id
        self._current_image_data = None
        self._current_rects = None
        self._current_click_points = None
        self.request_params = {
            'id': id_,
            'referer': referer,
            'fp_h': fp_h,
            'ua': ua
        }
        self.ss = self.set_session()
        self.ctx = get_compiled_js('dun163.js')

    def set_session(self):
        session = get_pooled_session()
        domain_host = self.domain.replace('https://', '').replace('http://', '')
        session.headers.update({
            "Referer": self.request_params['referer'],
            "User-Agent": self.request_params['ua'],
            "Host": domain_host,
            "X-Forwarded-For": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "X-Real-IP": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        })
        session.timeout = (3, 6)
        return session

    @staticmethod
    @lru_cache(maxsize=500)
    def get_jsonp(text):
        try:
            jsonp_str = re.search(r"\((.*)\)", text, re.S)
            if jsonp_str:
                return json.loads(jsonp_str.group(1))
            return {}
        except:
            return {}

    def request_getconf(self):
        try:
            url = self.domain + '/api/v2/getconf'
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "",
                "dt": "",
                "id": self.request_params['id'],
                "ipv6": "false",
                "runEnv": "10",
                "iv": "5",
                "loadVersion": "2.5.3",
                "lang": "en-US",
                "callback": "__JSONP_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7)) + '_0'
            }
            response = self.ss.get(url, params=params)
            response.raise_for_status()
            resp_json = self.get_jsonp(response.text)
            return resp_json.get('data', {})
        except:
            return {}

    def request_get(self, dt, bid, ac_token, ir_token=None):
        try:
            url = self.domain + '/api/v3/get'
            fp = self.ctx.call('get_fp', self.request_params['fp_h'], self.request_params['ua'])
            cb = self.ctx.call('get_cb')
            self.fp = fp
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31",
                "dt": dt,
                "id": bid,
                "fp": fp,
                "https": "true",
                "type": "",
                "version": "2.28.5",
                "dpr": "1",
                "dev": "1",
                "cb": cb,
                "ipv6": "false",
                "runEnv": "10",
                "group": "",
                "scene": "",
                "lang": "en-US",
                "sdkVersion": "",
                "loadVersion": "2.5.3",
                "iv": "4",
                "user": "",
                "width": "320",
                "audio": "false",
                "sizeType": "10",
                "smsVersion": "v3",
                "token": "",
                "callback": "__JSONP_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7)) + '_0'
            }
            if ir_token:
                params["irToken"] = ir_token
            resp_text = self.ss.get(url, params=params).text
            resp_json = self.get_jsonp(resp_text)
            return resp_json.get('data', {})
        except:
            return {}

    def request_check(self, dt, bid, *, token, captcha_type=7, click_data=None):
        try:
            url = self.domain + '/api/v3/check'
            if captcha_type == 7 and click_data:
                check_data = self.ctx.call('get_click_check_data', click_data, token)
            else:
                check_data = '{"d":"","m":"","p":"","ext":""}'
            cb = self.ctx.call('get_cb')
            params = {
                "referer": self.request_params['referer'],
                "zoneId": "CN31",
                "dt": dt,
                "id": bid,
                "token": token,
                "data": check_data,
                "width": "320",
                "type": str(captcha_type),
                "version": "2.28.5",
                "cb": cb,
                "user": "",
                "extraData": "",
                "bf": "0",
                "runEnv": "10",
                "sdkVersion": "",
                "loadVersion": "2.5.3",
                "iv": "4",
                "callback": "__JSONP_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7)) + '_1'
            }
            resp = self.ss.get(url, params=params)
            resp_json = self.get_jsonp(resp.text)
            return resp_json.get('data', {}), 0.0
        except:
            return {}, 0.0

    def handle_click_captcha_hybrid(self, bg_url, token, attempt_num=0):
        try:
            headers = {"User-Agent": self.request_params['ua']}
            resp = requests.get(bg_url, headers=headers, timeout=6)
            resp.raise_for_status()
            image_data = resp.content
            state = get_global_model()
            if not state:
                return self.generate_emergency_clicks(), 0.0
            rects = get_flags_rects_from_image(image_data, state)
            rect1 = safe_list_access(rects, 0)
            rect2 = safe_list_access(rects, 1)
            rect3 = safe_list_access(rects, 2)
            if rect1 is not None and rect2 is not None and rect3 is not None:
                click_points = []
                for rect in [rect1, rect2, rect3]:
                    if rect and len(rect) >= 4:
                        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        offset_x = random.randint(-1, 1)
                        offset_y = random.randint(-1, 1)
                        final_x = max(5, min(center_x + offset_x, 315))
                        final_y = max(5, min(center_y + offset_y, 195))
                        click_points.append({"x": final_x, "y": final_y})
                if len(click_points) >= 3:
                    self._current_click_points = click_points[:3]
                    return click_points[:3], 0.0
            click_points = self.generate_emergency_clicks()
            self._current_click_points = click_points
            return click_points, 0.0
        except Exception as e:
            return self.generate_emergency_clicks(), 0.0

    def generate_emergency_clicks(self):
        try:
            patterns = [
                [(80, 70), (160, 120), (240, 90)],
                [(70, 100), (160, 95), (250, 105)],
                [(160, 60), (110, 130), (210, 140)],
            ]
            selected_pattern = random.choice(patterns)
            click_points = []
            for x, y in selected_pattern:
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                final_x = max(15, min(x + offset_x, 305))
                final_y = max(15, min(y + offset_y, 185))
                click_points.append({"x": final_x, "y": final_y})
            return click_points
        except:
            return [{"x": 80, "y": 70}, {"x": 160, "y": 120}, {"x": 240, "y": 90}]

    def run_single(self, attempt_num=0):
        try:
            get_conf_data = self.request_getconf()
            if not get_conf_data:
                return None
            
            dt = get_conf_data.get('dt')
            ac_data = get_conf_data.get('ac', {})
            ac_token = ac_data.get('token')
            bid = ac_data.get('bid')
            
            ir_data = get_conf_data.get('ir', {})
            ir_token = ir_data.get('token') if ir_data.get('enable') else None
            
            get_data = self.request_get(dt, bid, ac_token, ir_token)
            if not get_data:
                return None
            
            captcha_type = get_data.get('type', 7)
            token = get_data.get('token')
            
            if not token:
                return None
            
            if captcha_type == 7:
                bg_urls = get_data.get('bg', [])
                if not bg_urls:
                    return None
                
                click_points, _ = self.handle_click_captcha_hybrid(bg_urls[0], token, attempt_num)
                resp_json, _ = self.request_check(dt, bid, token=token, captcha_type=7, click_data=click_points)
            else:
                return None
            
            self.resp_json2 = resp_json
            
            if resp_json.get('result') == True:
                validate_raw = resp_json.get('validate', '')
                validate_decoded = ""
                
                if validate_raw and self.ctx:
                    try:
                        validate_decoded = self.ctx.call('do_onVerify', validate_raw, self.fp)
                    except Exception as e:
                        return None
                
                if validate_decoded and len(validate_decoded.strip()) > 10:
                    return validate_decoded
                
            return None
            
        except Exception as e:
            return None

def run_single_token(thread_id, config):
    try:
        d = Dun163(
            id_=config['ID_'],
            referer=config['REFERER'],
            fp_h=config['FP_H'],
            ua=config['UA'],
            thread_id=thread_id,
            domain=config['DOMAIN']
        )
        return d.run_single()
    except Exception as e:
        return None

def main_batch():
    logger.info("Starting CN31 Solver - 10 Tokens/3s Mode")
    
    model_state = initialize_global_model()
    if not model_state:
        logger.error("Model not available")
        return
    
    js_ctx = get_compiled_js('dun163.js')
    if not js_ctx:
        logger.error("JavaScript not available")
        return
    
    logger.success("Resources loaded")
    
    config = {
        'ID_': ID,
        'REFERER': REFERER,
        'FP_H': FP_H,
        'UA': UserAgent().random,
        'DOMAIN': DUN163_DOMAINS[0]
    }
    
    # OPTIMIZED: 7 workers for 10 tokens/3s
    BATCH_SIZE = 10
    NUM_WORKERS = 7
    
    logger.info(f"Workers: {NUM_WORKERS}, Batch: {BATCH_SIZE}")
    logger.info(f"Target: 10 tokens per 3 seconds")
    logger.info("-" * 50)
    
    token_count = 0
    batch_count = 0
    
    while True:
        try:
            batch_start = time.time()
            batch_tokens = []
            
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = []
                for i in range(BATCH_SIZE):
                    thread_config = config.copy()
                    thread_config['UA'] = UserAgent().random
                    thread_config['DOMAIN'] = DUN163_DOMAINS[i % len(DUN163_DOMAINS)]
                    future = executor.submit(run_single_token, i+1, thread_config)
                    futures.append(future)
                
                for future in as_completed(futures, timeout=10):
                    try:
                        token = future.result(timeout=5)
                        if token:
                            batch_tokens.append(token)
                    except:
                        continue
            
            batch_time = time.time() - batch_start
            token_count += len(batch_tokens)
            batch_count += 1
            
            for token in batch_tokens:
                add_token_to_queue(token)
            
            rate = len(batch_tokens) / batch_time if batch_time > 0 else 0
            logger.success(
                f"Batch #{batch_count}: {len(batch_tokens)} tokens in {batch_time:.2f}s "
                f"(~{rate:.1f} tokens/s) | Total: {token_count}"
            )
            
            # Garbage collection every 3 batches
            if batch_count % 3 == 0:
                gc.collect()
            
            # Wait to maintain 3s interval
            if batch_time < 3.0:
                time.sleep(3.0 - batch_time)
            
        except KeyboardInterrupt:
            logger.warning("Stopping...")
            break
        except Exception as e:
            logger.error(f"Batch error: {e}")
            time.sleep(2)

def main():
    main_batch()

if __name__ == '__main__':
    main()
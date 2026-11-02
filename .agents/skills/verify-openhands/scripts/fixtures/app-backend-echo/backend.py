#!/usr/bin/python3
"""Owned no-dependency HTTP/WS echo fixture; no credential or environment dumps."""
import base64
import hashlib
import json
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = b'''<!doctype html><meta charset="utf-8"><title>QA bridge backend</title>
<h1>QA real HTTP and WebSocket backend</h1><pre id="result">checking</pre>
<script>
const result={kind:'qa-bridge-echo-result',http:null,websocket:null};
function publish(){document.querySelector('#result').textContent=JSON.stringify(result);parent.postMessage(result,'*');}
(async()=>{try{const r=await fetch('./http');result.http={status:r.status,body:await r.json(),date:r.headers.get('date'),server:r.headers.get('server')};}catch(e){result.http={error:e.name};}publish();})();
const u=new URL('./ws',location.href);u.protocol=location.protocol==='https:'?'wss:':'ws:';
const socket=new WebSocket(u,'qa.echo.v1');
socket.onopen=()=>socket.send('QA_ECHO_7731');
socket.onmessage=e=>{try{result.websocket={protocol:socket.protocol,reply:JSON.parse(e.data)};}catch{result.websocket={error:'invalid reply'};}publish();socket.close();};
socket.onerror=()=>{result.websocket={error:'connection error'};publish();};
setTimeout(()=>{if(!result.websocket){result.websocket={error:'5s deadline'};publish();socket.close();}},5000);
</script>'''

class Handler(BaseHTTPRequestHandler):
    server_version = 'QAOwnedBridge/1'
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def response(self, body, content_type='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/health':
            self.response(b'{"healthy":true}')
        elif path == '/http':
            # Presence booleans only: never return headers, cookie or tokens.
            self.response(json.dumps({'ok': True, 'fixture': 'QA_HTTP_7731',
                'control_key_forwarded': bool(self.headers.get('X-Session-API-Key')),
                'cookie_forwarded': bool(self.headers.get('Cookie'))}).encode())
        elif path == '/ws' and self.headers.get('Upgrade', '').lower() == 'websocket':
            self.websocket()
        elif path == '/':
            self.response(PAGE, 'text/html; charset=utf-8')
        else:
            self.send_error(404)

    def read_exact(self, size):
        out = b''
        while len(out) < size:
            chunk = self.rfile.read(size-len(out))
            if not chunk:
                raise EOFError
            out += chunk
        return out

    def websocket(self):
        key = self.headers.get('Sec-WebSocket-Key', '')
        digest = hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()
        offered = [v.strip() for v in self.headers.get('Sec-WebSocket-Protocol', '').split(',')]
        selected = 'qa.echo.v1' if 'qa.echo.v1' in offered else ''
        self.send_response(101)
        self.send_header('Upgrade', 'websocket')
        self.send_header('Connection', 'Upgrade')
        self.send_header('Sec-WebSocket-Accept', base64.b64encode(digest).decode())
        if selected:
            self.send_header('Sec-WebSocket-Protocol', selected)
        self.end_headers()
        self.connection.settimeout(6)
        try:
            first, second = self.read_exact(2)
            length = second & 127
            if length == 126:
                length = struct.unpack('!H', self.read_exact(2))[0]
            elif length == 127:
                length = struct.unpack('!Q', self.read_exact(8))[0]
            if first & 15 != 1 or length > 256:
                return
            mask = self.read_exact(4) if second & 128 else b''
            payload = self.read_exact(length)
            if mask:
                payload = bytes(v ^ mask[n % 4] for n,v in enumerate(payload))
            # Echo a fixed known dummy only; arbitrary input never reaches output.
            body = json.dumps({'echo': 'QA_ECHO_7731' if payload == b'QA_ECHO_7731' else 'unexpected',
                'backend_selected_protocol': selected,
                'backend_saw_requested_protocol': bool(selected)}).encode()
            head = bytes([0x81,len(body)]) if len(body)<126 else bytes([0x81,126])+struct.pack('!H',len(body))
            self.wfile.write(head+body)
            self.wfile.flush()
        except (EOFError, OSError):
            pass
        finally:
            self.close_connection = True

if __name__ == '__main__':
    # Port allocated by the real backend manager; bind loopback only.
    port = int(sys.argv[1])
    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(('127.0.0.1',port), Handler).serve_forever()

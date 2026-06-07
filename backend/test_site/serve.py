#!/usr/bin/env python3
import http.server, socketserver, os
os.chdir(os.path.dirname(__file__))
PORT = 8080
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Test site: http://localhost:{PORT}")
    httpd.serve_forever()

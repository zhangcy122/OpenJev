#!/usr/bin/env python3
"""
Simple local development server for OpenJev landing page.
Usage:
    python scripts/serve_site.py [--port 8080]
"""

import argparse
import http.server
import os
import socketserver
import sys

SITE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "site"))


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SITE_DIR, **kwargs)

    def end_headers(self):
        # Disable aggressive caching during development
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve OpenJev site locally")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    if not os.path.isdir(SITE_DIR):
        print(f"Error: Site directory not found at {SITE_DIR}", file=sys.stderr)
        sys.exit(1)

    with socketserver.TCPServer((args.host, args.port), CustomHandler) as httpd:
        print(f"Serving OpenJev site from: {SITE_DIR}")
        print(f"URL: http://localhost:{args.port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()

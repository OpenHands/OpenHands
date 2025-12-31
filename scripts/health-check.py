#!/usr/bin/env python3
"""
Health Check Script for OpenHands
בודק את מצב כל השירותים במערכת
"""

import sys
import requests
import json
from typing import Dict, Tuple
from datetime import datetime

# הגדרת שירותים לבדיקה
SERVICES = {
    "openhands": {
        "url": "http://localhost:3002/api/health",
        "timeout": 5,
        "expected_status": 200
    },
    "sglang": {
        "url": "http://localhost:30000/health",
        "timeout": 5,
        "expected_status": 200
    },
    "invariant-server": {
        "url": "http://localhost:8000/",
        "timeout": 5,
        "expected_status": 200
    },
    "code-server": {
        "url": "http://localhost:8081/healthz",
        "timeout": 5,
        "expected_status": 200
    },
    "prometheus": {
        "url": "http://localhost:9090/-/healthy",
        "timeout": 5,
        "expected_status": 200
    },
    "grafana": {
        "url": "http://localhost:3001/api/health",
        "timeout": 5,
        "expected_status": 200
    }
}


def check_service(name: str, config: Dict) -> Tuple[bool, str]:
    """בודק שירות בודד"""
    try:
        response = requests.get(
            config["url"],
            timeout=config["timeout"],
            allow_redirects=True
        )
        
        if response.status_code == config["expected_status"]:
            return True, f"Status: {response.status_code}"
        else:
            return False, f"Status: {response.status_code} (expected {config['expected_status']})"
            
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """פונקציה ראשית"""
    print("=" * 50)
    print("OpenHands Health Check")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print()
    
    all_healthy = True
    results = {}
    
    for name, config in SERVICES.items():
        is_healthy, message = check_service(name, config)
        results[name] = {
            "healthy": is_healthy,
            "message": message
        }
        
        status_icon = "✅" if is_healthy else "❌"
        print(f"{status_icon} {name:20s} - {message}")
        
        if not is_healthy:
            all_healthy = False
    
    print()
    print("=" * 50)
    
    # בדיקת Docker containers
    print("\nDocker Containers Status:")
    print("-" * 50)
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("⚠️  לא ניתן לבדוק Docker containers")
    except Exception as e:
        print(f"⚠️  שגיאה בבדיקת Docker: {e}")
    
    # סיכום
    print()
    print("=" * 50)
    if all_healthy:
        print("✅ כל השירותים פעילים!")
        return 0
    else:
        print("❌ יש שירותים לא פעילים!")
        return 1


if __name__ == "__main__":
    sys.exit(main())


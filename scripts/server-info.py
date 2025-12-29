#!/usr/bin/env python3
"""
סקריפט להצגת מידע מלא על השרת
נוצר: 2025-12-27
"""
import os
import json
import subprocess
import socket
from pathlib import Path

def get_docker_containers():
    """קבלת רשימת קונטיינרים"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', 'json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                containers.append(json.loads(line))
        return containers
    except Exception as e:
        return {"error": str(e)}

def get_systemd_services():
    """קבלת רשימת שירותי systemd"""
    try:
        result = subprocess.run(
            ['systemctl', 'list-units', '--type=service', '--state=running', '--no-pager'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except Exception as e:
        return f"לא ניתן לגשת ל-systemd: {e}"

def get_network_info():
    """קבלת מידע על הרשת"""
    info = {}
    try:
        # IP addresses
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        info['ip_addresses'] = result.stdout.strip().split()
        
        # Network interfaces
        result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
        info['interfaces'] = result.stdout
    except Exception as e:
        info['error'] = str(e)
    return info

def get_processes():
    """קבלת רשימת תהליכים"""
    processes = []
    proc_path = Path('/host/proc') if Path('/host/proc').exists() else Path('/proc')
    
    try:
        for pid_dir in list(proc_path.glob('[0-9]*'))[:50]:  # ראשונים 50
            try:
                pid = pid_dir.name
                comm = (pid_dir / 'comm').read_text().strip()
                cmdline = (pid_dir / 'cmdline').read_text().replace('\0', ' ')
                processes.append({
                    'pid': pid,
                    'name': comm,
                    'cmdline': cmdline[:200]  # מוגבל ל-200 תווים
                })
            except:
                continue
    except Exception as e:
        return {"error": str(e)}
    
    return processes

def get_ports():
    """קבלת פורטים פעילים"""
    try:
        result = subprocess.run(
            ['ss', '-tulnp'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except:
        try:
            result = subprocess.run(
                ['netstat', '-tulnp'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception as e:
            return f"לא ניתן לקבל מידע על פורטים: {e}"

def get_disk_usage():
    """קבלת שימוש בדיסק"""
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        return f"לא ניתן לקבל מידע על דיסק: {e}"

def get_memory():
    """קבלת מידע על זיכרון"""
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True, timeout=10)
        return result.stdout
    except Exception as e:
        return f"לא ניתן לקבל מידע על זיכרון: {e}"

def main():
    """פונקציה ראשית"""
    info = {
        'hostname': socket.gethostname(),
        'docker_containers': get_docker_containers(),
        'systemd_services': get_systemd_services(),
        'network': get_network_info(),
        'processes': get_processes(),
        'ports': get_ports(),
        'disk_usage': get_disk_usage(),
        'memory': get_memory(),
    }
    
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()







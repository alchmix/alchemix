import json
import re
import urllib.request

url = "https://grok.com/imagine/post/87b7c010-0099-4ba9-bab2-ec134f7a5e9e"
req = urllib.request.Request(
    url,
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"},
)
html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
print("html_len", len(html))

for pat in [
    r"https://[^\s\"'<>]+\.mp4[^\s\"'<>]*",
    r"https://assets\.grok\.com[^\s\"'<>]+",
    r"https://imagine-public\.x\.ai[^\s\"'<>]+",
]:
    found = re.findall(pat, html)
    print(pat, len(found))
    for x in found[:8]:
        print(" ", x[:150])

m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
if m:
    data = json.loads(m.group(1))
    blob = json.dumps(data)
    print("NEXT_DATA size", len(blob))
    for kw in ("mediaUrl", "hdMediaUrl", "videoUrl", "87b7c010"):
        print(kw, blob.count(kw))

# try public API without cookies
post_id = "87b7c010-0099-4ba9-bab2-ec134f7a5e9e"
for path, body in [
    ("/rest/media/post/get", {"id": post_id}),
    ("/rest/media/post/get", {"postId": post_id}),
]:
    api = "https://grok.com" + path
    r = urllib.request.Request(
        api,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Referer": url,
            "Origin": "https://grok.com",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(r, timeout=20)
        print("API ok", path, body.keys(), resp.status, resp.read()[:500])
    except Exception as e:
        print("API fail", path, type(e).__name__, e)

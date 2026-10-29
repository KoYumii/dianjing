const CACHE = 'dianjing-v3';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icon-180.png', './icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

/* 浠ｇ爜绫昏蛋 network-first锛堝紑鍙戞椂鎬昏兘鎷垮埌鏂扮増锛屾柇缃戝洖钀藉埌缂撳瓨锛?   璧勬簮绫昏蛋 cache-first锛堝浘鐗?LUT/妯″瀷锛岀渷娴侀噺锛?*/
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const u = new URL(req.url);
  if (u.origin !== location.origin) return;            // CDN 璧扮綉缁?
  const isCode = /\.(?:html|js|css|webmanifest)$|\/$/.test(u.pathname);

  if (isCode) {
    e.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
          return res;
        })
        .catch(() => caches.match(req).then((r) => r || caches.match('./index.html')))
    );
    return;
  }

  e.respondWith(
    caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(req, copy));
      return res;
    }))
  );
});

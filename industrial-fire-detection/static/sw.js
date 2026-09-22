// Service Worker for Industrial Fire Detection PWA
const CACHE_NAME = 'fire-alert-pwa-v1';
const STATIC_ASSETS = [
    '/',
    '/dashboard',
    '/alerts',
    '/offline',
    '/static/manifest.json',
    '/static/firefighter-logo.png',
    '/static/icon-192.png',
    '/static/icon-512.png',
    'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
    'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
    'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'
];

// Install: Cache critical assets
self.addEventListener('install', (event) => {
    console.log('[Service Worker] Installing version:', CACHE_NAME);
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[Service Worker] Caching static assets');
            // Cache static assets gracefully (catch individually in case of network issues with external CDNs)
            return Promise.allSettled(
                STATIC_ASSETS.map((url) =>
                    cache.add(url).catch((err) => console.warn('[Service Worker] Failed to cache:', url, err))
                )
            );
        }).then(() => self.skipWaiting())
    );
});

// Activate: Clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[Service Worker] Activating new Service Worker');
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.map((key) => {
                    if (key !== CACHE_NAME) {
                        console.log('[Service Worker] Removing old cache:', key);
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch: Network-first for APIs and live data, cache fallback for offline
self.addEventListener('fetch', (event) => {
    const requestUrl = new URL(event.request.url);

    // Only handle GET requests
    if (event.request.method !== 'GET') {
        return;
    }

    // Skip API calls from caching, but return offline JSON error if network fails
    if (requestUrl.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(
                    JSON.stringify({ error: 'Offline', message: 'You are currently offline. Live data unavailable.' }),
                    { headers: { 'Content-Type': 'application/json' }, status: 503 }
                );
            })
        );
        return;
    }

    // Network-first with cache fallback for pages and assets
    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // If valid response, clone and update cache
                if (response && response.status === 200) {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                    });
                }
                return response;
            })
            .catch(async () => {
                // Network failed, check cache
                const cachedResponse = await caches.match(event.request);
                if (cachedResponse) {
                    return cachedResponse;
                }
                // If requesting an HTML navigation page, show offline fallback
                if (event.request.headers.get('accept') && event.request.headers.get('accept').includes('text/html')) {
                    const offlinePage = await caches.match('/offline');
                    if (offlinePage) {
                        return offlinePage;
                    }
                }
                return new Response('Network error occurred and no offline copy is available.', {
                    status: 503,
                    statusText: 'Service Unavailable',
                    headers: new Headers({ 'Content-Type': 'text/plain' })
                });
            })
    );
});

// Push notification event listener
self.addEventListener('push', (event) => {
    console.log('[Service Worker] Push notification received');
    let data = {};
    if (event.data) {
        try {
            data = event.data.json();
        } catch (e) {
            data = { body: event.data.text() };
        }
    }

    const title = data.title || '🚨 Fire Anomaly Alert';
    const options = {
        body: data.body || 'A new industrial fire anomaly has been detected!',
        icon: data.icon || '/static/icon-192.png',
        badge: data.badge || '/static/icon-192.png',
        tag: data.tag || 'fire-anomaly',
        data: {
            url: data.url || '/dashboard',
            timestamp: Date.now()
        },
        vibrate: [300, 100, 300, 100, 300],
        requireInteraction: true,
        actions: [
            { action: 'open_map', title: '🗺️ Open Map' },
            { action: 'dismiss', title: 'Dismiss' }
        ]
    };

    event.waitUntil(self.registration.showNotification(title, options));
});

// Notification click event handler
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.action === 'dismiss') {
        return;
    }

    const targetUrl = (event.notification.data && event.notification.data.url) ? event.notification.data.url : '/dashboard';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
            // If already open, focus on that tab
            for (let client of clientList) {
                if (client.url.includes(targetUrl) && 'focus' in client) {
                    return client.focus();
                }
            }
            // Otherwise open a new window
            if (clients.openWindow) {
                return clients.openWindow(targetUrl);
            }
        })
    );
});

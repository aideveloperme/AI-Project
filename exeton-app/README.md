# Exeton GPU Store (iPhone app)

A mobile shopping app for GPUs, AI servers and networking gear, built as a
Progressive Web App (PWA). On iPhone it installs to the Home Screen and runs
full-screen like a native app, with offline support. No build step, no App
Store account needed to test.

## Features
- Catalog with categories: Data Center GPUs, Workstation GPUs, GPU Servers, Networking
- Product pages with full specs, stock status and quantity picker
- Search across names, specs and descriptions
- Cart with quantity controls; "Request quote" items for servers without list prices
- Checkout form (contact, shipping, payment method, notes) and order confirmation
- Order history on the Account tab
- iPhone-native feel: bottom tab bar, safe-area (notch / home indicator) support, dark mode, Home Screen icon

## Test it on your iPhone

### Option A: GitHub Pages (works from anywhere)
1. On GitHub open **Settings → Pages** and set **Source** to **GitHub Actions**.
2. Under **Settings → Environments → github-pages**, allow the branch
   `claude/epic-darwin-wq6ldk` (or merge it into `main`).
3. Re-run the **Deploy Exeton app to GitHub Pages** workflow from the **Actions** tab.
4. Open `https://<your-username>.github.io/<repo-name>/` in **Safari** on your iPhone.

### Option B: your computer on the same Wi-Fi
```bash
cd exeton-app
python3 -m http.server 8000
```
Open `http://<your-computer-ip>:8000` in Safari on the iPhone.
(Offline mode needs HTTPS, so it only works with Option A.)

### Install it
In Safari tap **Share → Add to Home Screen**. Launch "Exeton" from the Home Screen.

## Make it yours
- **Products / prices:** edit `products.js` (the current catalog is sample data; set `price: null` for "Request quote").
- **Tax:** set `TAX_RATE` at the top of `app.js`.
- **Colors:** edit the variables at the top of `styles.css`.
- **After changing files:** bump `VERSION` in `sw.js` so installed phones pick up the update.

## Going live
Orders are currently saved only on the device (demo mode; no payment is taken).
To take real orders, send the order object built in `checkoutView()` (`app.js`)
to your backend, e.g. a Shopify/WooCommerce API, a Stripe Checkout session, or an
email/CRM webhook. To publish on the App Store later, wrap this folder with
Capacitor (`npx cap add ios`) and build in Xcode.

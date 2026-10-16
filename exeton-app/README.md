# Exeton GPU Store (iPhone app)

A mobile shopping app for GPUs, AI servers and networking gear, built as a
Progressive Web App (PWA). On iPhone it installs to the Home Screen and runs
full-screen like a native app, with offline support. No build step, no App
Store account needed to test.

## Features
- Full Exeton catalog: 328 products from `data/exeton-products.csv`, sorted into GPUs, AI Systems,
  Servers, Workstations & PCs, Storage, Components, Networking, Power & Racks and Accessories
- Featured rail, brand filter, "Show more" paging, related products
- Product pages with description, specs (brand, memory, architecture, form factor, part number)
  and a link to the full page on exeton.com
- Search across names, brands, specs and descriptions (multi-word, e.g. "supermicro b200")
- Quote list with quantities, then a quote request form (contact, delivery, preferred payment, notes)
- Quote history on the Account tab
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

## Updating the catalog
`products.js` is generated from the CSV. After editing or replacing `data/exeton-products.csv`:
```bash
cd exeton-app
python3 tools/csv_to_products.py data/exeton-products.csv > products.js
```
Required columns: `SL. No`, `URL`, `Descritpion`, `Title`. Optional columns:
- `Price` (e.g. `27900`): shows the price instead of "Price on request" and adds it to the estimated total
- `Category` (one of `gpus`, `ai-systems`, `servers`, `workstations`, `storage`, `components`,
  `networking`, `power-racks`, `accessories`): overrides the automatic category
Rows 1 to 16 are shown in the Featured rail.

## Make it yours
- **Sales inbox:** set `SALES_EMAIL` at the top of `app.js` to show an "Email this request to Exeton" button.
- **Colors:** edit the variables at the top of `styles.css`.
- **After changing files:** bump `VERSION` in `sw.js` so installed phones pick up the update.

## Going live
Quote requests are saved on the device (no payment is taken).
To receive them automatically, send the order object built in `checkoutView()` (`app.js`)
to your backend, e.g. a Shopify/WooCommerce API, a Stripe Checkout session, or an
email/CRM webhook. To publish on the App Store later, wrap this folder with
Capacitor (`npx cap add ios`) and build in Xcode.

(function () {
  "use strict";

  const { categories, products } = window.CATALOG;
  const byId = Object.fromEntries(products.map((p) => [p.id, p]));
  const view = document.getElementById("view");
  const titleEl = document.getElementById("title");
  const backBtn = document.getElementById("backBtn");
  const badge = document.getElementById("cartBadge");
  const toastEl = document.getElementById("toast");

  const TAX_RATE = 0.0; // set your sales tax / VAT rate here
  const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

  // ---------- storage ----------
  function load(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  }
  function save(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
  }
  let cart = load("exeton.cart", {});     // { productId: qty }
  let orders = load("exeton.orders", []); // [{ id, date, items, total, customer }]
  let activeCategory = "all";

  // ---------- helpers ----------
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const priceLabel = (p) => (p.price == null ? "Request quote" : money.format(p.price));
  const catName = (id) => (categories.find((c) => c.id === id) || {}).name || "";

  function art(p) {
    const h = p.hue;
    const isServer = p.category === "servers";
    const isNet = p.category === "networking";
    let shape;
    if (isServer) {
      shape = `<rect x="20" y="30" width="160" height="90" rx="6" fill="hsl(${h} 15% 22%)"/>
        ${[0, 1, 2, 3].map((i) => `<rect x="30" y="${40 + i * 20}" width="140" height="12" rx="2" fill="hsl(${h} 20% 32%)"/><circle cx="160" cy="${46 + i * 20}" r="2.5" fill="hsl(${h} 90% 60%)"/>`).join("")}`;
    } else if (isNet) {
      shape = `<rect x="20" y="55" width="160" height="40" rx="5" fill="hsl(${h} 15% 22%)"/>
        ${Array.from({ length: 10 }, (_, i) => `<rect x="${30 + i * 14}" y="65" width="10" height="20" rx="1.5" fill="hsl(${h} 30% 40%)"/>`).join("")}
        <circle cx="172" cy="62" r="2.5" fill="hsl(${h} 90% 65%)"/>`;
    } else {
      shape = `<rect x="18" y="40" width="164" height="70" rx="8" fill="hsl(${h} 12% 18%)"/>
        <rect x="18" y="40" width="164" height="8" rx="3" fill="hsl(${h} 70% 50%)"/>
        <circle cx="70" cy="77" r="22" fill="hsl(${h} 10% 26%)" stroke="hsl(${h} 15% 40%)" stroke-width="2"/>
        <circle cx="130" cy="77" r="22" fill="hsl(${h} 10% 26%)" stroke="hsl(${h} 15% 40%)" stroke-width="2"/>
        <circle cx="70" cy="77" r="5" fill="hsl(${h} 15% 45%)"/><circle cx="130" cy="77" r="5" fill="hsl(${h} 15% 45%)"/>
        <rect x="40" y="110" width="90" height="8" rx="1" fill="hsl(45 70% 55%)"/>`;
    }
    return `<div class="art" style="background:linear-gradient(160deg,hsl(${h} 35% 30%),hsl(${h} 30% 10%))">
      ${p.tag ? `<span class="tag">${esc(p.tag)}</span>` : ""}
      <svg viewBox="0 0 200 150" aria-hidden="true">${shape}</svg></div>`;
  }

  function card(p) {
    return `<a class="card" href="#/p/${p.id}">
      ${art(p)}
      <div class="body">
        <div class="name">${esc(p.name)}</div>
        <div class="meta">${esc(p.specs.Memory || p.specs["GPU Memory"] || p.specs.Speed || p.specs.Ports || "")}</div>
        <div class="price">${priceLabel(p)}</div>
      </div></a>`;
  }

  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toast.t);
    toast.t = setTimeout(() => toastEl.classList.remove("show"), 1800);
  }

  function cartCount() { return Object.values(cart).reduce((a, b) => a + b, 0); }
  function updateBadge() {
    const n = cartCount();
    badge.hidden = n === 0;
    badge.textContent = n > 99 ? "99+" : n;
  }
  function setCart(id, qty) {
    if (qty <= 0) delete cart[id]; else cart[id] = Math.min(qty, 99);
    save("exeton.cart", cart);
    updateBadge();
  }
  function cartLines() {
    return Object.entries(cart).filter(([id]) => byId[id]).map(([id, qty]) => ({ p: byId[id], qty }));
  }
  function totals(lines) {
    const priced = lines.filter((l) => l.p.price != null);
    const subtotal = priced.reduce((s, l) => s + l.p.price * l.qty, 0);
    const tax = Math.round(subtotal * TAX_RATE);
    return { subtotal, tax, total: subtotal + tax, quoteItems: lines.length - priced.length };
  }

  function setChrome(title, { back = false, tab = null } = {}) {
    titleEl.textContent = title;
    document.title = title === "Exeton" ? "Exeton GPU Store" : `${title} · Exeton`;
    backBtn.hidden = !back;
    document.querySelectorAll(".tabbar a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
  }

  // ---------- views ----------
  function shopView() {
    setChrome("Exeton", { tab: "shop" });
    const list = activeCategory === "all" ? products : products.filter((p) => p.category === activeCategory);
    view.innerHTML = `
      <section class="hero">
        <small>AI &amp; HPC hardware</small>
        <h2>GPUs and AI servers, ready to ship</h2>
        <p>Data center accelerators, workstation cards and turnkey HGX systems.</p>
        <a class="btn" href="#/" data-cat="servers">Explore AI servers</a>
      </section>
      <div class="chips" role="tablist">
        ${categories.map((c) => `<button class="chip ${c.id === activeCategory ? "active" : ""}" data-cat="${c.id}" role="tab" aria-selected="${c.id === activeCategory}">${esc(c.name)}</button>`).join("")}
      </div>
      <div class="section-title">${esc(catName(activeCategory))} · ${list.length} products</div>
      <div class="grid">${list.map(card).join("")}</div>`;
    view.querySelectorAll("[data-cat]").forEach((el) =>
      el.addEventListener("click", (e) => {
        e.preventDefault();
        activeCategory = el.dataset.cat;
        shopView();
        if (el.classList.contains("btn")) view.querySelector(".chips").scrollIntoView({ behavior: "smooth" });
      })
    );
  }

  function productView(id) {
    const p = byId[id];
    if (!p) return notFound();
    setChrome(catName(p.category), { back: true, tab: "shop" });
    let qty = 1;
    const low = p.stock <= 5;
    view.innerHTML = `
      <div class="detail">
        <div class="detail-art">${art(p)}</div>
        <h2>${esc(p.name)}</h2>
        <div class="price-lg">${priceLabel(p)}</div>
        <div class="stock ${low ? "low" : ""}">${low ? `Only ${p.stock} left in stock` : "In stock · ships in 2–5 days"}</div>
        <p>${esc(p.summary)}</p>
        <dl class="specs">${Object.entries(p.specs).map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
        <div class="buybar">
          <div class="qty"><button id="minus" aria-label="Decrease quantity">−</button><span id="q">1</span><button id="plus" aria-label="Increase quantity">+</button></div>
          <button class="btn grow" id="add">${p.price == null ? "Add to quote" : "Add to cart"}</button>
        </div>
      </div>`;
    const q = view.querySelector("#q");
    view.querySelector("#minus").onclick = () => { qty = Math.max(1, qty - 1); q.textContent = qty; };
    view.querySelector("#plus").onclick = () => { qty = Math.min(p.stock, qty + 1); q.textContent = qty; };
    view.querySelector("#add").onclick = () => {
      setCart(p.id, (cart[p.id] || 0) + qty);
      if (navigator.vibrate) navigator.vibrate(15);
      toast(`Added ${qty} × ${p.name}`);
    };
  }

  function searchView() {
    setChrome("Search", { tab: "search" });
    view.innerHTML = `
      <div class="search">
        <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/></svg>
        <input id="q" type="search" placeholder="H100, 48 GB, Blackwell…" autocomplete="off" enterkeyhint="search" aria-label="Search products">
      </div>
      <div id="results"></div>`;
    const input = view.querySelector("#q");
    const results = view.querySelector("#results");
    const run = () => {
      const term = input.value.trim().toLowerCase();
      if (!term) { results.innerHTML = `<div class="section-title">All products</div><div class="grid">${products.map(card).join("")}</div>`; return; }
      const hits = products.filter((p) =>
        [p.name, p.summary, catName(p.category), ...Object.values(p.specs)].join(" ").toLowerCase().includes(term)
      );
      results.innerHTML = hits.length
        ? `<div class="section-title">${hits.length} result${hits.length > 1 ? "s" : ""}</div><div class="grid">${hits.map(card).join("")}</div>`
        : `<div class="empty"><h3>No matches</h3>Try a model name like “L40S” or a spec like “HBM3e”.</div>`;
    };
    input.addEventListener("input", run);
    run();
  }

  function cartView() {
    setChrome("Cart", { tab: "cart" });
    const lines = cartLines();
    if (!lines.length) {
      view.innerHTML = `<div class="empty">
        <svg viewBox="0 0 24 24"><circle cx="9" cy="20" r="1.4"/><circle cx="17" cy="20" r="1.4"/><path d="M3 4h2l2.4 11h11L21 8H6.2"/></svg>
        <h3>Your cart is empty</h3><p>Browse GPUs and servers to get started.</p>
        <a class="btn" href="#/">Start shopping</a></div>`;
      return;
    }
    const t = totals(lines);
    view.innerHTML = `
      <div class="list">${lines.map(({ p, qty }) => `
        <div class="row">
          <a class="thumb" href="#/p/${p.id}">${art({ ...p, tag: null })}</a>
          <div class="info">
            <div class="name">${esc(p.name)}</div>
            <div class="meta">${p.price == null ? "Quote item" : money.format(p.price) + " each"}</div>
            <button class="remove" data-remove="${p.id}">Remove</button>
          </div>
          <div class="qty"><button data-dec="${p.id}" aria-label="Decrease">−</button><span>${qty}</span><button data-inc="${p.id}" aria-label="Increase">+</button></div>
        </div>`).join("")}
      </div>
      <div class="totals">
        <div><span>Subtotal</span><span>${money.format(t.subtotal)}</span></div>
        ${TAX_RATE ? `<div><span>Tax</span><span>${money.format(t.tax)}</span></div>` : ""}
        <div><span>Shipping</span><span>Free</span></div>
        ${t.quoteItems ? `<div><span>Items priced on quote</span><span>${t.quoteItems}</span></div>` : ""}
        <div class="grand"><span>Total</span><span>${money.format(t.total)}${t.quoteItems ? " +" : ""}</span></div>
      </div>
      <a class="btn block" href="#/checkout">Checkout</a>`;
    view.querySelectorAll("[data-remove]").forEach((b) => (b.onclick = () => { setCart(b.dataset.remove, 0); cartView(); }));
    view.querySelectorAll("[data-dec]").forEach((b) => (b.onclick = () => { setCart(b.dataset.dec, cart[b.dataset.dec] - 1); cartView(); }));
    view.querySelectorAll("[data-inc]").forEach((b) => (b.onclick = () => {
      const id = b.dataset.inc;
      setCart(id, Math.min(byId[id].stock, cart[id] + 1));
      cartView();
    }));
  }

  function checkoutView() {
    const lines = cartLines();
    if (!lines.length) { location.hash = "#/cart"; return; }
    setChrome("Checkout", { back: true, tab: "cart" });
    const t = totals(lines);
    const saved = load("exeton.customer", {});
    view.innerHTML = `
      <form class="checkout" novalidate>
        <div class="section-title">Contact</div>
        <label>Full name<input name="name" required autocomplete="name" value="${esc(saved.name || "")}"></label>
        <label>Company<input name="company" autocomplete="organization" value="${esc(saved.company || "")}"></label>
        <div class="two">
          <label>Email<input name="email" type="email" required autocomplete="email" inputmode="email" value="${esc(saved.email || "")}"></label>
          <label>Phone<input name="phone" type="tel" autocomplete="tel" inputmode="tel" value="${esc(saved.phone || "")}"></label>
        </div>
        <div class="section-title">Shipping</div>
        <label>Address<input name="address" required autocomplete="street-address" value="${esc(saved.address || "")}"></label>
        <div class="two">
          <label>City<input name="city" required autocomplete="address-level2" value="${esc(saved.city || "")}"></label>
          <label>Country<input name="country" required autocomplete="country-name" value="${esc(saved.country || "")}"></label>
        </div>
        <label>Payment
          <select name="payment">
            <option>Bank transfer / invoice</option>
            <option>Purchase order</option>
            <option>Credit card (pay on confirmation)</option>
          </select>
        </label>
        <label>Notes<textarea name="notes" placeholder="Delivery window, configuration requests…"></textarea></label>
        <div class="totals">
          <div class="grand"><span>Total</span><span>${money.format(t.total)}${t.quoteItems ? " +" : ""}</span></div>
        </div>
        <p class="note">This is a demo app: no payment is taken. Placing an order saves it on this device so you can test the flow.</p>
        <button class="btn block" type="submit">Place order</button>
      </form>`;
    const form = view.querySelector("form");
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(form));
      const missing = ["name", "email", "address", "city", "country"].filter((k) => !String(data[k] || "").trim());
      if (missing.length || !/^\S+@\S+\.\S+$/.test(data.email)) {
        toast("Please fill in all required fields");
        const first = form.elements[missing[0] || "email"];
        first && first.focus();
        return;
      }
      const { notes, payment, ...customer } = data;
      save("exeton.customer", customer);
      const order = {
        id: "EX-" + Date.now().toString(36).toUpperCase(),
        date: new Date().toISOString(),
        items: lines.map(({ p, qty }) => ({ id: p.id, name: p.name, price: p.price, qty })),
        total: t.total, quoteItems: t.quoteItems, payment, notes, customer
      };
      orders.unshift(order);
      save("exeton.orders", orders);
      cart = {}; save("exeton.cart", cart); updateBadge();
      location.hash = `#/order/${order.id}`;
    });
  }

  function orderView(id) {
    const o = orders.find((x) => x.id === id);
    if (!o) return notFound();
    setChrome("Order", { back: true, tab: "account" });
    view.innerHTML = `
      <div class="empty" style="padding:30px 10px">
        <svg viewBox="0 0 24 24" style="color:var(--accent)"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/></svg>
        <h3>Order received</h3>
        <p>Order <b>${esc(o.id)}</b> · ${new Date(o.date).toLocaleString()}</p>
        <p>We'll email ${esc(o.customer.email)} to confirm${o.quoteItems ? " pricing and" : ""} delivery.</p>
      </div>
      <div class="list">${o.items.map((i) => `
        <div class="row"><div class="info"><div class="name">${esc(i.name)}</div><div class="meta">Qty ${i.qty} · ${i.price == null ? "Quote" : money.format(i.price * i.qty)}</div></div></div>`).join("")}
      </div>
      <div class="totals"><div class="grand"><span>Total</span><span>${money.format(o.total)}${o.quoteItems ? " +" : ""}</span></div></div>
      <a class="btn block ghost" href="#/">Continue shopping</a>`;
  }

  function accountView() {
    setChrome("Account", { tab: "account" });
    const c = load("exeton.customer", {});
    const standalone = window.navigator.standalone || matchMedia("(display-mode: standalone)").matches;
    view.innerHTML = `
      <div class="section-title">Profile</div>
      <div class="list"><div class="row"><div class="info">
        <div class="name">${esc(c.name || "Guest")}</div>
        <div class="meta">${esc(c.email || "Your details are saved at checkout")}</div>
      </div></div></div>
      <div class="section-title" style="margin-top:20px">Orders</div>
      ${orders.length ? `<div class="list">${orders.map((o) => `
        <a class="row" href="#/order/${o.id}"><div class="info">
          <div class="name">${esc(o.id)}</div>
          <div class="meta">${new Date(o.date).toLocaleDateString()} · ${o.items.reduce((s, i) => s + i.qty, 0)} items · ${money.format(o.total)}${o.quoteItems ? " +" : ""}</div>
        </div><svg viewBox="0 0 24 24" style="color:var(--muted);width:18px"><path d="M9 5l7 7-7 7"/></svg></a>`).join("")}</div>`
        : `<div class="list"><div class="row"><div class="info"><div class="meta">No orders yet.</div></div></div></div>`}
      ${standalone ? "" : `<div class="install-tip"><b>Install on iPhone:</b> tap the Share button in Safari, then <b>Add to Home Screen</b>. Exeton opens full-screen like a native app.</div>`}
      ${orders.length ? `<button class="btn block ghost" id="clear" style="margin-top:18px">Clear order history</button>` : ""}`;
    const clear = view.querySelector("#clear");
    if (clear) clear.onclick = () => {
      if (!confirm("Delete all saved orders on this device?")) return;
      orders = []; save("exeton.orders", orders); accountView();
    };
  }

  function notFound() {
    setChrome("Not found", { back: true });
    view.innerHTML = `<div class="empty"><h3>Page not found</h3><a class="btn" href="#/">Go to shop</a></div>`;
  }

  // ---------- router ----------
  function route() {
    const parts = location.hash.replace(/^#\/?/, "").split("/");
    const [page, arg] = parts;
    window.scrollTo(0, 0);
    toastEl.classList.remove("show");
    switch (page) {
      case "": shopView(); break;
      case "p": productView(arg); break;
      case "search": searchView(); break;
      case "cart": cartView(); break;
      case "checkout": checkoutView(); break;
      case "order": orderView(arg); break;
      case "account": accountView(); break;
      default: notFound();
    }
  }

  backBtn.addEventListener("click", () => (history.length > 1 ? history.back() : (location.hash = "#/")));
  window.addEventListener("hashchange", route);
  updateBadge();
  route();

  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
})();

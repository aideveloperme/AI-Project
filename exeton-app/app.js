(function () {
  "use strict";

  const { categories, products } = window.CATALOG;
  const byId = Object.fromEntries(products.map((p) => [p.id, p]));
  const view = document.getElementById("view");
  const titleEl = document.getElementById("title");
  const backBtn = document.getElementById("backBtn");
  const badge = document.getElementById("cartBadge");
  const toastEl = document.getElementById("toast");

  // Where quote requests are emailed from the confirmation screen. Set to your sales inbox.
  const SALES_EMAIL = "";
  const PAGE_SIZE = 24;
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
  const shop = { category: "all", brand: "", shown: PAGE_SIZE };

  // ---------- helpers ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const priceLabel = (p) => (p.price == null ? "Price on request" : money.format(p.price));
  const catName = (id) => (categories.find((c) => c.id === id) || {}).name || "";
  const inCategory = (id) => (id === "all" ? products : products.filter((p) => p.category === id));
  const brandsOf = (list) => [...new Set(list.map((p) => p.brand).filter(Boolean))].sort((a, b) => a.localeCompare(b));

  const CAT_HUE = { gpus: 110, "ai-systems": 150, servers: 200, workstations: 180, storage: 30, components: 260, networking: 280, "power-racks": 0, accessories: 220 };
  function hueOf(p) {
    let h = 0;
    for (const ch of p.id) h = (h * 31 + ch.charCodeAt(0)) % 997;
    return (CAT_HUE[p.category] ?? 120) + (h % 24) - 12;
  }

  function art(p, { tag = true } = {}) {
    const h = hueOf(p);
    const dark = `hsl(${h} 12% 18%)`, mid = `hsl(${h} 18% 32%)`, led = `hsl(${h} 85% 60%)`;
    let shape;
    switch (p.category) {
      case "gpus":
        shape = `<rect x="18" y="40" width="164" height="70" rx="8" fill="${dark}"/>
          <rect x="18" y="40" width="164" height="8" rx="3" fill="hsl(${h} 70% 50%)"/>
          <circle cx="70" cy="77" r="22" fill="hsl(${h} 10% 26%)" stroke="hsl(${h} 15% 40%)" stroke-width="2"/>
          <circle cx="130" cy="77" r="22" fill="hsl(${h} 10% 26%)" stroke="hsl(${h} 15% 40%)" stroke-width="2"/>
          <circle cx="70" cy="77" r="5" fill="hsl(${h} 15% 45%)"/><circle cx="130" cy="77" r="5" fill="hsl(${h} 15% 45%)"/>
          <rect x="40" y="110" width="90" height="8" rx="1" fill="hsl(45 70% 55%)"/>`;
        break;
      case "ai-systems":
      case "servers":
      case "storage": {
        const rows = p.category === "ai-systems" ? 5 : 4;
        shape = `<rect x="20" y="22" width="160" height="${rows * 20 + 16}" rx="6" fill="${dark}"/>` +
          Array.from({ length: rows }, (_, i) => p.category === "storage"
            ? Array.from({ length: 6 }, (_, j) => `<rect x="${30 + j * 23}" y="${30 + i * 20}" width="19" height="14" rx="2" fill="${mid}"/>`).join("")
            : `<rect x="30" y="${30 + i * 20}" width="140" height="13" rx="2" fill="${mid}"/><circle cx="160" cy="${36.5 + i * 20}" r="2.5" fill="${led}"/>`).join("");
        break;
      }
      case "workstations":
        shape = `<rect x="65" y="18" width="70" height="114" rx="7" fill="${dark}"/>
          ${[0, 1, 2, 3, 4].map((i) => `<rect x="75" y="${30 + i * 12}" width="50" height="6" rx="2" fill="${mid}"/>`).join("")}
          <circle cx="100" cy="112" r="5" fill="${led}"/>`;
        break;
      case "components":
        shape = `<rect x="55" y="30" width="90" height="90" rx="8" fill="${dark}"/>
          <rect x="75" y="50" width="50" height="50" rx="4" fill="${mid}"/>
          ${[0, 1, 2, 3, 4].map((i) => `<rect x="${64 + i * 17}" y="20" width="5" height="10" fill="${led}"/><rect x="${64 + i * 17}" y="120" width="5" height="10" fill="${led}"/>`).join("")}`;
        break;
      case "networking":
        shape = `<rect x="20" y="55" width="160" height="40" rx="5" fill="${dark}"/>
          ${Array.from({ length: 10 }, (_, i) => `<rect x="${30 + i * 14}" y="65" width="10" height="20" rx="1.5" fill="${mid}"/>`).join("")}
          <circle cx="172" cy="62" r="2.5" fill="${led}"/>`;
        break;
      case "power-racks":
        shape = `<rect x="60" y="14" width="80" height="122" rx="5" fill="${dark}"/>
          ${Array.from({ length: 7 }, (_, i) => `<rect x="68" y="${22 + i * 15}" width="64" height="10" rx="1.5" fill="${mid}"/>`).join("")}
          <circle cx="126" cy="27" r="2.5" fill="${led}"/>`;
        break;
      default:
        shape = `<path d="M40 95 C 70 40, 130 110, 160 55" stroke="${led}" stroke-width="8" fill="none" stroke-linecap="round"/>
          <rect x="22" y="85" width="26" height="20" rx="4" fill="${dark}"/><rect x="152" y="45" width="26" height="20" rx="4" fill="${dark}"/>`;
    }
    const badgeText = tag && (p.tag || (p.featured ? "Featured" : ""));
    return `<div class="art" style="background:linear-gradient(160deg,hsl(${h} 35% 30%),hsl(${h} 30% 10%))">
      ${badgeText ? `<span class="tag">${esc(badgeText)}</span>` : ""}
      <svg viewBox="0 0 200 150" aria-hidden="true">${shape}</svg></div>`;
  }

  function keySpec(p) {
    const s = p.specs || {};
    return [s.Memory || s.Capacity, s["Form factor"], s.Architecture, s.Power].filter(Boolean).slice(0, 2).join(" · ");
  }

  function card(p) {
    return `<a class="card" href="#/p/${encodeURIComponent(p.id)}">
      ${art(p)}
      <div class="body">
        ${p.brand ? `<div class="brand">${esc(p.brand)}</div>` : ""}
        <div class="name">${esc(p.name)}</div>
        <div class="meta">${esc(keySpec(p))}</div>
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
    const total = priced.reduce((s, l) => s + l.p.price * l.qty, 0);
    return { total, priced: priced.length, quoteItems: lines.length - priced.length, units: lines.reduce((s, l) => s + l.qty, 0) };
  }
  function totalLabel(t) {
    if (!t.priced) return "On request";
    return money.format(t.total) + (t.quoteItems ? " +" : "");
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
    const inCat = inCategory(shop.category);
    const brands = brandsOf(inCat);
    if (shop.brand && !brands.includes(shop.brand)) shop.brand = "";
    const list = shop.brand ? inCat.filter((p) => p.brand === shop.brand) : inCat;
    const featured = products.filter((p) => p.featured);
    const showFeatured = shop.category === "all" && !shop.brand;
    view.innerHTML = `
      <section class="hero">
        <small>AI &amp; HPC hardware</small>
        <h2>GPUs, AI servers and data center gear</h2>
        <p>${products.length} products from NVIDIA, Supermicro, GIGABYTE, Dell, HPE, Lenovo and more. Request a quote in seconds.</p>
        <button class="btn" data-cat="ai-systems">Explore AI systems</button>
      </section>
      <div class="chips" role="tablist">
        ${categories.map((c) => `<button class="chip ${c.id === shop.category ? "active" : ""}" data-cat="${c.id}" role="tab" aria-selected="${c.id === shop.category}">${esc(c.name)} <span class="count">${inCategory(c.id).length}</span></button>`).join("")}
      </div>
      ${showFeatured ? `
        <div class="section-title">Featured</div>
        <div class="rail">${featured.map(card).join("")}</div>` : ""}
      <div class="list-head">
        <div class="section-title">${esc(catName(shop.category))} · ${list.length} products</div>
        ${brands.length > 1 ? `<select id="brand" aria-label="Filter by brand">
          <option value="">All brands</option>
          ${brands.map((b) => `<option ${b === shop.brand ? "selected" : ""}>${esc(b)}</option>`).join("")}
        </select>` : ""}
      </div>
      <div class="grid">${list.slice(0, shop.shown).map(card).join("")}</div>
      ${list.length > shop.shown ? `<button class="btn block ghost more" id="more">Show more (${list.length - shop.shown} left)</button>` : ""}`;
    view.querySelectorAll("[data-cat]").forEach((el) =>
      el.addEventListener("click", () => {
        Object.assign(shop, { category: el.dataset.cat, brand: "", shown: PAGE_SIZE });
        shopView();
        const target = el.classList.contains("chip") ? null : view.querySelector(".chips");
        if (target) target.scrollIntoView({ behavior: "smooth" });
        else view.querySelector(`.chip[data-cat="${shop.category}"]`)?.scrollIntoView({ inline: "center", block: "nearest" });
      })
    );
    const brandSel = view.querySelector("#brand");
    if (brandSel) brandSel.onchange = () => { shop.brand = brandSel.value; shop.shown = PAGE_SIZE; keepScroll(shopView); };
    const more = view.querySelector("#more");
    if (more) more.onclick = () => { shop.shown += PAGE_SIZE; keepScroll(shopView); };
  }

  function keepScroll(render) {
    const y = window.scrollY;
    render();
    window.scrollTo(0, y);
  }

  function productView(id) {
    const p = byId[id];
    if (!p) return notFound();
    setChrome(catName(p.category), { back: true, tab: "shop" });
    let qty = 1;
    const specs = Object.entries(p.specs || {});
    const related = products.filter((x) => x.category === p.category && x.id !== p.id && (x.brand === p.brand || !p.brand)).slice(0, 8);
    view.innerHTML = `
      <div class="detail">
        <div class="detail-art">${art(p)}</div>
        ${p.brand ? `<div class="brand">${esc(p.brand)}</div>` : ""}
        <h2>${esc(p.name)}</h2>
        <div class="price-lg">${priceLabel(p)}</div>
        <div class="stock">${p.price == null ? "Volume and project pricing available" : "Ships in 2–5 days"}</div>
        <p>${esc(p.summary)}</p>
        ${specs.length ? `<dl class="specs">${specs.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>` : ""}
        <div class="buybar">
          <div class="qty"><button id="minus" aria-label="Decrease quantity">−</button><span id="q">1</span><button id="plus" aria-label="Increase quantity">+</button></div>
          <button class="btn grow" id="add">${p.price == null ? "Add to quote" : "Add to cart"}</button>
        </div>
        ${p.url ? `<a class="btn block ghost" style="margin-top:10px" href="${esc(p.url)}" target="_blank" rel="noopener">Full specs on exeton.com</a>` : ""}
        ${related.length ? `<div class="section-title" style="margin-top:24px">Related products</div><div class="rail">${related.map(card).join("")}</div>` : ""}
      </div>`;
    const q = view.querySelector("#q");
    view.querySelector("#minus").onclick = () => { qty = Math.max(1, qty - 1); q.textContent = qty; };
    view.querySelector("#plus").onclick = () => { qty = Math.min(99, qty + 1); q.textContent = qty; };
    view.querySelector("#add").onclick = () => {
      setCart(p.id, (cart[p.id] || 0) + qty);
      if (navigator.vibrate) navigator.vibrate(15);
      toast(`Added ${qty} × ${p.name}`);
    };
  }

  let lastSearch = "";
  function searchView() {
    setChrome("Search", { tab: "search" });
    view.innerHTML = `
      <div class="search">
        <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/></svg>
        <input id="q" type="search" placeholder="H200, RTX 5090, Supermicro, DDR5…" autocomplete="off" enterkeyhint="search" aria-label="Search products">
      </div>
      <div id="results"></div>`;
    const input = view.querySelector("#q");
    const results = view.querySelector("#results");
    const haystack = products.map((p) => [p.name, p.brand, p.summary, catName(p.category), ...Object.values(p.specs || {})].join(" ").toLowerCase());
    const run = () => {
      lastSearch = input.value;
      const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!terms.length) {
        results.innerHTML = `<div class="section-title">Browse categories</div>
          <div class="list">${categories.filter((c) => c.id !== "all").map((c) => `
            <a class="row" href="#/" data-cat="${c.id}"><div class="info"><div class="name">${esc(c.name)}</div><div class="meta">${inCategory(c.id).length} products</div></div>
            <svg viewBox="0 0 24 24" style="color:var(--muted);width:18px"><path d="M9 5l7 7-7 7"/></svg></a>`).join("")}</div>`;
        results.querySelectorAll("[data-cat]").forEach((a) => a.addEventListener("click", () => Object.assign(shop, { category: a.dataset.cat, brand: "", shown: PAGE_SIZE })));
        return;
      }
      const hits = products.filter((_, i) => terms.every((t) => haystack[i].includes(t)));
      results.innerHTML = hits.length
        ? `<div class="section-title">${hits.length} result${hits.length > 1 ? "s" : ""}</div><div class="grid">${hits.slice(0, 60).map(card).join("")}</div>
           ${hits.length > 60 ? `<p class="note" style="text-align:center">Showing 60 of ${hits.length}. Add another word to narrow it down.</p>` : ""}`
        : `<div class="empty"><h3>No matches</h3>Try a model like “H200”, a brand like “Supermicro” or a spec like “DDR5”.</div>`;
    };
    input.value = lastSearch;
    input.addEventListener("input", run);
    run();
  }

  function cartView() {
    setChrome("Quote", { tab: "cart" });
    const lines = cartLines();
    if (!lines.length) {
      view.innerHTML = `<div class="empty">
        <svg viewBox="0 0 24 24"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h6"/></svg>
        <h3>Your quote is empty</h3><p>Add GPUs, servers or components and we'll price them for you.</p>
        <a class="btn" href="#/">Browse products</a></div>`;
      return;
    }
    const t = totals(lines);
    view.innerHTML = `
      <div class="list">${lines.map(({ p, qty }) => `
        <div class="row">
          <a class="thumb" href="#/p/${encodeURIComponent(p.id)}">${art(p, { tag: false })}</a>
          <div class="info">
            <div class="name">${esc(p.name)}</div>
            <div class="meta">${p.price == null ? "Price on request" : money.format(p.price) + " each"}</div>
            <button class="remove" data-remove="${esc(p.id)}">Remove</button>
          </div>
          <div class="qty"><button data-dec="${esc(p.id)}" aria-label="Decrease">−</button><span>${qty}</span><button data-inc="${esc(p.id)}" aria-label="Increase">+</button></div>
        </div>`).join("")}
      </div>
      <div class="totals">
        <div><span>Products</span><span>${lines.length}</span></div>
        <div><span>Units</span><span>${t.units}</span></div>
        <div class="grand"><span>Estimated total</span><span>${totalLabel(t)}</span></div>
      </div>
      <p class="note">Exeton's sales team confirms pricing, availability and lead time for every item.</p>
      <a class="btn block" href="#/checkout">Request quote</a>`;
    view.querySelectorAll("[data-remove]").forEach((b) => (b.onclick = () => { setCart(b.dataset.remove, 0); cartView(); }));
    view.querySelectorAll("[data-dec]").forEach((b) => (b.onclick = () => { setCart(b.dataset.dec, cart[b.dataset.dec] - 1); cartView(); }));
    view.querySelectorAll("[data-inc]").forEach((b) => (b.onclick = () => { setCart(b.dataset.inc, cart[b.dataset.inc] + 1); cartView(); }));
  }

  function checkoutView() {
    const lines = cartLines();
    if (!lines.length) { location.hash = "#/cart"; return; }
    setChrome("Request quote", { back: true, tab: "cart" });
    const t = totals(lines);
    const saved = load("exeton.customer", {});
    view.innerHTML = `
      <form class="checkout" novalidate>
        <div class="section-title">Contact</div>
        <label>Full name<input name="name" required autocomplete="name" value="${esc(saved.name)}"></label>
        <label>Company<input name="company" autocomplete="organization" value="${esc(saved.company)}"></label>
        <div class="two">
          <label>Email<input name="email" type="email" required autocomplete="email" inputmode="email" value="${esc(saved.email)}"></label>
          <label>Phone<input name="phone" type="tel" autocomplete="tel" inputmode="tel" value="${esc(saved.phone)}"></label>
        </div>
        <div class="section-title">Delivery</div>
        <label>Address<input name="address" required autocomplete="street-address" value="${esc(saved.address)}"></label>
        <div class="two">
          <label>City<input name="city" required autocomplete="address-level2" value="${esc(saved.city)}"></label>
          <label>Country<input name="country" required autocomplete="country-name" value="${esc(saved.country)}"></label>
        </div>
        <label>Preferred payment
          <select name="payment">
            <option>Bank transfer / invoice</option>
            <option>Purchase order</option>
            <option>Credit card</option>
            <option>Leasing / financing</option>
          </select>
        </label>
        <label>Notes<textarea name="notes" placeholder="Configuration, delivery timeline, budget…"></textarea></label>
        <div class="totals">
          <div><span>Products</span><span>${lines.length} (${t.units} units)</span></div>
          <div class="grand"><span>Estimated total</span><span>${totalLabel(t)}</span></div>
        </div>
        <p class="note">No payment is taken now. Your request is saved on this device${SALES_EMAIL ? " and you can email it to Exeton sales" : ""}.</p>
        <button class="btn block" type="submit">Send quote request</button>
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
        items: lines.map(({ p, qty }) => ({ id: p.id, name: p.name, price: p.price, qty, url: p.url })),
        total: t.total, priced: t.priced, quoteItems: t.quoteItems, payment, notes, customer
      };
      orders.unshift(order);
      save("exeton.orders", orders);
      cart = {}; save("exeton.cart", cart); updateBadge();
      location.hash = `#/order/${order.id}`;
    });
  }

  function mailtoFor(o) {
    const c = o.customer;
    const body = [
      `Quote request ${o.id}`, "",
      ...o.items.map((i) => `${i.qty} x ${i.name}${i.url ? `\n   ${i.url}` : ""}`), "",
      `Name: ${c.name}`, c.company && `Company: ${c.company}`, `Email: ${c.email}`, c.phone && `Phone: ${c.phone}`,
      `Delivery: ${c.address}, ${c.city}, ${c.country}`, `Preferred payment: ${o.payment}`, o.notes && `Notes: ${o.notes}`
    ].filter((x) => x !== undefined && x !== "").join("\n");
    return `mailto:${SALES_EMAIL}?subject=${encodeURIComponent(`Quote request ${o.id}`)}&body=${encodeURIComponent(body)}`;
  }

  function orderView(id) {
    const o = orders.find((x) => x.id === id);
    if (!o) return notFound();
    setChrome("Quote request", { back: true, tab: "account" });
    view.innerHTML = `
      <div class="empty" style="padding:30px 10px">
        <svg viewBox="0 0 24 24" style="color:var(--accent)"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/></svg>
        <h3>Quote request saved</h3>
        <p>Reference <b>${esc(o.id)}</b> · ${new Date(o.date).toLocaleString()}</p>
        <p>Exeton will reply to ${esc(o.customer.email)} with pricing and lead times.</p>
      </div>
      <div class="list">${o.items.map((i) => `
        <a class="row" href="#/p/${encodeURIComponent(i.id)}"><div class="info"><div class="name">${esc(i.name)}</div><div class="meta">Qty ${i.qty} · ${i.price == null ? "Price on request" : money.format(i.price * i.qty)}</div></div></a>`).join("")}
      </div>
      <div class="totals"><div class="grand"><span>Estimated total</span><span>${totalLabel(o)}</span></div></div>
      ${SALES_EMAIL ? `<a class="btn block" href="${esc(mailtoFor(o))}">Email this request to Exeton</a>` : ""}
      <a class="btn block ghost" style="margin-top:10px" href="#/">Continue shopping</a>`;
  }

  function accountView() {
    setChrome("Account", { tab: "account" });
    const c = load("exeton.customer", {});
    const standalone = window.navigator.standalone || matchMedia("(display-mode: standalone)").matches;
    view.innerHTML = `
      <div class="section-title">Profile</div>
      <div class="list"><div class="row"><div class="info">
        <div class="name">${esc(c.name || "Guest")}</div>
        <div class="meta">${esc(c.email || "Your details are saved when you request a quote")}</div>
      </div></div></div>
      <div class="section-title" style="margin-top:20px">Quote requests</div>
      ${orders.length ? `<div class="list">${orders.map((o) => `
        <a class="row" href="#/order/${o.id}"><div class="info">
          <div class="name">${esc(o.id)}</div>
          <div class="meta">${new Date(o.date).toLocaleDateString()} · ${o.items.reduce((s, i) => s + i.qty, 0)} units · ${totalLabel(o)}</div>
        </div><svg viewBox="0 0 24 24" style="color:var(--muted);width:18px"><path d="M9 5l7 7-7 7"/></svg></a>`).join("")}</div>`
        : `<div class="list"><div class="row"><div class="info"><div class="meta">No quote requests yet.</div></div></div></div>`}
      ${standalone ? "" : `<div class="install-tip"><b>Install on iPhone:</b> tap the Share button in Safari, then <b>Add to Home Screen</b>. Exeton opens full-screen like a native app.</div>`}
      ${orders.length ? `<button class="btn block ghost" id="clear" style="margin-top:18px">Clear history</button>` : ""}`;
    const clear = view.querySelector("#clear");
    if (clear) clear.onclick = () => {
      if (!confirm("Delete all saved quote requests on this device?")) return;
      orders = []; save("exeton.orders", orders); accountView();
    };
  }

  function notFound() {
    setChrome("Not found", { back: true });
    view.innerHTML = `<div class="empty"><h3>Page not found</h3><a class="btn" href="#/">Go to shop</a></div>`;
  }

  // ---------- router ----------
  const scrollMemory = {};
  function route() {
    const [page, arg] = location.hash.replace(/^#\/?/, "").split("/");
    toastEl.classList.remove("show");
    switch (page) {
      case "": shopView(); break;
      case "p": productView(decodeURIComponent(arg || "")); break;
      case "search": searchView(); break;
      case "cart": cartView(); break;
      case "checkout": checkoutView(); break;
      case "order": orderView(arg); break;
      case "account": accountView(); break;
      default: notFound();
    }
    // Return to the previous scroll position on the shop list after "Back"; otherwise start at the top.
    window.scrollTo(0, scrollMemory[location.hash] || 0);
  }

  window.addEventListener("scroll", () => { scrollMemory[location.hash] = window.scrollY; }, { passive: true });
  backBtn.addEventListener("click", () => (history.length > 1 ? history.back() : (location.hash = "#/")));
  window.addEventListener("hashchange", route);
  updateBadge();
  route();

  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
})();

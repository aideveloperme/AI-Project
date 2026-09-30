/** Static export: the dashboard is plain files served by nginx, which also
 * reverse-proxies /api to the backend (no Node runtime in production, easy to
 * run air-gapped). In `next dev`, /api is proxied to the local backend. */
const isDev = process.env.NODE_ENV !== "production";
const backend = process.env.SENTINEL_API_URL || "http://localhost:8000";

const config = {
  reactStrictMode: true,
  ...(isDev
    ? { async rewrites() { return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }]; } }
    : { output: "export", trailingSlash: true }),
  images: { unoptimized: true },
};
export default config;

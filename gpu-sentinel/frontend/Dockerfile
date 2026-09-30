# Build the static dashboard, serve it with nginx (which also proxies /api → backend).
FROM node:22-alpine AS build
WORKDIR /src
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY . .
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
RUN npx next build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY --from=build /src/out /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080

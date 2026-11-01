# REST gateway Python -> Hyperledger Fabric
FROM node:22-slim

ENV NODE_ENV=production
WORKDIR /app

COPY package*.json ./
# --omit=optional skips pkcs11js (HSM support, needs a C++ toolchain, not used here)
RUN npm install --omit=dev --omit=optional --no-audit --no-fund

COPY server.js .

USER node
EXPOSE 3001
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s \
  CMD node -e "fetch('http://localhost:3001/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
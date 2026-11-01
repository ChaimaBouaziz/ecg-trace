'use strict';

/**
 * Blockchain gateway: a small REST API in front of Hyperledger Fabric.
 *
 * Why: the official Fabric Gateway SDK exists for Node/Go/Java, not Python.
 * The Python pipeline (ledger/fabric_client.py) calls this service over HTTP;
 * this service signs and submits transactions with the Org1 "User1" identity.
 *
 *   POST /artifacts                 body = ArtifactRecord JSON      -> 201 stored record (+ tx_id)
 *   GET  /artifacts/:id                                             -> 200 record | 404
 *   GET  /artifacts/:id/key-history                                 -> writes on that key
 *   GET  /cases/:caseId/history                                     -> 200 [records]
 *   GET  /health
 *
 * Config (env vars, defaults match fabric-samples/test-network):
 *   FABRIC_SAMPLES=~/fabric-samples  CHANNEL=ecgchannel  CHAINCODE=ecgtrace  PORT=3001
 */

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const express = require('express');
const grpc = require('@grpc/grpc-js');
const { connect, hash, signers } = require('@hyperledger/fabric-gateway');

const SAMPLES = process.env.FABRIC_SAMPLES || path.join(os.homedir(), 'fabric-samples');
const CHANNEL = process.env.CHANNEL || 'ecgchannel';
const CHAINCODE = process.env.CHAINCODE || 'ecgtrace';
const PORT = Number(process.env.PORT || 3001);
const MSP_ID = process.env.MSP_ID || 'Org1MSP';
const ORG_DIR = path.join(SAMPLES, 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com');
const USER_DIR = path.join(ORG_DIR, 'users', process.env.FABRIC_USER || 'User1@org1.example.com', 'msp');
const PEER_ENDPOINT = process.env.PEER_ENDPOINT || 'localhost:7051';
const PEER_HOST_ALIAS = process.env.PEER_HOST_ALIAS || 'peer0.org1.example.com';
const TLS_CERT = path.join(ORG_DIR, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt');

const firstFile = (dir) => path.join(dir, fs.readdirSync(dir)[0]);

async function newContract() {
    const client = new grpc.Client(PEER_ENDPOINT, grpc.credentials.createSsl(fs.readFileSync(TLS_CERT)), {
        'grpc.ssl_target_name_override': PEER_HOST_ALIAS,
    });
    const gateway = connect({
        client,
        identity: { mspId: MSP_ID, credentials: fs.readFileSync(firstFile(path.join(USER_DIR, 'signcerts'))) },
        signer: signers.newPrivateKeySigner(crypto.createPrivateKey(fs.readFileSync(firstFile(path.join(USER_DIR, 'keystore'))))),
        hash: hash.sha256,
        evaluateOptions: () => ({ deadline: Date.now() + 5000 }),
        endorseOptions: () => ({ deadline: Date.now() + 15000 }),
        submitOptions: () => ({ deadline: Date.now() + 5000 }),
        commitStatusOptions: () => ({ deadline: Date.now() + 60000 }),
    });
    return gateway.getNetwork(CHANNEL).getContract(CHAINCODE);
}

const decode = (bytes) => new TextDecoder().decode(bytes);

// Fabric errors carry the chaincode message in err.details[].message
function chaincodeMessage(err) {
    const details = (err.details || []).map((d) => d.message).join(' | ');
    return details || err.message || String(err);
}

function httpStatus(msg) {
    if (/does not exist/.test(msg) && !/Parent/.test(msg)) return 404;
    if (/already exists/.test(msg)) return 409;
    if (/Access denied/.test(msg)) return 403;
    if (/Missing fields|Invalid JSON|sha256 must|Unknown artifact_type|Parent|parent_ids/.test(msg)) return 400;
    return 500;
}

async function main() {
    const contract = await newContract();
    const app = express();
    app.use(express.json({ limit: '1mb' }));

    const handle = (fn) => async (req, res) => {
        try {
            await fn(req, res);
        } catch (err) {
            const msg = chaincodeMessage(err);
            const code = httpStatus(msg);
            if (code === 500) console.error(err);
            res.status(code).json({ error: msg });
        }
    };

    app.get('/health', handle(async (req, res) => {
        res.json({ status: 'ok', channel: CHANNEL, chaincode: CHAINCODE, msp: MSP_ID });
    }));

    app.post('/artifacts', handle(async (req, res) => {
        const submitted = await contract.submitAsync('CreateArtifact', { arguments: [JSON.stringify(req.body)] });
        const status = await submitted.getStatus();
        if (!status.successful) {
            throw new Error(`Transaction ${status.transactionId} failed to commit (code ${status.code})`);
        }
        const record = JSON.parse(decode(submitted.getResult()));
        console.log(`[tx ${status.transactionId.slice(0, 12)}…] ${record.artifact_type} ${record.artifact_id}`);
        res.status(201).json(record);
    }));

    app.get('/artifacts/:id', handle(async (req, res) => {
        res.json(JSON.parse(decode(await contract.evaluateTransaction('ReadArtifact', req.params.id))));
    }));

    app.get('/artifacts/:id/key-history', handle(async (req, res) => {
        res.json(JSON.parse(decode(await contract.evaluateTransaction('GetArtifactKeyHistory', req.params.id))));
    }));

    app.get('/cases/:caseId/history', handle(async (req, res) => {
        res.json(JSON.parse(decode(await contract.evaluateTransaction('GetCaseHistory', req.params.caseId))));
    }));

    app.listen(PORT, () => console.log(`Blockchain gateway on http://localhost:${PORT} (channel=${CHANNEL}, chaincode=${CHAINCODE})`));
}

main().catch((err) => {
    console.error('Gateway failed to start:', err.message);
    console.error(`Is the Fabric test-network running and FABRIC_SAMPLES (${SAMPLES}) correct?`);
    process.exit(1);
});
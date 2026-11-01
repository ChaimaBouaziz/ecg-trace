'use strict';

// Unit tests with an in-memory mock of the Fabric stub — run with: npm test
const test = require('node:test');
const assert = require('node:assert');
const { EcgTraceContract } = require('../lib/ecgTraceContract');

function mockCtx(msp = 'Org1MSP') {
    const state = new Map();
    let txCounter = 0;
    const history = new Map();
    const stub = {
        getState: async (k) => state.get(k) || Buffer.alloc(0),
        putState: async (k, v) => {
            state.set(k, v);
            if (!history.has(k)) history.set(k, []);
            history.get(k).push({ txId: `tx${txCounter}`, isDelete: false });
        },
        getTxID: () => `tx${++txCounter}`,
        getTxTimestamp: () => ({ seconds: 1767225600, nanos: 0 }),
        createCompositeKey: (type, attrs) => `\u0000${type}\u0000${attrs.join('\u0000')}\u0000`,
        splitCompositeKey: (key) => {
            const parts = key.split('\u0000').filter(Boolean);
            return { objectType: parts[0], attributes: parts.slice(1) };
        },
        getStateByPartialCompositeKey: async (type, attrs) => {
            const prefix = `\u0000${type}\u0000${attrs.join('\u0000')}\u0000`;
            const keys = [...state.keys()].filter((k) => k.startsWith(prefix)).sort();
            let i = 0;
            return {
                next: async () => (i < keys.length ? { value: { key: keys[i++] }, done: false } : { done: true }),
                close: async () => {},
            };
        },
        getHistoryForKey: async (k) => {
            const h = history.get(k) || [];
            let i = 0;
            return { next: async () => (i < h.length ? { value: h[i++], done: false } : { done: true }), close: async () => {} };
        },
        setEvent: () => {},
    };
    return { stub, clientIdentity: { getMSPID: () => msp, getID: () => 'x509::CN=User1' } };
}

const rec = (id, type, parents = [], caseId = 'CASE-1', t = '2026-11-01T10:00:00+00:00') => JSON.stringify({
    artifact_id: id, case_id: caseId, artifact_type: type, sha256: 'a'.repeat(64), created_at: t,
    software_version: 'test-1', parent_ids: parents, offchain_uri: `${caseId}/${id}.json`, creator_id: 'agent.test',
});

test('create, read and case history', async () => {
    const c = new EcgTraceContract();
    const ctx = mockCtx();
    const out = JSON.parse(await c.CreateArtifact(ctx, rec('RAW-1', 'RAW_ECG')));
    assert.strictEqual(out.tx_id, 'tx1');
    assert.match(out.submitter, /^Org1MSP::/);
    await c.CreateArtifact(ctx, rec('PROC-1', 'PROCESSED', ['RAW-1'], 'CASE-1', '2026-11-01T10:00:01+00:00'));
    await c.CreateArtifact(ctx, rec('RAW-9', 'RAW_ECG', [], 'CASE-2'));
    const read = JSON.parse(await c.ReadArtifact(ctx, 'PROC-1'));
    assert.deepStrictEqual(read.parent_ids, ['RAW-1']);
    const hist = JSON.parse(await c.GetCaseHistory(ctx, 'CASE-1'));
    assert.deepStrictEqual(hist.map((r) => r.artifact_id), ['RAW-1', 'PROC-1']);
});

test('append-only: no overwrite', async () => {
    const c = new EcgTraceContract();
    const ctx = mockCtx();
    await c.CreateArtifact(ctx, rec('RAW-1', 'RAW_ECG'));
    await assert.rejects(c.CreateArtifact(ctx, rec('RAW-1', 'RAW_ECG')), /already exists/);
});

test('parents must exist and belong to the same case', async () => {
    const c = new EcgTraceContract();
    const ctx = mockCtx();
    await assert.rejects(c.CreateArtifact(ctx, rec('P', 'PROCESSED', ['NOPE'])), /does not exist/);
    await c.CreateArtifact(ctx, rec('RAW-2', 'RAW_ECG', [], 'CASE-2'));
    await assert.rejects(c.CreateArtifact(ctx, rec('P', 'PROCESSED', ['RAW-2'])), /another case/);
});

test('input validation', async () => {
    const c = new EcgTraceContract();
    const ctx = mockCtx();
    const bad = JSON.parse(rec('X', 'RAW_ECG'));
    await assert.rejects(c.CreateArtifact(ctx, JSON.stringify({ ...bad, sha256: 'xyz' })), /sha256/);
    await assert.rejects(c.CreateArtifact(ctx, JSON.stringify({ ...bad, artifact_type: 'FOO' })), /Unknown/);
    delete bad.case_id;
    await assert.rejects(c.CreateArtifact(ctx, JSON.stringify(bad)), /Missing fields: case_id/);
});

test('only authorised organisation can write', async () => {
    const c = new EcgTraceContract();
    await assert.rejects(c.CreateArtifact(mockCtx('Org2MSP'), rec('R', 'RAW_ECG')), /Access denied/);
});

test('read unknown artifact fails', async () => {
    const c = new EcgTraceContract();
    await assert.rejects(c.ReadArtifact(mockCtx(), 'NOPE'), /does not exist/);
});
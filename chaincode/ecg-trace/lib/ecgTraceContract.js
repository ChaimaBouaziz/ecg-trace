'use strict';

/**
 * ECG-Trace chaincode — stores artifact fingerprints + provenance (spec §4.6).
 *
 * World state:
 *   key  = artifact_id                     value = artifact record (JSON, sorted keys)
 *   key  = composite('case~artifact', [case_id, artifact_id])  -> index for case history
 *
 * Rules enforced ON-CHAIN (not only in Python):
 *   - an artifact can never be overwritten (append-only)
 *   - sha256 must be 64 hex chars, artifact_type must be known
 *   - every parent must already exist on the ledger and belong to the same case
 *   - only authorised organisations (WRITER_MSPS) may create records
 *   - the submitter identity and ledger timestamp are recorded by the chaincode itself
 */

const { Contract } = require('fabric-contract-api');

const ARTIFACT_TYPES = ['RAW_ECG', 'RAW_REPORT', 'PROCESSED', 'FEATURES',
    'PREDICTION', 'VALIDATION', 'EXPLANATION', 'FINAL'];
const REQUIRED = ['artifact_id', 'case_id', 'artifact_type', 'sha256', 'created_at',
    'software_version', 'parent_ids', 'offchain_uri', 'creator_id'];
const WRITER_MSPS = ['Org1MSP'];          // the "hospital AI service" organisation
const CASE_INDEX = 'case~artifact';

// Deterministic JSON (sorted keys) so that every endorsing peer produces identical bytes.
function stableStringify(value) {
    if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(',')}]`;
    }
    if (value && typeof value === 'object') {
        return `{${Object.keys(value).sort()
            .map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
    }
    return JSON.stringify(value);
}

function txTimestampISO(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    const seconds = typeof ts.seconds === 'object' && ts.seconds !== null && 'low' in ts.seconds
        ? ts.seconds.low : Number(ts.seconds);
    return new Date(seconds * 1000 + Math.floor(ts.nanos / 1e6)).toISOString();
}

class EcgTraceContract extends Contract {
    constructor() {
        super('EcgTraceContract');
    }

    async _get(ctx, artifactId) {
        const bytes = await ctx.stub.getState(artifactId);
        return bytes && bytes.length > 0 ? JSON.parse(bytes.toString()) : null;
    }

    async ArtifactExists(ctx, artifactId) {
        return (await this._get(ctx, artifactId)) !== null;
    }

    async CreateArtifact(ctx, recordJson) {
        const msp = ctx.clientIdentity.getMSPID();
        if (!WRITER_MSPS.includes(msp)) {
            throw new Error(`Access denied: organisation ${msp} is not allowed to create artifacts`);
        }

        let rec;
        try {
            rec = JSON.parse(recordJson);
        } catch (e) {
            throw new Error('Invalid JSON record');
        }
        const missing = REQUIRED.filter((f) => rec[f] === undefined || rec[f] === null);
        if (missing.length) {
            throw new Error(`Missing fields: ${missing.join(', ')}`);
        }
        if (!/^[0-9a-f]{64}$/.test(rec.sha256)) {
            throw new Error('sha256 must be 64 lowercase hex characters');
        }
        if (!ARTIFACT_TYPES.includes(rec.artifact_type)) {
            throw new Error(`Unknown artifact_type ${rec.artifact_type}`);
        }
        if (!Array.isArray(rec.parent_ids)) {
            throw new Error('parent_ids must be an array');
        }
        if (await this.ArtifactExists(ctx, rec.artifact_id)) {
            throw new Error(`Artifact ${rec.artifact_id} already exists`);
        }
        for (const parentId of rec.parent_ids) {
            const parent = await this._get(ctx, parentId);
            if (!parent) {
                throw new Error(`Parent ${parentId} does not exist`);
            }
            if (parent.case_id !== rec.case_id) {
                throw new Error(`Parent ${parentId} belongs to another case`);
            }
        }

        const stored = {
            artifact_id: rec.artifact_id,
            case_id: rec.case_id,
            artifact_type: rec.artifact_type,
            sha256: rec.sha256,
            created_at: rec.created_at,
            software_version: rec.software_version,
            parent_ids: rec.parent_ids,
            offchain_uri: rec.offchain_uri,
            creator_id: rec.creator_id,
            // set by the chaincode, cannot be forged by the client:
            tx_id: ctx.stub.getTxID(),
            ledger_timestamp: txTimestampISO(ctx),
            submitter: `${msp}::${ctx.clientIdentity.getID()}`,
        };
        const payload = stableStringify(stored);
        await ctx.stub.putState(rec.artifact_id, Buffer.from(payload));
        const indexKey = ctx.stub.createCompositeKey(CASE_INDEX, [rec.case_id, rec.artifact_id]);
        await ctx.stub.putState(indexKey, Buffer.from('\u0000'));
        ctx.stub.setEvent('ArtifactCreated', Buffer.from(stableStringify({
            artifact_id: stored.artifact_id, case_id: stored.case_id,
            artifact_type: stored.artifact_type, sha256: stored.sha256,
        })));
        return payload;
    }

    async ReadArtifact(ctx, artifactId) {
        const rec = await this._get(ctx, artifactId);
        if (!rec) {
            throw new Error(`Artifact ${artifactId} does not exist`);
        }
        return stableStringify(rec);
    }

    async GetCaseHistory(ctx, caseId) {
        const iterator = await ctx.stub.getStateByPartialCompositeKey(CASE_INDEX, [caseId]);
        const records = [];
        let res = await iterator.next();
        while (!res.done) {
            const { attributes } = ctx.stub.splitCompositeKey(res.value.key);
            const rec = await this._get(ctx, attributes[1]);
            if (rec) {
                records.push(rec);
            }
            res = await iterator.next();
        }
        await iterator.close();
        records.sort((a, b) => (a.created_at < b.created_at ? -1 : a.created_at > b.created_at ? 1 : 0));
        return stableStringify(records);
    }

    // Every write ever made to this key — shows the record was written once and never changed.
    async GetArtifactKeyHistory(ctx, artifactId) {
        const iterator = await ctx.stub.getHistoryForKey(artifactId);
        const out = [];
        let res = await iterator.next();
        while (!res.done) {
            out.push({ tx_id: res.value.txId, is_delete: res.value.isDelete });
            res = await iterator.next();
        }
        await iterator.close();
        return stableStringify(out);
    }
}

module.exports = { EcgTraceContract, stableStringify, WRITER_MSPS };
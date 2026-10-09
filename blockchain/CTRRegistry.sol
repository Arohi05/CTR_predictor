// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title CTRRegistry
/// @notice Public, append-only registry of fingerprints for the CTR predictor.
///         Raw data never goes on-chain, only SHA-256 hashes of it:
///         1) prediction batches (Merkle root of the prediction records), and
///         2) model provenance (dataset hash, model-file hash, metrics hash).
///         Anyone can later recompute a hash locally and compare it with the
///         value stored here to prove nothing was altered.
contract CTRRegistry {
    struct Batch {
        bytes32 blockHash;   // hash of the ledger block (commits to the previous block too)
        bytes32 merkleRoot;  // Merkle root of the prediction records in the batch
        uint32 recordCount;
        uint64 timestamp;
    }

    struct ModelInfo {
        bytes32 modelHash;
        bytes32 datasetHash;
        bytes32 metricsHash;
        string version;
        uint64 timestamp;
    }

    address public owner;
    Batch[] private _batches;
    ModelInfo[] private _models;

    event BatchAnchored(uint256 indexed index, bytes32 blockHash, bytes32 merkleRoot, uint32 recordCount);
    event ModelRegistered(uint256 indexed index, bytes32 modelHash, string version);

    error NotOwner();
    error EmptyBatch();
    error IndexOutOfRange();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // ------------------------------------------------------------ writes
    /// @notice Anchor a sealed ledger batch. Batches are numbered 0, 1, 2 ...
    ///         in the same order as the off-chain ledger.
    function anchorBatch(bytes32 blockHash, bytes32 merkleRoot, uint32 recordCount)
        external
        onlyOwner
        returns (uint256 index)
    {
        if (recordCount == 0) revert EmptyBatch();
        index = _batches.length;
        _batches.push(Batch(blockHash, merkleRoot, recordCount, uint64(block.timestamp)));
        emit BatchAnchored(index, blockHash, merkleRoot, recordCount);
    }

    /// @notice Register a trained model: what data it was trained on, the model
    ///         file itself and the evaluation metrics.
    function registerModel(
        bytes32 modelHash,
        bytes32 datasetHash,
        bytes32 metricsHash,
        string calldata version
    ) external onlyOwner returns (uint256 index) {
        index = _models.length;
        _models.push(ModelInfo(modelHash, datasetHash, metricsHash, version, uint64(block.timestamp)));
        emit ModelRegistered(index, modelHash, version);
    }

    // ------------------------------------------------------------- reads
    function batchCount() external view returns (uint256) {
        return _batches.length;
    }

    function modelCount() external view returns (uint256) {
        return _models.length;
    }

    function getBatch(uint256 index) external view returns (Batch memory) {
        if (index >= _batches.length) revert IndexOutOfRange();
        return _batches[index];
    }

    function getModel(uint256 index) external view returns (ModelInfo memory) {
        if (index >= _models.length) revert IndexOutOfRange();
        return _models[index];
    }

    /// @notice True if batch `index` was anchored with exactly this block hash.
    function verifyBatch(uint256 index, bytes32 blockHash) external view returns (bool) {
        return index < _batches.length && _batches[index].blockHash == blockHash;
    }

    /// @notice Is this model file hash registered? Returns (found, index).
    function findModel(bytes32 modelHash) external view returns (bool found, uint256 index) {
        for (uint256 i = _models.length; i > 0; i--) {
            if (_models[i - 1].modelHash == modelHash) return (true, i - 1);
        }
        return (false, 0);
    }
}

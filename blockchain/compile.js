// Compile CTRRegistry.sol -> CTRRegistry.json (ABI + bytecode).
//   npm install solc@0.8.24
//   node blockchain/compile.js
// The compiled JSON is committed, so you only need this after editing the contract.
const fs = require("fs");
const path = require("path");
const solc = require("solc");

const file = path.join(__dirname, "CTRRegistry.sol");
const input = {
  language: "Solidity",
  sources: { "CTRRegistry.sol": { content: fs.readFileSync(file, "utf8") } },
  settings: {
    optimizer: { enabled: true, runs: 200 },
    outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } },
  },
};

const out = JSON.parse(solc.compile(JSON.stringify(input)));
const errors = (out.errors || []).filter((e) => e.severity === "error");
(out.errors || []).forEach((e) => console.log(e.formattedMessage));
if (errors.length) process.exit(1);

const c = out.contracts["CTRRegistry.sol"].CTRRegistry;
fs.writeFileSync(
  path.join(__dirname, "CTRRegistry.json"),
  JSON.stringify({ contractName: "CTRRegistry", solc: solc.version(), abi: c.abi, bytecode: "0x" + c.evm.bytecode.object }, null, 2)
);
console.log("Compiled with", solc.version(), "-> blockchain/CTRRegistry.json");

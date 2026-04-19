```markdown
---
name: quip-protocol-rs
description: Rust implementation of the Quip Protocol blockchain node, forked from Substrate/Polkadot SDK solochain template
triggers:
  - build a substrate node in rust
  - quip protocol blockchain
  - substrate solochain template
  - create a custom pallet in substrate
  - run a substrate development chain
  - substrate FRAME runtime development
  - polkadot sdk rust blockchain
  - substrate node template setup
---

# Quip Protocol RS

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

A Rust implementation of the Quip Protocol blockchain node, forked from the [Substrate](https://substrate.io/) / Polkadot SDK solochain template. It provides a ready-to-hack blockchain node with FRAME pallets, AURA block authoring, GRANDPA finality, and a JSON-RPC server.

---

## What This Project Does

- Runs a standalone Substrate-based blockchain node (solochain)
- Uses FRAME to compose a runtime from pallets (modules)
- Exposes an RPC server (default `ws://localhost:9944`)
- Ships with a template pallet for custom business logic
- Supports single-node dev chains and multi-node testnets

---

## Project Structure

```
quip-protocol-rs/
├── node/
│   └── src/
│       ├── chain_spec.rs   # Genesis state configuration
│       ├── service.rs      # Node implementation (consensus, networking)
│       └── main.rs
├── runtime/
│   └── src/
│       └── lib.rs          # FRAME runtime, pallet composition
├── pallets/
│   └── template/
│       └── src/
│           └── lib.rs      # Example custom pallet
└── Cargo.toml
```

---

## Installation & Prerequisites

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
  build-essential clang curl git libssl-dev \
  llvm libudev-dev make protobuf-compiler

# macOS
brew install cmake protobuf
```

### Rust Toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

rustup default stable
rustup update
rustup target add wasm32-unknown-unknown
rustup component add rust-src
```

### Clone & Build

```bash
git clone https://github.com/QuipNetwork/quip-protocol-rs.git
cd quip-protocol-rs

# Full release build
cargo build --release

# Dev build (faster, for iteration)
cargo build
```

---

## Key CLI Commands

```bash
# Start single-node dev chain (no state persistence)
./target/release/solochain-template-node --dev

# Start dev chain with persistent state
mkdir -p ./my-chain-state
./target/release/solochain-template-node --dev --base-path ./my-chain-state/

# Purge dev chain state
./target/release/solochain-template-node purge-chain --dev

# Detailed logging
RUST_BACKTRACE=1 ./target/release/solochain-template-node -ldebug --dev

# Custom log targets
RUST_LOG=runtime=debug,txpool=trace ./target/release/solochain-template-node --dev

# Show all CLI options
./target/release/solochain-template-node --help

# Generate and open Rust docs
cargo +nightly doc --open
```

---

## Writing a Custom Pallet

Pallets live in `pallets/<name>/src/lib.rs`. The template pallet is the canonical starting point.

### Minimal Pallet Example

```rust
// pallets/my_pallet/src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

#[frame_support::pallet]
pub mod pallet {
    use frame_support::{
        pallet_prelude::*,
        traits::Currency,
    };
    use frame_system::pallet_prelude::*;

    type BalanceOf<T> = <<T as Config>::Currency as Currency<
        <T as frame_system::Config>::AccountId,
    >>::Balance;

    #[pallet::config]
    pub trait Config: frame_system::Config {
        /// The runtime event type.
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
        /// Currency used for staking.
        type Currency: Currency<Self::AccountId>;
        /// Max length of a stored value.
        #[pallet::constant]
        type MaxValueLength: Get<u32>;
    }

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    // Storage: map AccountId -> BoundedVec<u8>
    #[pallet::storage]
    #[pallet::getter(fn stored_value)]
    pub type StoredValue<T: Config> = StorageMap<
        _,
        Blake2_128Concat,
        T::AccountId,
        BoundedVec<u8, T::MaxValueLength>,
        OptionQuery,
    >;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// A value was stored. [who, value]
        ValueStored { who: T::AccountId, value: BoundedVec<u8, T::MaxValueLength> },
        /// A value was cleared. [who]
        ValueCleared { who: T::AccountId },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Value exceeds maximum allowed length.
        ValueTooLong,
        /// No value found for this account.
        NothingStored,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Store a value for the caller.
        #[pallet::call_index(0)]
        #[pallet::weight(T::DbWeight::get().writes(1))]
        pub fn store_value(
            origin: OriginFor<T>,
            value: Vec<u8>,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;

            let bounded: BoundedVec<u8, T::MaxValueLength> =
                value.try_into().map_err(|_| Error::<T>::ValueTooLong)?;

            StoredValue::<T>::insert(&who, &bounded);
            Self::deposit_event(Event::ValueStored { who, value: bounded });
            Ok(())
        }

        /// Clear the caller's stored value.
        #[pallet::call_index(1)]
        #[pallet::weight(T::DbWeight::get().writes(1))]
        pub fn clear_value(origin: OriginFor<T>) -> DispatchResult {
            let who = ensure_signed(origin)?;
            ensure!(StoredValue::<T>::contains_key(&who), Error::<T>::NothingStored);
            StoredValue::<T>::remove(&who);
            Self::deposit_event(Event::ValueCleared { who });
            Ok(())
        }
    }
}
```

### Adding the Pallet to `Cargo.toml`

```toml
# pallets/my_pallet/Cargo.toml
[package]
name = "pallet-my-pallet"
version = "0.1.0"
edition = "2021"

[dependencies]
codec = { package = "parity-scale-codec", version = "3.6.1", default-features = false, features = ["derive"] }
scale-info = { version = "2.10.0", default-features = false, features = ["derive"] }
frame-benchmarking = { git = "https://github.com/paritytech/polkadot-sdk", optional = true, default-features = false }
frame-support = { git = "https://github.com/paritytech/polkadot-sdk", default-features = false }
frame-system = { git = "https://github.com/paritytech/polkadot-sdk", default-features = false }

[features]
default = ["std"]
std = [
    "codec/std",
    "frame-support/std",
    "frame-system/std",
    "scale-info/std",
]
runtime-benchmarks = ["frame-benchmarking/runtime-benchmarks"]
try-runtime = ["frame-support/try-runtime"]
```

### Wiring the Pallet into the Runtime

```rust
// runtime/src/lib.rs

// 1. Declare the pallet parameter types
parameter_types! {
    pub const MaxValueLength: u32 = 256;
}

// 2. Implement Config for your pallet
impl pallet_my_pallet::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
    type Currency = Balances;
    type MaxValueLength = MaxValueLength;
}

// 3. Add to the #[runtime] macro construct_runtime! block
#[runtime]
mod runtime {
    // ... existing pallets ...
    #[runtime::pallet_index(42)]
    pub type MyPallet = pallet_my_pallet::Pallet<Runtime>;
}
```

---

## Chain Specification (Genesis Config)

```rust
// node/src/chain_spec.rs

use sc_service::ChainType;
use sp_consensus_aura::sr25519::AuthorityId as AuraId;
use sp_consensus_grandpa::AuthorityId as GrandpaId;
use sp_keyring::AccountKeyring;

pub fn development_config() -> Result<ChainSpec, String> {
    Ok(ChainSpec::builder(
        WASM_BINARY.ok_or_else(|| "Development wasm not available".to_string())?,
        None,
    )
    .with_name("Development")
    .with_id("dev")
    .with_chain_type(ChainType::Development)
    .with_genesis_config_patch(testnet_genesis(
        // Initial authorities (Aura, Grandpa)
        vec![authority_keys_from_seed("Alice")],
        // Sudo account
        AccountKeyring::Alice.to_account_id(),
        // Pre-funded accounts
        vec![
            AccountKeyring::Alice.to_account_id(),
            AccountKeyring::Bob.to_account_id(),
            AccountKeyring::AliceStash.to_account_id(),
            AccountKeyring::BobStash.to_account_id(),
        ],
        true,
    ))
    .build())
}

fn testnet_genesis(
    initial_authorities: Vec<(AuraId, GrandpaId)>,
    root_key: AccountId,
    endowed_accounts: Vec<AccountId>,
    _enable_println: bool,
) -> serde_json::Value {
    serde_json::json!({
        "balances": {
            "balances": endowed_accounts.iter().cloned()
                .map(|k| (k, 1u64 << 60))
                .collect::<Vec<_>>(),
        },
        "aura": {
            "authorities": initial_authorities.iter().map(|x| (x.0.clone())).collect::<Vec<_>>(),
        },
        "grandpa": {
            "authorities": initial_authorities.iter().map(|x| (x.1.clone(), 1)).collect::<Vec<_>>(),
        },
        "sudo": { "key": Some(root_key) },
    })
}
```

---

## Pallet Unit Tests

```rust
// pallets/my_pallet/src/tests.rs
use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok};

#[test]
fn store_value_works() {
    new_test_ext().execute_with(|| {
        System::set_block_number(1);
        assert_ok!(MyPallet::store_value(
            RuntimeOrigin::signed(1),
            b"hello world".to_vec()
        ));
        assert_eq!(MyPallet::stored_value(1).unwrap().as_slice(), b"hello world");
        System::assert_last_event(
            Event::ValueStored {
                who: 1,
                value: b"hello world".to_vec().try_into().unwrap(),
            }.into()
        );
    });
}

#[test]
fn store_value_fails_when_too_long() {
    new_test_ext().execute_with(|| {
        let too_long = vec![0u8; 257]; // MaxValueLength = 256
        assert_noop!(
            MyPallet::store_value(RuntimeOrigin::signed(1), too_long),
            Error::<Test>::ValueTooLong
        );
    });
}

#[test]
fn clear_value_fails_when_nothing_stored() {
    new_test_ext().execute_with(|| {
        assert_noop!(
            MyPallet::clear_value(RuntimeOrigin::signed(1)),
            Error::<Test>::NothingStored
        );
    });
}
```

### Mock Runtime for Tests

```rust
// pallets/my_pallet/src/mock.rs
use crate as pallet_my_pallet;
use frame_support::{
    construct_runtime, parameter_types,
    traits::{ConstU16, ConstU32, ConstU64},
};
use sp_runtime::{traits::{BlakeTwo256, IdentityLookup}, BuildStorage};

type Block = frame_system::mocking::MockBlock<Test>;

construct_runtime!(
    pub enum Test {
        System: frame_system,
        Balances: pallet_balances,
        MyPallet: pallet_my_pallet,
    }
);

parameter_types! {
    pub const MaxValueLength: u32 = 256;
}

impl frame_system::Config for Test {
    type BaseCallFilter = frame_support::traits::Everything;
    type BlockWeights = ();
    type BlockLength = ();
    type DbWeight = ();
    type RuntimeOrigin = RuntimeOrigin;
    type RuntimeCall = RuntimeCall;
    type Nonce = u64;
    type Hash = sp_core::H256;
    type Hashing = BlakeTwo256;
    type AccountId = u64;
    type Lookup = IdentityLookup<Self::AccountId>;
    type Block = Block;
    type RuntimeEvent = RuntimeEvent;
    type BlockHashCount = ConstU64<250>;
    type Version = ();
    type PalletInfo = PalletInfo;
    type AccountData = pallet_balances::AccountData<u64>;
    type OnNewAccount = ();
    type OnKilledAccount = ();
    type SystemWeightInfo = ();
    type SS58Prefix = ConstU16<42>;
    type OnSetCode = ();
    type MaxConsumers = ConstU32<16>;
}

impl pallet_balances::Config for Test {
    type MaxLocks = ConstU32<50>;
    type MaxReserves = ();
    type ReserveIdentifier = [u8; 8];
    type Balance = u64;
    type RuntimeEvent = RuntimeEvent;
    type DustRemoval = ();
    type ExistentialDeposit = ConstU64<1>;
    type AccountStore = System;
    type WeightInfo = ();
    type FreezeIdentifier = ();
    type MaxFreezes = ();
    type RuntimeHoldReason = ();
    type MaxHolds = ();
}

impl pallet_my_pallet::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type Currency = Balances;
    type MaxValueLength = MaxValueLength;
}

pub fn new_test_ext() -> sp_io::TestExternalities {
    let t = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();
    let mut ext = sp_io::TestExternalities::new(t);
    ext.execute_with(|| System::set_block_number(1));
    ext
}
```

---

## Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific pallet
cargo test -p pallet-my-pallet

# Run with output
cargo test -- --nocapture

# Run a specific test
cargo test -p pallet-my-pallet store_value_works
```

---

## RPC / Frontend Integration

Connect to the running node via Polkadot-JS Apps:

- Hosted: https://polkadot.js.org/apps/#/explorer?rpc=ws://localhost:9944
- Default WebSocket endpoint: `ws://127.0.0.1:9944`
- Default HTTP endpoint: `http://127.0.0.1:9933`

### Interacting via `@polkadot/api` (JavaScript)

```javascript
import { ApiPromise, WsProvider } from '@polkadot/api';

const provider = new WsProvider('ws://127.0.0.1:9944');
const api = await ApiPromise.create({ provider });

// Query storage
const value = await api.query.myPallet.storedValue('5GrwvaEF...');

// Submit extrinsic
const keyring = new Keyring({ type: 'sr25519' });
const alice = keyring.addFromUri('//Alice');
await api.tx.myPallet
  .storeValue(Buffer.from('hello'))
  .signAndSend(alice);
```

---

## Docker

```bash
# Build Docker image (follows upstream Substrate Docker instructions)
docker build -f ./docker/Dockerfile -t quip-node .

# Run dev node in Docker
docker run --rm -p 9944:9944 -p 9933:9933 quip-node \
  --dev --ws-external --rpc-external
```

---

## Nix Development Environment

```bash
# Install nix + nix-direnv, then:
direnv allow
# All Rust/WASM dependencies are automatically available
```

---

## Common Patterns & Tips

### Accessing Storage in a Pallet

```rust
// Single value
#[pallet::storage]
pub type MyValue<T> = StorageValue<_, u32, ValueQuery>;

// Map
#[pallet::storage]
pub type MyMap<T: Config> = StorageMap<_, Blake2_128Concat, T::AccountId, u32, ValueQuery>;

// Double map
#[pallet::storage]
pub type MyDoubleMap<T: Config> = StorageDoubleMap<
    _, Blake2_128Concat, T::AccountId, Blake2_128Concat, u32, Vec<u8>, OptionQuery,
>;
```

### Emitting Events

```rust
// In dispatchable:
Self::deposit_event(Event::SomethingHappened { who, value });
```

### Weight Annotations

```rust
// Simple constant weight
#[pallet::weight(10_000 + T::DbWeight::get().writes(1).ref_time())]

// Using benchmarked weights (recommended for production)
#[pallet::weight(<T as Config>::WeightInfo::my_call())]
```

### Ensuring Origin

```rust
let who = ensure_signed(origin)?;    // any signed account
ensure_root(origin)?;                 // sudo/root only
ensure_none(origin)?;                 // unsigned
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `wasm32-unknown-unknown` target missing | `rustup target add wasm32-unknown-unknown` |
| `error: linker cc not found` | Install `build-essential` (Linux) or Xcode CLI tools (macOS) |
| `protoc` not found | `apt install protobuf-compiler` or `brew install protobuf` |
| Node won't start: port in use | Kill existing process or use `--port`, `--ws-port`, `--rpc-port` flags |
| State corruption on restart | Run `purge-chain --dev` to wipe dev state |
| WASM blob outdated after runtime change | `cargo build --release` to recompile |
| `BlockImportError` on multi-node | Ensure all nodes use identical chain spec and genesis |

```bash
# Check currently installed Rust targets
rustup show

# Verbose build output for debugging
cargo build --release -v 2>&1 | grep error

# Check node version
./target/release/solochain-template-node --version
```
```
